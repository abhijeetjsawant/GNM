"""Frame-accurate video decoding and MediaPipe facial-performance capture.

The capture schema deliberately keeps source timing, raw detector observations,
and detector availability separate.  FFprobe supplies integer presentation
timestamps and FFmpeg emits exactly one RGB frame for every probed timestamp;
any disagreement is rejected instead of silently manufacturing a constant FPS.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import mediapipe as mp
import numpy as np

from .errors import AutoAnimError
from .image import validate_model
from .serialization import write_npz


CAPTURE_SCHEMA_VERSION = "autoanim.capture.v1"
LANDMARK_COUNT = 478
MONOCULAR_SCALE_CAVEAT = (
    "MediaPipe's monocular facial transform is relative to its canonical face; "
    "translation has approximate canonical-model scale, not calibrated person-specific metric scale."
)

# The order published by the pinned MediaPipe Face Landmarker model.  Keeping a
# declared order makes an unexpected model/schema change a typed error rather
# than a silent column permutation.
MEDIAPIPE_BLENDSHAPE_NAMES = (
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
)


def _readonly_array(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise AutoAnimError(
            "DEPENDENCY_MISSING", f"Could not run {executable}: {exc}"
        ) from exc
    return result.stdout.splitlines()[0].strip()


@dataclass(frozen=True, slots=True)
class VideoDecodeLimits:
    max_file_bytes: int = 2 * 1024 * 1024 * 1024
    max_pixels_per_frame: int = 16_000_000
    max_frames: int = 18_000

    def __post_init__(self) -> None:
        if min(self.max_file_bytes, self.max_pixels_per_frame, self.max_frames) <= 0:
            raise ValueError("Video decode limits must be positive")


@dataclass(frozen=True, slots=True)
class VideoProbe:
    path: Path
    width: int
    height: int
    codec: str
    time_base_numerator: int
    time_base_denominator: int
    source_pts: np.ndarray
    timestamps_seconds: np.ndarray
    mediapipe_timestamps_ms: np.ndarray
    display_rotation_degrees: int
    ffprobe_command: tuple[str, ...]

    def __post_init__(self) -> None:
        pts = _readonly_array(self.source_pts, np.int64)
        timestamps = _readonly_array(self.timestamps_seconds, np.float64)
        timestamps_ms = _readonly_array(self.mediapipe_timestamps_ms, np.int64)
        object.__setattr__(self, "source_pts", pts)
        object.__setattr__(self, "timestamps_seconds", timestamps)
        object.__setattr__(self, "mediapipe_timestamps_ms", timestamps_ms)
        count = len(pts)
        if self.width <= 0 or self.height <= 0 or count == 0:
            raise ValueError("Video probe has invalid dimensions or no frames")
        if self.time_base_numerator <= 0 or self.time_base_denominator <= 0:
            raise ValueError("Video time base must be positive")
        if timestamps.shape != (count,) or timestamps_ms.shape != (count,):
            raise ValueError("Video timing arrays have inconsistent lengths")
        if count > 1 and (
            np.any(np.diff(pts) <= 0)
            or np.any(np.diff(timestamps) <= 0)
            or np.any(np.diff(timestamps_ms) <= 0)
        ):
            raise ValueError("Video timestamps must be strictly increasing")
        if timestamps[0] != 0.0 or timestamps_ms[0] != 0:
            raise ValueError("Normalized video timestamps must start at zero")

    @property
    def frame_count(self) -> int:
        return len(self.source_pts)

    @property
    def time_base(self) -> Fraction:
        return Fraction(self.time_base_numerator, self.time_base_denominator)


@dataclass(frozen=True, slots=True)
class DecodedVideoFrame:
    frame_index: int
    source_pts: int
    timestamp_seconds: float
    mediapipe_timestamp_ms: int
    rgb: np.ndarray


@dataclass(frozen=True, slots=True)
class CaptureProvenance:
    source_name: str
    source_sha256: str
    source_bytes: int
    model_name: str
    model_sha256: str
    mediapipe_version: str
    ffprobe_version: str
    ffmpeg_version: str
    codec: str
    time_base_numerator: int
    time_base_denominator: int
    source_start_pts: int
    display_rotation_degrees: int
    ffprobe_command: tuple[str, ...]
    ffmpeg_command: tuple[str, ...]
    caveats: tuple[str, ...] = (MONOCULAR_SCALE_CAVEAT,)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
            "mediapipe_version": self.mediapipe_version,
            "ffprobe_version": self.ffprobe_version,
            "ffmpeg_version": self.ffmpeg_version,
            "codec": self.codec,
            "time_base": [self.time_base_numerator, self.time_base_denominator],
            "source_start_pts": self.source_start_pts,
            "display_rotation_degrees": self.display_rotation_degrees,
            "ffprobe_command": list(self.ffprobe_command),
            "ffmpeg_command": list(self.ffmpeg_command),
            "caveats": list(self.caveats),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CaptureProvenance:
        time_base = value["time_base"]
        return cls(
            source_name=str(value["source_name"]),
            source_sha256=str(value["source_sha256"]),
            source_bytes=int(value["source_bytes"]),
            model_name=str(value["model_name"]),
            model_sha256=str(value["model_sha256"]),
            mediapipe_version=str(value["mediapipe_version"]),
            ffprobe_version=str(value["ffprobe_version"]),
            ffmpeg_version=str(value["ffmpeg_version"]),
            codec=str(value["codec"]),
            time_base_numerator=int(time_base[0]),
            time_base_denominator=int(time_base[1]),
            source_start_pts=int(value["source_start_pts"]),
            display_rotation_degrees=int(value["display_rotation_degrees"]),
            ffprobe_command=tuple(str(item) for item in value["ffprobe_command"]),
            ffmpeg_command=tuple(str(item) for item in value["ffmpeg_command"]),
            caveats=tuple(str(item) for item in value["caveats"]),
        )


@dataclass(frozen=True, slots=True)
class CaptureTrack:
    """Immutable, normalized raw performance observations for one face."""

    source_pts: np.ndarray
    timestamps_seconds: np.ndarray
    mediapipe_timestamps_ms: np.ndarray
    detected: np.ndarray
    landmarks_xyz: np.ndarray
    landmark_visibility: np.ndarray
    landmark_presence: np.ndarray
    blendshape_names: tuple[str, ...]
    blendshape_scores: np.ndarray
    facial_transforms: np.ndarray
    face_confidence: np.ndarray
    tracking_quality: np.ndarray
    width: int
    height: int
    provenance: CaptureProvenance
    schema_version: str = CAPTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        arrays = {
            "source_pts": (self.source_pts, np.int64),
            "timestamps_seconds": (self.timestamps_seconds, np.float64),
            "mediapipe_timestamps_ms": (self.mediapipe_timestamps_ms, np.int64),
            "detected": (self.detected, np.bool_),
            "landmarks_xyz": (self.landmarks_xyz, np.float32),
            "landmark_visibility": (self.landmark_visibility, np.float32),
            "landmark_presence": (self.landmark_presence, np.float32),
            "blendshape_scores": (self.blendshape_scores, np.float32),
            "facial_transforms": (self.facial_transforms, np.float32),
            "face_confidence": (self.face_confidence, np.float32),
            "tracking_quality": (self.tracking_quality, np.float32),
        }
        for name, (value, dtype) in arrays.items():
            object.__setattr__(self, name, _readonly_array(value, dtype))
        count = len(self.source_pts)
        names = tuple(self.blendshape_names)
        object.__setattr__(self, "blendshape_names", names)
        if self.schema_version != CAPTURE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported capture schema: {self.schema_version}")
        if self.width <= 0 or self.height <= 0 or count == 0:
            raise ValueError("Capture track has invalid dimensions or no frames")
        expected = {
            "timestamps_seconds": (count,),
            "mediapipe_timestamps_ms": (count,),
            "detected": (count,),
            "landmarks_xyz": (count, LANDMARK_COUNT, 3),
            "landmark_visibility": (count, LANDMARK_COUNT),
            "landmark_presence": (count, LANDMARK_COUNT),
            "blendshape_scores": (count, len(names)),
            "facial_transforms": (count, 4, 4),
            "face_confidence": (count,),
            "tracking_quality": (count,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"Capture {name} must have shape {shape}, got {getattr(self, name).shape}")
        if len(set(names)) != len(names):
            raise ValueError("Blendshape names must be unique")
        if count > 1 and (
            np.any(np.diff(self.source_pts) <= 0)
            or np.any(np.diff(self.timestamps_seconds) <= 0)
            or np.any(np.diff(self.mediapipe_timestamps_ms) <= 0)
        ):
            raise ValueError("Capture timestamps must be strictly increasing")
        if self.timestamps_seconds[0] != 0.0 or self.mediapipe_timestamps_ms[0] != 0:
            raise ValueError("Capture timestamps must start at zero")
        if not np.isfinite(self.blendshape_scores).all():
            raise ValueError("Blendshape scores must be finite")
        if np.any((self.blendshape_scores < 0) | (self.blendshape_scores > 1)):
            raise ValueError("Blendshape scores must lie in [0,1]")
        if not np.isfinite(self.tracking_quality).all() or np.any(
            (self.tracking_quality < 0) | (self.tracking_quality > 1)
        ):
            raise ValueError("Tracking quality must lie in [0,1]")
        for name in ("landmark_visibility", "landmark_presence", "face_confidence"):
            optional = getattr(self, name)
            available = optional[np.isfinite(optional)]
            if np.any((available < 0) | (available > 1)):
                raise ValueError(f"Available {name} values must lie in [0,1]")
        present = self.detected
        if not (
            np.isfinite(self.landmarks_xyz[present]).all()
            and np.isfinite(self.facial_transforms[present]).all()
        ):
            raise ValueError("Detected frames must contain finite landmarks and transforms")
        if np.any(np.isfinite(self.landmarks_xyz[~present])):
            raise ValueError("Undetected frames must use NaN landmark sentinels")
        if np.any(self.tracking_quality[~present] != 0):
            raise ValueError("Undetected frames must have zero tracking quality")

    @property
    def frame_count(self) -> int:
        return len(self.source_pts)

    @property
    def duration_seconds(self) -> float:
        return float(self.timestamps_seconds[-1])


def _parse_fraction(value: object, *, field: str) -> Fraction:
    try:
        fraction = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise AutoAnimError("MEDIA_INVALID", f"Invalid video {field}: {value!r}") from exc
    if fraction <= 0:
        raise AutoAnimError("MEDIA_INVALID", f"Video {field} must be positive")
    return fraction


def probe_video(
    path: str | Path,
    *,
    ffprobe_bin: str = "ffprobe",
    limits: VideoDecodeLimits = VideoDecodeLimits(),
) -> VideoProbe:
    """Read exact display-order frame PTS and validate MediaPipe's clock."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise AutoAnimError("INPUT_INVALID", f"Video input does not exist: {source}")
    size = source.stat().st_size
    if size <= 0 or size > limits.max_file_bytes:
        raise AutoAnimError("LIMIT_EXCEEDED", "Video input exceeds the configured file-size limit")
    command = (
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,time_base,codec_name:stream_side_data=rotation:frame=best_effort_timestamp",
        "-show_streams",
        "-show_frames",
        "-of",
        "json",
        str(source),
    )
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=120
        )
        payload = json.loads(result.stdout)
    except FileNotFoundError as exc:
        raise AutoAnimError("DEPENDENCY_MISSING", f"FFprobe is unavailable: {ffprobe_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AutoAnimError("LIMIT_EXCEEDED", "FFprobe timed out while inspecting the video") from exc
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        stderr = getattr(exc, "stderr", "")
        raise AutoAnimError("MEDIA_INVALID", f"FFprobe could not inspect the video: {stderr}") from exc
    streams = payload.get("streams", [])
    frames = payload.get("frames", [])
    if len(streams) != 1 or not frames:
        raise AutoAnimError("MEDIA_INVALID", "Video must contain a decodable primary video stream")
    stream = streams[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
        codec = str(stream.get("codec_name", "unknown"))
        time_base = _parse_fraction(stream["time_base"], field="time_base")
        pts = np.asarray([int(frame["best_effort_timestamp"]) for frame in frames], dtype=np.int64)
    except (KeyError, TypeError, ValueError) as exc:
        raise AutoAnimError("MEDIA_INVALID", "Video is missing exact frame PTS or dimensions") from exc
    rotation = 0
    for item in stream.get("side_data_list", []):
        if "rotation" in item:
            rotation = int(round(float(item["rotation"]))) % 360
            break
    if rotation in {90, 270}:
        width, height = height, width
    if width <= 0 or height <= 0 or width * height > limits.max_pixels_per_frame:
        raise AutoAnimError("LIMIT_EXCEEDED", "Video frame dimensions exceed configured limits")
    if len(pts) > limits.max_frames:
        raise AutoAnimError("LIMIT_EXCEEDED", "Video contains too many frames")
    if len(pts) > 1 and np.any(np.diff(pts) <= 0):
        raise AutoAnimError(
            "MEDIA_INVALID",
            "Display-order video PTS are not strictly increasing; exact capture would be ambiguous",
        )
    normalized = pts - pts[0]
    timestamps = np.asarray(
        [float(Fraction(int(value)) * time_base) for value in normalized], dtype=np.float64
    )
    timestamps_ms = np.asarray(
        [int(round(Fraction(int(value)) * time_base * 1000)) for value in normalized],
        dtype=np.int64,
    )
    if len(timestamps_ms) > 1 and np.any(np.diff(timestamps_ms) <= 0):
        raise AutoAnimError(
            "MEDIA_INVALID",
            "Frame PTS cannot be represented as strictly increasing integer milliseconds "
            "required by MediaPipe VIDEO mode",
        )
    return VideoProbe(
        path=source,
        width=width,
        height=height,
        codec=codec,
        time_base_numerator=time_base.numerator,
        time_base_denominator=time_base.denominator,
        source_pts=pts,
        timestamps_seconds=timestamps,
        mediapipe_timestamps_ms=timestamps_ms,
        display_rotation_degrees=rotation,
        ffprobe_command=command,
    )


def ffmpeg_decode_command(probe: VideoProbe, ffmpeg_bin: str = "ffmpeg") -> tuple[str, ...]:
    return (
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(probe.path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    )


def _read_exact(handle: Any, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@contextmanager
def decoded_video_frames(
    probe: VideoProbe,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> Iterator[Iterator[DecodedVideoFrame]]:
    """Yield RGB frames while enforcing a one-to-one frame/PTS contract."""

    command = ffmpeg_decode_command(probe, ffmpeg_bin)
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise AutoAnimError("DEPENDENCY_MISSING", f"FFmpeg is unavailable: {ffmpeg_bin}") from exc
    assert process.stdout is not None
    assert process.stderr is not None
    completed = False

    def iterator() -> Iterator[DecodedVideoFrame]:
        nonlocal completed
        byte_count = probe.width * probe.height * 3
        for index in range(probe.frame_count):
            payload = _read_exact(process.stdout, byte_count)
            if len(payload) != byte_count:
                stderr = process.stderr.read().decode("utf-8", errors="replace")
                raise AutoAnimError(
                    "MEDIA_INVALID",
                    f"FFmpeg decoded {index} of {probe.frame_count} probed frames: {stderr}",
                )
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape(probe.height, probe.width, 3)
            rgb.setflags(write=False)
            yield DecodedVideoFrame(
                frame_index=index,
                source_pts=int(probe.source_pts[index]),
                timestamp_seconds=float(probe.timestamps_seconds[index]),
                mediapipe_timestamp_ms=int(probe.mediapipe_timestamps_ms[index]),
                rgb=rgb,
            )
        extra = process.stdout.read(1)
        if extra:
            raise AutoAnimError(
                "MEDIA_INVALID", "FFmpeg produced more frames than FFprobe reported"
            )
        return_code = process.wait(timeout=30)
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if return_code != 0:
            raise AutoAnimError("MEDIA_INVALID", f"FFmpeg decode failed: {stderr}")
        completed = True

    try:
        yield iterator()
    finally:
        if not completed and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def _optional_landmark_value(point: Any, name: str) -> float:
    value = getattr(point, name, None)
    return np.nan if value is None else float(value)


def capture_video(
    video_path: str | Path,
    model_path: str | Path,
    *,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
    limits: VideoDecodeLimits = VideoDecodeLimits(),
    min_face_detection_confidence: float = 0.5,
    min_face_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> CaptureTrack:
    """Capture one face in MediaPipe Face Landmarker ``VIDEO`` mode."""

    thresholds = (
        min_face_detection_confidence,
        min_face_presence_confidence,
        min_tracking_confidence,
    )
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise AutoAnimError("INPUT_INVALID", "MediaPipe confidence thresholds must lie in [0,1]")
    model = validate_model(model_path).resolve()
    probe = probe_video(video_path, ffprobe_bin=ffprobe_bin, limits=limits)
    count = probe.frame_count
    landmarks = np.full((count, LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    visibility = np.full((count, LANDMARK_COUNT), np.nan, dtype=np.float32)
    presence = np.full((count, LANDMARK_COUNT), np.nan, dtype=np.float32)
    scores = np.zeros((count, len(MEDIAPIPE_BLENDSHAPE_NAMES)), dtype=np.float32)
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], count, axis=0)
    detected = np.zeros(count, dtype=bool)
    face_confidence = np.full(count, np.nan, dtype=np.float32)
    tracking_quality = np.zeros(count, dtype=np.float32)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=float(min_face_detection_confidence),
        min_face_presence_confidence=float(min_face_presence_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    name_to_index = {name: index for index, name in enumerate(MEDIAPIPE_BLENDSHAPE_NAMES)}
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        with decoded_video_frames(probe, ffmpeg_bin=ffmpeg_bin) as frames:
            for frame in frames:
                media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.rgb)
                result = detector.detect_for_video(media_image, frame.mediapipe_timestamp_ms)
                if not result.face_landmarks:
                    continue
                if not result.face_blendshapes or not result.facial_transformation_matrixes:
                    raise AutoAnimError(
                        "INTERNAL_ERROR",
                        "MediaPipe returned a face without requested blendshapes or transform",
                    )
                points = result.face_landmarks[0]
                if len(points) != LANDMARK_COUNT:
                    raise AutoAnimError(
                        "INTERNAL_ERROR",
                        f"MediaPipe landmark schema changed: expected {LANDMARK_COUNT}, got {len(points)}",
                    )
                categories = result.face_blendshapes[0]
                category_names = tuple(category.category_name for category in categories)
                if len(category_names) != len(MEDIAPIPE_BLENDSHAPE_NAMES) or set(
                    category_names
                ) != set(MEDIAPIPE_BLENDSHAPE_NAMES):
                    raise AutoAnimError("INTERNAL_ERROR", "MediaPipe blendshape schema changed")
                index = frame.frame_index
                landmarks[index] = np.asarray(
                    [(point.x, point.y, point.z) for point in points], dtype=np.float32
                )
                visibility[index] = np.asarray(
                    [_optional_landmark_value(point, "visibility") for point in points],
                    dtype=np.float32,
                )
                presence[index] = np.asarray(
                    [_optional_landmark_value(point, "presence") for point in points],
                    dtype=np.float32,
                )
                for category in categories:
                    scores[index, name_to_index[category.category_name]] = np.float32(
                        np.clip(category.score, 0.0, 1.0)
                    )
                matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise AutoAnimError("INTERNAL_ERROR", "MediaPipe returned an invalid facial transform")
                transforms[index] = matrix
                detected[index] = True
                available_presence = presence[index][np.isfinite(presence[index])]
                available_visibility = visibility[index][np.isfinite(visibility[index])]
                confidence_values = np.concatenate((available_presence, available_visibility))
                if len(confidence_values):
                    face_confidence[index] = np.float32(np.median(confidence_values))
                normalized_xy = landmarks[index, :, :2]
                in_bounds = np.mean(
                    (normalized_xy[:, 0] >= -0.05)
                    & (normalized_xy[:, 0] <= 1.05)
                    & (normalized_xy[:, 1] >= -0.05)
                    & (normalized_xy[:, 1] <= 1.05)
                )
                tracking_quality[index] = np.float32(np.clip(in_bounds, 0.0, 1.0))
    ffmpeg_command = ffmpeg_decode_command(probe, ffmpeg_bin)
    provenance = CaptureProvenance(
        source_name=probe.path.name,
        source_sha256=_file_sha256(probe.path),
        source_bytes=probe.path.stat().st_size,
        model_name=model.name,
        model_sha256=_file_sha256(model),
        mediapipe_version=mp.__version__,
        ffprobe_version=_tool_version(ffprobe_bin),
        ffmpeg_version=_tool_version(ffmpeg_bin),
        codec=probe.codec,
        time_base_numerator=probe.time_base_numerator,
        time_base_denominator=probe.time_base_denominator,
        source_start_pts=int(probe.source_pts[0]),
        display_rotation_degrees=probe.display_rotation_degrees,
        ffprobe_command=probe.ffprobe_command,
        ffmpeg_command=ffmpeg_command,
    )
    return CaptureTrack(
        source_pts=probe.source_pts,
        timestamps_seconds=probe.timestamps_seconds,
        mediapipe_timestamps_ms=probe.mediapipe_timestamps_ms,
        detected=detected,
        landmarks_xyz=landmarks,
        landmark_visibility=visibility,
        landmark_presence=presence,
        blendshape_names=MEDIAPIPE_BLENDSHAPE_NAMES,
        blendshape_scores=scores,
        facial_transforms=transforms,
        face_confidence=face_confidence,
        tracking_quality=tracking_quality,
        width=probe.width,
        height=probe.height,
        provenance=provenance,
    )


def write_capture_npz(path: str | Path, track: CaptureTrack) -> Path:
    provenance = json.dumps(
        track.provenance.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return write_npz(
        path,
        schema_version=np.asarray(track.schema_version),
        source_pts=track.source_pts,
        timestamps_seconds=track.timestamps_seconds,
        mediapipe_timestamps_ms=track.mediapipe_timestamps_ms,
        detected=track.detected,
        landmarks_xyz=track.landmarks_xyz,
        landmark_visibility=track.landmark_visibility,
        landmark_presence=track.landmark_presence,
        blendshape_names=np.asarray(track.blendshape_names),
        blendshape_scores=track.blendshape_scores,
        facial_transforms=track.facial_transforms,
        face_confidence=track.face_confidence,
        tracking_quality=track.tracking_quality,
        frame_size=np.asarray((track.width, track.height), dtype=np.int32),
        provenance_json=np.asarray(provenance),
    )


def load_capture_npz(path: str | Path) -> CaptureTrack:
    try:
        with np.load(Path(path), allow_pickle=False) as values:
            width, height = values["frame_size"].tolist()
            provenance = CaptureProvenance.from_dict(
                json.loads(str(values["provenance_json"].item()))
            )
            return CaptureTrack(
                schema_version=str(values["schema_version"].item()),
                source_pts=values["source_pts"],
                timestamps_seconds=values["timestamps_seconds"],
                mediapipe_timestamps_ms=values["mediapipe_timestamps_ms"],
                detected=values["detected"],
                landmarks_xyz=values["landmarks_xyz"],
                landmark_visibility=values["landmark_visibility"],
                landmark_presence=values["landmark_presence"],
                blendshape_names=tuple(str(item) for item in values["blendshape_names"].tolist()),
                blendshape_scores=values["blendshape_scores"],
                facial_transforms=values["facial_transforms"],
                face_confidence=values["face_confidence"],
                tracking_quality=values["tracking_quality"],
                width=int(width),
                height=int(height),
                provenance=provenance,
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise AutoAnimError("MEDIA_INVALID", f"Invalid capture NPZ: {exc}") from exc


def _nullable(value: np.ndarray) -> Any:
    if value.ndim == 0:
        scalar = value.item()
        if isinstance(scalar, float) and not np.isfinite(scalar):
            return None
        return scalar
    return [_nullable(item) for item in value]


def write_capture_jsonl(path: str | Path, track: CaptureTrack) -> Path:
    """Write a self-describing JSONL stream with null optional confidence."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        metadata = {
            "recordType": "metadata",
            "schemaVersion": track.schema_version,
            "frameCount": track.frame_count,
            "frameSize": [track.width, track.height],
            "blendshapeNames": list(track.blendshape_names),
            "provenance": track.provenance.as_dict(),
        }
        handle.write(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n")
        for index in range(track.frame_count):
            is_detected = bool(track.detected[index])
            record = {
                "recordType": "frame",
                "frameIndex": index,
                "sourcePTS": int(track.source_pts[index]),
                "timestampSeconds": float(track.timestamps_seconds[index]),
                "mediapipeTimestampMs": int(track.mediapipe_timestamps_ms[index]),
                "detected": is_detected,
                "landmarksXYZ": (
                    _nullable(track.landmarks_xyz[index]) if is_detected else None
                ),
                "landmarkVisibility": (
                    _nullable(track.landmark_visibility[index]) if is_detected else None
                ),
                "landmarkPresence": (
                    _nullable(track.landmark_presence[index]) if is_detected else None
                ),
                "blendshapeScores": track.blendshape_scores[index].tolist(),
                "facialTransform": (
                    track.facial_transforms[index].tolist() if is_detected else None
                ),
                "faceConfidence": _nullable(track.face_confidence[index]),
                "trackingQuality": float(track.tracking_quality[index]),
            }
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, destination)
    except Exception:
        handle.close()
        temporary.unlink(missing_ok=True)
        raise
    return destination


def serialize_capture(directory: str | Path, track: CaptureTrack) -> tuple[Path, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    return (
        write_capture_npz(root / "capture.npz", track),
        write_capture_jsonl(root / "capture.jsonl", track),
    )
