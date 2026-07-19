"""Native audio normalization, Rhubarb execution, and transparent emotion hints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import wave

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .errors import AutoAnimError


VALID_CUES = frozenset("XABCDEFGH")
PRIMARY_AUDIO_STREAM_SPECIFIER = "0:a:0"
EMOTIONS = frozenset(("neutral", "joy", "sad", "anger", "fear", "disgust", "surprise", "contempt"))
WORD_LISTS = {
    "joy": {"happy", "glad", "delighted", "love", "wonderful", "excited", "joy"},
    "sad": {"sad", "sorry", "grief", "lonely", "cry", "unhappy"},
    "anger": {"angry", "mad", "furious", "hate", "rage", "annoyed"},
    "fear": {"afraid", "scared", "fear", "terrified", "worried"},
    "disgust": {"disgust", "gross", "revolting", "nasty"},
    "surprise": {"wow", "surprised", "unexpected", "amazing", "astonished"},
    "contempt": {"idiot", "pathetic", "ridiculous", "worthless"},
}


@dataclass(frozen=True, slots=True)
class MouthCue:
    start: float
    end: float
    value: str

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EmotionAnalysis:
    emotion: str
    confidence: float
    validated: bool
    source: str
    features: dict[str, float]


@dataclass(frozen=True, slots=True)
class ProsodyTrack:
    """Speaker-relative, frame-aligned performance features.

    These values describe timing and emphasis. They are deliberately kept
    separate from the coarse categorical emotion result.
    """

    timestamps: np.ndarray
    rms_dbfs: np.ndarray
    energy: np.ndarray
    speech_activity: np.ndarray
    pitch_semitones: np.ndarray
    accent: np.ndarray
    phrase_id: np.ndarray


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def probe_media(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise AutoAnimError("INPUT_INVALID", f"Input does not exist: {path}")
    if path.stat().st_size > 100 * 1024 * 1024:
        raise AutoAnimError("LIMIT_EXCEEDED", "Input exceeds 100 MiB")
    try:
        result = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
        data = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise AutoAnimError(
            "DEPENDENCY_MISSING",
            "ffprobe is required and must be on PATH",
        ) from exc
    audio_streams = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"]
    if not audio_streams:
        raise AutoAnimError("MEDIA_INVALID", "Input contains no audio stream")
    duration_raw = data.get("format", {}).get("duration") or audio_streams[0].get("duration")
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError) as exc:
        raise AutoAnimError("MEDIA_INVALID", "Could not determine audio duration") from exc
    if not math.isfinite(duration) or duration <= 0 or duration > 600:
        raise AutoAnimError("LIMIT_EXCEEDED", f"Audio duration must be in (0, 600] seconds; got {duration}")
    return {"duration": duration, "streams": data["streams"], "format": data.get("format", {})}


def normalize_audio(input_path: str | Path, output_path: str | Path) -> float:
    probe_media(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-map",
        PRIMARY_AUDIO_STREAM_SPECIFIER,
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        _run(command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise AutoAnimError("MEDIA_INVALID", "ffmpeg could not decode the input audio") from exc
    with wave.open(str(output_path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != 16_000 or handle.getsampwidth() != 2:
            raise AutoAnimError("INTERNAL_ERROR", "Normalized WAV contract was not met")
        return handle.getnframes() / handle.getframerate()


def resolve_rhubarb(explicit: str | Path | None = None) -> Path:
    candidate = str(explicit) if explicit is not None else os.environ.get("RHUBARB_BIN")
    if not candidate:
        candidate = shutil.which("rhubarb")
    if not candidate or not Path(candidate).is_file():
        raise AutoAnimError(
            "DEPENDENCY_MISSING",
            "Rhubarb 1.14 is required. Set RHUBARB_BIN to its executable path.",
            {"install": "https://github.com/DanielSWolf/rhubarb-lip-sync/releases/tag/v1.14.0"},
        )
    executable = Path(candidate)
    dictionary = executable.parent / "res" / "sphinx" / "cmudict-en-us.dict"
    if not dictionary.is_file():
        raise AutoAnimError(
            "DEPENDENCY_MISSING",
            "Rhubarb's companion res/sphinx bundle is missing; install the complete release archive.",
            {"expected": str(dictionary)},
        )
    return executable


def run_rhubarb(
    wav_path: str | Path,
    output_path: str | Path,
    *,
    rhubarb_bin: str | Path | None = None,
    dialog: str | None = None,
) -> list[dict]:
    executable = resolve_rhubarb(rhubarb_bin)
    output_path = Path(output_path)
    command = [str(executable), "-f", "json", "-o", str(output_path)]
    dialog_path: Path | None = None
    try:
        if dialog:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
                handle.write(dialog)
                dialog_path = Path(handle.name)
            command.extend(["--dialogFile", str(dialog_path)])
        command.append(str(wav_path))
        _run(command)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        cues = payload.get("mouthCues")
        if not isinstance(cues, list):
            raise ValueError("mouthCues is missing")
        return cues
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = detail.splitlines()[-1] if detail else str(exc)
        raise AutoAnimError("CUE_INVALID", f"Rhubarb failed: {message}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise AutoAnimError("CUE_INVALID", f"Rhubarb failed to produce valid JSON: {exc}") from exc
    finally:
        if dialog_path is not None:
            dialog_path.unlink(missing_ok=True)


def normalize_cues(raw_cues: list[dict], duration: float) -> list[MouthCue]:
    if not math.isfinite(duration) or duration <= 0:
        raise AutoAnimError("CUE_INVALID", "Cue duration must be positive and finite")
    parsed: list[MouthCue] = []
    for raw in raw_cues:
        try:
            start = max(0.0, min(duration, float(raw["start"])))
            end = max(0.0, min(duration, float(raw["end"])))
            value = str(raw["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AutoAnimError("CUE_INVALID", f"Malformed cue: {raw}") from exc
        if value not in VALID_CUES or not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise AutoAnimError("CUE_INVALID", f"Invalid cue: {raw}")
        parsed.append(MouthCue(start, end, value))
    parsed.sort(key=lambda cue: (cue.start, cue.end))
    output: list[MouthCue] = []
    cursor = 0.0
    for cue in parsed:
        start = cue.start
        if start < cursor - 0.001:
            raise AutoAnimError("CUE_INVALID", "Rhubarb cues overlap by more than 1 ms")
        if start > cursor + 0.001:
            output.append(MouthCue(cursor, start, "X"))
        else:
            start = cursor
        if cue.end <= start:
            continue
        output.append(MouthCue(start, cue.end, cue.value))
        cursor = cue.end
    if cursor < duration:
        output.append(MouthCue(cursor, duration, "X"))
    if not output:
        output = [MouthCue(0.0, duration, "X")]
    output[0] = MouthCue(0.0, output[0].end, output[0].value)
    output[-1] = MouthCue(output[-1].start, duration, output[-1].value)
    merged: list[MouthCue] = []
    for cue in output:
        if merged and merged[-1].value == cue.value:
            merged[-1] = MouthCue(merged[-1].start, cue.end, cue.value)
        else:
            merged.append(cue)
    return merged


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise AutoAnimError("MEDIA_INVALID", "Expected mono 16-bit PCM WAV")
        rate = handle.getframerate()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0, rate


def _estimate_f0(frame: np.ndarray, rate: int) -> float:
    """Small YIN-style estimator used only for relative prosody."""

    centered = np.asarray(frame, dtype=np.float64) - float(np.mean(frame))
    if float(np.sqrt(np.mean(centered * centered))) < 1e-4:
        return 0.0
    min_lag = max(1, int(rate / 400))
    max_lag = min(int(rate / 70), len(centered) // 2)
    if max_lag <= min_lag:
        return 0.0
    difference = np.asarray(
        [np.sum((centered[:-lag] - centered[lag:]) ** 2) for lag in range(1, max_lag + 1)],
        dtype=np.float64,
    )
    cumulative = np.cumsum(difference)
    normalized = np.ones(max_lag + 1, dtype=np.float64)
    normalized[1:] = difference * np.arange(1, max_lag + 1) / np.maximum(cumulative, 1e-12)
    for candidate in range(min_lag, max_lag):
        if normalized[candidate] < 0.20:
            while candidate + 1 <= max_lag and normalized[candidate + 1] < normalized[candidate]:
                candidate += 1
            return float(rate / candidate)
    return 0.0


def extract_prosody(
    wav_path: str | Path,
    cues: list[MouthCue],
    fps: int,
) -> ProsodyTrack:
    """Extract a deterministic, speaker-relative performance timeline."""

    if not cues:
        raise AutoAnimError("CUE_INVALID", "At least one normalized mouth cue is required")
    if not 12 <= fps <= 60:
        raise AutoAnimError("INPUT_INVALID", "FPS must be in [12, 60]")
    samples, rate = read_wav(wav_path)
    duration = len(samples) / rate
    frame_count = int(math.ceil(duration * fps))
    timestamps = np.arange(frame_count, dtype=np.float32) / np.float32(fps)
    window = max(32, int(round(0.040 * rate)))
    half = window // 2
    padded = np.pad(samples, (half, half), mode="constant")

    rms = np.empty(frame_count, dtype=np.float32)
    f0 = np.zeros(frame_count, dtype=np.float32)
    for index, timestamp in enumerate(timestamps):
        center = min(len(samples) - 1, max(0, int(round(float(timestamp) * rate))))
        frame = padded[center : center + window]
        rms[index] = np.float32(np.sqrt(np.mean(frame * frame) + 1e-16))
    db = 20.0 * np.log10(np.maximum(rms, 1e-8))

    ends = np.asarray([cue.end for cue in cues], dtype=np.float64)
    cue_indices = np.minimum(
        np.searchsorted(ends, timestamps.astype(np.float64), side="right"),
        len(cues) - 1,
    )
    cue_active = np.asarray([cues[int(index)].value != "X" for index in cue_indices], dtype=bool)
    noise_floor = float(np.percentile(db, 10))
    acoustic_active = (db >= -60.0) & (db >= noise_floor + 9.0)
    active = cue_active & acoustic_active
    if not np.any(active):
        # Cues remain the authoritative speech mask for very quiet speech.
        active = cue_active

    active_db = db[active]
    if active_db.size:
        low, high = np.percentile(active_db, [10, 90])
        span = max(6.0, float(high - low))
        energy = np.clip((db - float(low)) / span, 0.0, 1.0)
    else:
        energy = np.zeros_like(db)
    energy *= cue_active.astype(np.float32)
    energy = gaussian_filter1d(energy.astype(np.float64), sigma=max(0.5, 0.055 * fps), mode="nearest")
    energy = np.clip(energy, 0.0, 1.0).astype(np.float32)

    voiced_candidates = active & (energy >= 0.10)
    for index in np.flatnonzero(voiced_candidates):
        center = min(len(samples) - 1, max(0, int(round(float(timestamps[index]) * rate))))
        f0[index] = np.float32(_estimate_f0(padded[center : center + window], rate))
    voiced = f0 > 0
    pitch_semitones = np.zeros(frame_count, dtype=np.float32)
    if np.count_nonzero(voiced) >= 3:
        median_f0 = float(np.median(f0[voiced]))
        pitch_semitones[voiced] = 12.0 * np.log2(f0[voiced] / median_f0)
        pitch_semitones = np.clip(pitch_semitones, -12.0, 12.0)

    # Keep Rhubarb authoritative about which phonetic interval is active, but
    # let the waveform settle the performance in real acoustic pauses and at
    # clip boundaries. This avoids a final non-neutral pose when a coarse cue
    # extends through trailing room tone.
    speech_activity = gaussian_filter1d(
        active.astype(np.float64), sigma=max(0.35, 0.025 * fps), mode="nearest"
    )
    speech_activity = np.clip(speech_activity, 0.0, 1.0).astype(np.float32)
    pitch_lift = np.clip(pitch_semitones / 6.0, 0.0, 1.0)
    accent = 0.78 * energy + 0.22 * pitch_lift
    accent = gaussian_filter1d(accent.astype(np.float64), sigma=max(0.5, 0.090 * fps), mode="nearest")
    accent = np.clip(accent, 0.0, 1.0).astype(np.float32)

    phrase_for_cue = np.zeros(len(cues), dtype=np.int32)
    phrase = -1
    after_long_pause = True
    for index, cue in enumerate(cues):
        if cue.value != "X":
            if after_long_pause:
                phrase += 1
                after_long_pause = False
            phrase_for_cue[index] = max(0, phrase)
        else:
            phrase_for_cue[index] = max(0, phrase)
            if cue.end - cue.start >= 0.20:
                after_long_pause = True
    phrase_id = phrase_for_cue[cue_indices].astype(np.int32)

    arrays = (timestamps, db, energy, speech_activity, pitch_semitones, accent)
    if any(not np.isfinite(array).all() for array in arrays):
        raise AutoAnimError("INTERNAL_ERROR", "Prosody extraction produced nonfinite values")
    return ProsodyTrack(
        timestamps=timestamps,
        rms_dbfs=db.astype(np.float32),
        energy=energy,
        speech_activity=speech_activity,
        pitch_semitones=pitch_semitones.astype(np.float32),
        accent=accent,
        phrase_id=phrase_id,
    )


def _audio_features(wav_path: str | Path, cues: list[MouthCue]) -> dict[str, float]:
    samples, rate = read_wav(wav_path)
    frame_size = int(round(0.030 * rate))
    hop = int(round(0.010 * rate))
    if len(samples) < frame_size:
        samples = np.pad(samples, (0, frame_size - len(samples)))
    starts = np.arange(0, len(samples) - frame_size + 1, hop)
    frames = np.stack([samples[start : start + frame_size] for start in starts])
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-16)
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    active = (db >= -60.0) & (db >= np.percentile(db, 10) + 12.0)
    if not np.any(active):
        raise AutoAnimError("AUDIO_SILENT", "No active speech-like audio was detected")
    pitches: list[float] = []
    for frame in frames[active]:
        pitch = _estimate_f0(frame, rate)
        if pitch > 0:
            pitches.append(pitch)
    f0_cv = float(np.std(pitches) / np.mean(pitches)) if len(pitches) >= 5 else 0.0
    duration = len(samples) / rate
    cue_rate = sum(cue.value != "X" for cue in cues) / duration
    return {
        "rms_dbfs": float(np.mean(db[active])),
        "f0_median_hz": float(np.median(pitches)) if pitches else 0.0,
        "f0_cv": f0_cv,
        "cues_per_second": float(cue_rate),
    }


def analyze_emotion(
    wav_path: str | Path,
    cues: list[MouthCue],
    *,
    manual: str = "auto",
    dialog: str | None = None,
) -> EmotionAnalysis:
    features = _audio_features(wav_path, cues)
    if manual != "auto":
        if manual not in EMOTIONS:
            raise AutoAnimError("INPUT_INVALID", f"Unknown emotion: {manual}")
        return EmotionAnalysis(manual, 1.0, True, "manual", features)
    if dialog:
        tokens = re.findall(r"[a-z]+", dialog.lower().replace("'", ""))
        counts = {emotion: sum(token in words for token in tokens) for emotion, words in WORD_LISTS.items()}
        maximum = max(counts.values(), default=0)
        winners = [emotion for emotion, count in counts.items() if count == maximum and count > 0]
        if len(winners) == 1:
            confidence = min(0.85, 0.55 + 0.10 * maximum)
            return EmotionAnalysis(winners[0], confidence, confidence >= 0.65, "dialog_heuristic", features)
        if len(winners) > 1:
            return EmotionAnalysis("neutral", 0.40, False, "dialog_tie", features)
    arousal = float(
        np.clip(
            0.45 * (features["rms_dbfs"] + 45.0) / 30.0
            + 0.35 * features["f0_cv"] / 0.35
            + 0.20 * features["cues_per_second"] / 8.0,
            0.0,
            1.0,
        )
    )
    features["arousal"] = arousal
    rms_dbfs = features["rms_dbfs"]
    f0_median = features["f0_median_hz"]
    f0_cv = features["f0_cv"]
    if rms_dbfs > -33 and f0_median > 280:
        return EmotionAnalysis("anger", 0.62, False, "audio_heuristic", features)
    if f0_median > 150 and f0_cv > 0.24:
        return EmotionAnalysis("surprise", 0.58, False, "audio_heuristic", features)
    if rms_dbfs < -35 and 0 < f0_median < 180:
        return EmotionAnalysis("sad", 0.58, False, "audio_heuristic", features)
    if -36 < rms_dbfs < -30 and f0_median > 210 and f0_cv < 0.22:
        return EmotionAnalysis("joy", 0.55, False, "audio_heuristic", features)
    if rms_dbfs > -31 and f0_median > 150 and f0_cv > 0.15:
        return EmotionAnalysis("fear", 0.52, False, "audio_heuristic", features)
    return EmotionAnalysis("neutral", 0.50, False, "audio_heuristic", features)
