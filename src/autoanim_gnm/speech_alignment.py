"""Stable supplied-transcript preparation shared by app and CUDA worker."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
import unicodedata


MAX_TRANSCRIPT_CHARACTERS = 80_000
MAX_ALIGNMENT_JSON_BYTES = 8_000_000
MAX_ALIGNED_WORDS = 20_000


@dataclass(frozen=True, slots=True)
class AlignedWord:
    """One validated word interval produced by a local ASR engine."""

    word: str
    start_seconds: float
    end_seconds: float
    confidence: float


@dataclass(frozen=True, slots=True)
class SpeechAlignment:
    """Readable transcript plus model-ready, word-level timing evidence."""

    transcript: str
    duration_seconds: float
    words: tuple[AlignedWord, ...]
    engine: str
    model_revision: str


def normalize_supplied_transcript(value: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_TRANSCRIPT_CHARACTERS:
        raise ValueError("Supplied transcript is missing or too long")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise ValueError("Supplied transcript is empty")
    if "\x00" in normalized:
        raise ValueError("Supplied transcript contains NUL")
    return normalized


def write_mfa_lab(path: str | Path, transcript: str) -> str:
    output = Path(path)
    if output.suffix != ".lab" or output.is_symlink():
        raise ValueError("MFA transcript target must be a regular .lab path")
    normalized = normalize_supplied_transcript(transcript)
    output.write_text(normalized + "\n", encoding="utf-8")
    return normalized


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _gesturelsm_word_mark(value: str) -> str:
    """Match MFA-style lexical labels used by GestureLSM's learned vocabulary."""

    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"^[^\w]+|[^\w]+$", "", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized or "\x00" in normalized or '"' in normalized:
        raise ValueError("Aligned word cannot be converted to a safe lexical label")
    return normalized


def load_speaktype_parakeet_alignment(path: str | Path) -> SpeechAlignment:
    """Load FluidAudio Parakeet JSON emitted by SpeakType's local model lane."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("SpeakType alignment JSON must be a regular file")
    payload = source.read_bytes()
    if not payload or len(payload) > MAX_ALIGNMENT_JSON_BYTES:
        raise ValueError("SpeakType alignment JSON is empty or too large")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("SpeakType alignment JSON is invalid") from error
    if not isinstance(document, dict):
        raise ValueError("SpeakType alignment JSON must contain an object")
    if document.get("mode") != "batch":
        raise ValueError("SpeakType alignment must come from batch transcription")
    model = document.get("modelVersion")
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", model):
        raise ValueError("SpeakType model version is invalid")
    transcript = normalize_supplied_transcript(document.get("text"))
    duration = _finite_number(document.get("durationSeconds"), "durationSeconds")
    if not 0.0 < duration <= 24 * 60 * 60:
        raise ValueError("SpeakType alignment duration is outside limits")
    raw_words = document.get("wordTimings")
    if (
        not isinstance(raw_words, list)
        or not 1 <= len(raw_words) <= MAX_ALIGNED_WORDS
    ):
        raise ValueError("SpeakType alignment has no usable word timings")

    words: list[AlignedWord] = []
    previous_end = 0.0
    for index, raw_word in enumerate(raw_words):
        if not isinstance(raw_word, dict):
            raise ValueError(f"wordTimings[{index}] must be an object")
        mark = raw_word.get("word")
        if not isinstance(mark, str) or len(mark) > 256:
            raise ValueError(f"wordTimings[{index}].word is invalid")
        start = _finite_number(raw_word.get("startTime"), f"wordTimings[{index}].startTime")
        end = _finite_number(raw_word.get("endTime"), f"wordTimings[{index}].endTime")
        confidence = _finite_number(
            raw_word.get("confidence"), f"wordTimings[{index}].confidence"
        )
        if start < previous_end - 1e-6 or end <= start or end > duration + 1e-6:
            raise ValueError("SpeakType word timings overlap or exceed the audio timeline")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("SpeakType word confidence is outside [0, 1]")
        words.append(
            AlignedWord(
                word=_gesturelsm_word_mark(mark),
                start_seconds=max(previous_end, start),
                end_seconds=min(duration, end),
                confidence=confidence,
            )
        )
        previous_end = end
    return SpeechAlignment(
        transcript=transcript,
        duration_seconds=duration,
        words=tuple(words),
        engine="speaktype_parakeet",
        model_revision=model,
    )


def render_word_textgrid(alignment: SpeechAlignment) -> str:
    """Render a contiguous word tier; gaps are explicit PAD/silence intervals."""

    intervals: list[tuple[float, float, str]] = []
    cursor = 0.0
    for word in alignment.words:
        if word.start_seconds > cursor + 1e-9:
            intervals.append((cursor, word.start_seconds, " "))
        intervals.append((word.start_seconds, word.end_seconds, word.word))
        cursor = word.end_seconds
    if cursor < alignment.duration_seconds - 1e-9:
        intervals.append((cursor, alignment.duration_seconds, " "))
    if not intervals:
        raise ValueError("Cannot render an empty word alignment")

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {alignment.duration_seconds:.9f}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "words"',
        "        xmin = 0",
        f"        xmax = {alignment.duration_seconds:.9f}",
        f"        intervals: size = {len(intervals)}",
    ]
    for index, (start, end, mark) in enumerate(intervals, start=1):
        lines.extend(
            (
                f"        intervals [{index}]:",
                f"            xmin = {start:.9f}",
                f"            xmax = {end:.9f}",
                f'            text = "{mark}"',
            )
        )
    return "\n".join(lines) + "\n"


def write_speech_alignment(
    output_directory: str | Path, alignment: SpeechAlignment
) -> tuple[Path, Path]:
    """Write the two supplied inputs consumed by the isolated GestureLSM worker."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    transcript_path = output / "transcript.txt"
    textgrid_path = output / "alignment.TextGrid"
    if transcript_path.is_symlink() or textgrid_path.is_symlink():
        raise ValueError("Speech alignment targets must not be symlinks")
    transcript_path.write_text(alignment.transcript + "\n", encoding="utf-8")
    textgrid_path.write_text(render_word_textgrid(alignment), encoding="utf-8")
    return transcript_path, textgrid_path


__all__ = [
    "AlignedWord",
    "SpeechAlignment",
    "load_speaktype_parakeet_alignment",
    "normalize_supplied_transcript",
    "render_word_textgrid",
    "write_mfa_lab",
    "write_speech_alignment",
]
