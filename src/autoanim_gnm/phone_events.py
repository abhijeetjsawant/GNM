"""Fail-closed phone-event evidence for production lipsync qualification.

The animation model never authors this evidence.  A reviewed Praat/MFA
TextGrid is imported on the same 48 kHz project clock used by acting and body
tracks.  Interval midpoints are retained as useful diagnostic apex estimates,
but are never mislabeled as independently reviewed apex annotations.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256 as sha256_digest
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from .errors import AutoAnimError
from .serialization import write_json


PHONE_EVENT_SCHEMA_VERSION = "autoanim.phone-events/1.0"
PHONE_TIMING_REPORT_SCHEMA_VERSION = "autoanim.phone-timing-report/1.0"
TICKS_PER_SECOND = 48_000
MAX_TEXTGRID_BYTES = 8 * 1024 * 1024
MAX_PHONE_EVENTS = 100_000
MAX_DURATION_SECONDS = 6 * 60 * 60
MAX_PHONE_LABEL_BYTES = 64
MAX_WORD_LABEL_BYTES = 256
MAX_COPIED_WORD_BYTES = 1 * 1024 * 1024
MAX_SERIALIZED_PHONE_EVENT_BYTES = 32 * 1024 * 1024

_ITEM_RE = re.compile(
    r"(?ms)^\s*item\s*\[\d+\]\s*:\s*(.*?)(?=^\s*item\s*\[\d+\]\s*:|\Z)"
)
_INTERVAL_RE = re.compile(
    r"(?ms)^\s*intervals\s*\[\d+\]\s*:\s*(.*?)(?=^\s*intervals\s*\[\d+\]\s*:|^\s*points\s*\[\d+\]\s*:|\Z)"
)
_POINT_RE = re.compile(
    r"(?ms)^\s*points\s*\[\d+\]\s*:\s*(.*?)(?=^\s*points\s*\[\d+\]\s*:|^\s*intervals\s*\[\d+\]\s*:|\Z)"
)

# MFA's ``spn`` is an unknown/spoken-noise phone, not silence. It can contain
# speech-like or visible nonverbal articulation and must remain reviewable.
_SILENCE = frozenset({"", "SIL", "SP", "PAU", "<SIL>", "<SP>"})
_PHONE_FEATURES: dict[str, tuple[str, str, bool, bool]] = {
    # phone: manner, place, voiced, rounded
    "P": ("stop", "bilabial", False, False),
    "B": ("stop", "bilabial", True, False),
    "M": ("nasal", "bilabial", True, False),
    "F": ("fricative", "labiodental", False, False),
    "V": ("fricative", "labiodental", True, False),
    "T": ("stop", "alveolar", False, False),
    "D": ("stop", "alveolar", True, False),
    "N": ("nasal", "alveolar", True, False),
    "L": ("approximant", "alveolar", True, False),
    "S": ("fricative", "alveolar", False, False),
    "Z": ("fricative", "alveolar", True, False),
    "TH": ("fricative", "dental", False, False),
    "DH": ("fricative", "dental", True, False),
    "UW": ("vowel", "vowel", True, True),
    "UH": ("vowel", "vowel", True, True),
    "OW": ("vowel", "vowel", True, True),
    "AO": ("vowel", "vowel", True, True),
    "OY": ("diphthong", "vowel", True, True),
    "W": ("approximant", "labiovelar", True, True),
}


def _invalid(message: str, **details: object) -> AutoAnimError:
    return AutoAnimError("PHONE_EVIDENCE_INVALID", message, details)


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) < 2 or text[0] != '"' or text[-1] != '"':
        raise _invalid("TextGrid string fields must be quoted")
    return text[1:-1].replace('""', '"')


def _field(block: str, name: str, *, quoted: bool = False) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(.+?)\s*$", block)
    if match is None:
        raise _invalid(f"TextGrid block is missing {name}")
    return _unquote(match.group(1)) if quoted else match.group(1).strip()


def _finite_time(block: str, name: str) -> float:
    try:
        value = float(_field(block, name))
    except ValueError as exc:
        raise _invalid(f"TextGrid {name} must be numeric") from exc
    if not math.isfinite(value):
        raise _invalid(f"TextGrid {name} must be finite")
    return value


def _normalize_phone(value: str) -> tuple[str, int | None]:
    label = " ".join(value.strip().split())
    if not label:
        return "SIL", None
    upper = label.upper()
    stress: int | None = None
    match = re.fullmatch(r"([A-Z]+)([0-2])", upper)
    if match is not None:
        upper = match.group(1)
        stress = int(match.group(2))
    aliases = {
        "[SIL]": "SIL",
        "[SP]": "SP",
        "[PAU]": "PAU",
        "[SPN]": "SPN",
        "P": "P",
        "B": "B",
        "M": "M",
        "F": "F",
        "V": "V",
        "T": "T",
        "D": "D",
        "N": "N",
        "L": "L",
        "S": "S",
        "Z": "Z",
        "Θ": "TH",
        "Ð": "DH",
        "U": "UW",
        "O": "OW",
    }
    return aliases.get(upper, upper), stress


def _ticks(seconds: float) -> int:
    return int(round(seconds * TICKS_PER_SECOND))


@dataclass(frozen=True, slots=True)
class PhoneEvent:
    event_id: str
    phone: str
    source_label: str
    word: str | None
    start_tick: int
    apex_tick: int
    end_tick: int
    apex_reviewed: bool
    stress: int | None
    manner: str
    place: str
    voiced: bool | None
    rounded: bool | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_label, str)
            or len(self.source_label.encode("utf-8")) > MAX_PHONE_LABEL_BYTES
            or not isinstance(self.phone, str)
            or len(self.phone.encode("utf-8")) > MAX_PHONE_LABEL_BYTES
        ):
            raise _invalid("Phone event label exceeds the bounded evidence schema")
        if self.word is not None and (
            not isinstance(self.word, str)
            or len(self.word.encode("utf-8")) > MAX_WORD_LABEL_BYTES
        ):
            raise _invalid("Phone event word exceeds the bounded evidence schema")

    @property
    def is_silence(self) -> bool:
        return self.phone in _SILENCE or self.manner == "silence"

    @property
    def is_bilabial(self) -> bool:
        return self.place == "bilabial"

    @property
    def is_labiodental(self) -> bool:
        return self.place == "labiodental"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.event_id,
            "phone": self.phone,
            "source_label": self.source_label,
            "word": self.word,
            "start_tick": self.start_tick,
            "apex_tick": self.apex_tick,
            "end_tick": self.end_tick,
            "apex_reviewed": self.apex_reviewed,
            "stress": self.stress,
            "manner": self.manner,
            "place": self.place,
            "voiced": self.voiced,
            "rounded": self.rounded,
        }


@dataclass(frozen=True, slots=True)
class PhoneAnnotationSet:
    events: tuple[PhoneEvent, ...]
    duration_ticks: int
    source_textgrid_sha256: str
    source_audio_sha256: str
    phone_tier: str
    word_tier: str | None
    apex_tier: str | None
    independently_reviewed: bool
    reviewer: str | None

    def __post_init__(self) -> None:
        if len(self.events) > MAX_PHONE_EVENTS:
            raise _invalid("Phone annotation set contains too many events")
        copied_word_bytes = sum(
            len(event.word.encode("utf-8"))
            for event in self.events
            if event.word is not None
        )
        if copied_word_bytes > MAX_COPIED_WORD_BYTES:
            raise _invalid(
                "Phone annotation words exceed the bounded copied-label budget",
                bytes=copied_word_bytes,
                maximum=MAX_COPIED_WORD_BYTES,
            )

    @property
    def production_review_complete(self) -> bool:
        articulatory = tuple(event for event in self.events if not event.is_silence)
        return bool(
            self.independently_reviewed
            and self.reviewer
            and articulatory
            and all(event.apex_reviewed for event in articulatory)
        )

    def as_dict(self) -> dict[str, Any]:
        articulatory = tuple(event for event in self.events if not event.is_silence)
        return {
            "schema_version": PHONE_EVENT_SCHEMA_VERSION,
            "timebase": {"ticks_per_second": TICKS_PER_SECOND},
            "bindings": {
                "textgrid_sha256": self.source_textgrid_sha256,
                "audio_sha256": self.source_audio_sha256,
            },
            "tiers": {
                "phones": self.phone_tier,
                "words": self.word_tier,
                "apexes": self.apex_tier,
            },
            "review": {
                "independently_reviewed": self.independently_reviewed,
                "reviewer": self.reviewer,
                "all_articulatory_apexes_reviewed": bool(
                    articulatory and all(event.apex_reviewed for event in articulatory)
                ),
                "production_review_complete": self.production_review_complete,
            },
            "duration_ticks": self.duration_ticks,
            "event_count": len(self.events),
            "articulatory_event_count": len(articulatory),
            "bilabial_event_count": sum(event.is_bilabial for event in self.events),
            "labiodental_event_count": sum(
                event.is_labiodental for event in self.events
            ),
            "events": [event.as_dict() for event in self.events],
            "claims": {
                "animation_authored_by_annotations": False,
                "apex_midpoint_is_independent_evidence": False,
                "production_validated": False,
            },
        }


@dataclass(frozen=True, slots=True)
class _Tier:
    name: str
    kind: str
    intervals: tuple[tuple[float, float, str], ...]
    points: tuple[tuple[float, str], ...]


def _parse_tiers(text: str) -> tuple[_Tier, ...]:
    tiers: list[_Tier] = []
    for item in _ITEM_RE.findall(text):
        kind = _field(item, "class", quoted=True)
        name = _field(item, "name", quoted=True)
        intervals: list[tuple[float, float, str]] = []
        points: list[tuple[float, str]] = []
        if kind == "IntervalTier":
            for block in _INTERVAL_RE.findall(item):
                intervals.append(
                    (
                        _finite_time(block, "xmin"),
                        _finite_time(block, "xmax"),
                        _field(block, "text", quoted=True),
                    )
                )
        elif kind == "TextTier":
            for block in _POINT_RE.findall(item):
                points.append(
                    (
                        _finite_time(block, "number"),
                        _field(block, "mark", quoted=True),
                    )
                )
        else:
            continue
        tiers.append(_Tier(name, kind, tuple(intervals), tuple(points)))
    if not tiers:
        raise _invalid("TextGrid has no supported long-format tiers")
    return tuple(tiers)


def _select_tier(
    tiers: Iterable[_Tier], candidates: tuple[str, ...], kind: str
) -> _Tier | None:
    normalized = {name.casefold() for name in candidates}
    matches = [
        tier
        for tier in tiers
        if tier.kind == kind and tier.name.strip().casefold() in normalized
    ]
    if len(matches) > 1:
        raise _invalid("TextGrid contains multiple matching tiers", tiers=[t.name for t in matches])
    return matches[0] if matches else None


def _validate_intervals(
    intervals: tuple[tuple[float, float, str], ...], duration: float, name: str
) -> None:
    previous_end = 0.0
    for index, (start, end, _) in enumerate(intervals):
        if start < -1e-7 or end <= start or end > duration + 1e-6:
            raise _invalid(
                f"TextGrid {name} interval is outside the audio duration",
                index=index,
                start=start,
                end=end,
                duration=duration,
            )
        if start < previous_end - 1e-7:
            raise _invalid(f"TextGrid {name} intervals overlap", index=index)
        previous_end = end


def load_textgrid_phone_events(
    textgrid_path: str | Path,
    *,
    audio_path: str | Path,
    duration_seconds: float,
    independently_reviewed: bool = False,
    reviewer: str | None = None,
) -> PhoneAnnotationSet:
    """Load a bounded Praat/MFA long TextGrid and bind it to exact audio bytes."""

    annotation_path = Path(textgrid_path)
    media_path = Path(audio_path)
    if not annotation_path.is_file() or not media_path.is_file():
        raise _invalid("Phone annotation and source audio must both be files")
    size = annotation_path.stat().st_size
    if size <= 0 or size > MAX_TEXTGRID_BYTES:
        raise _invalid("TextGrid size is outside the accepted bounds", bytes=size)
    if (
        not math.isfinite(duration_seconds)
        or duration_seconds <= 0
        or duration_seconds > MAX_DURATION_SECONDS
    ):
        raise _invalid("Audio duration is outside the phone-evidence bounds")
    if independently_reviewed and (reviewer is None or not reviewer.strip()):
        raise _invalid("Reviewed phone annotations require a reviewer")
    if reviewer is not None and len(reviewer.strip()) > 160:
        raise _invalid("Phone annotation reviewer exceeds 160 characters")
    try:
        raw = annotation_path.read_bytes()
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _invalid("TextGrid must be UTF-8") from exc
    tiers = _parse_tiers(text)
    phone_tier = _select_tier(tiers, ("phones", "phone", "phonemes"), "IntervalTier")
    if phone_tier is None:
        raise _invalid("TextGrid has no phones interval tier")
    word_tier = _select_tier(tiers, ("words", "word"), "IntervalTier")
    apex_tier = _select_tier(
        tiers, ("phone_apex", "phone_apexes", "apexes"), "TextTier"
    )
    _validate_intervals(phone_tier.intervals, duration_seconds, phone_tier.name)
    if word_tier is not None:
        _validate_intervals(word_tier.intervals, duration_seconds, word_tier.name)
    if len(phone_tier.intervals) > MAX_PHONE_EVENTS:
        raise _invalid("TextGrid contains too many phone events")
    if word_tier is not None and len(word_tier.intervals) > MAX_PHONE_EVENTS:
        raise _invalid("TextGrid contains too many word events")

    apex_points = list(apex_tier.points if apex_tier is not None else ())
    if len(apex_points) > MAX_PHONE_EVENTS:
        raise _invalid("TextGrid contains too many phone apex events")
    phone_starts = [interval[0] for interval in phone_tier.intervals]
    apex_by_phone: dict[int, float] = {}
    previous_apex = -math.inf
    for point_index, (point_time, point_label) in enumerate(apex_points):
        if (
            not math.isfinite(point_time)
            or point_time < 0
            or point_time > duration_seconds + 1e-6
            or point_time <= previous_apex
        ):
            raise _invalid(
                "Phone apex points must be finite, in-range, and strictly ordered",
                index=point_index,
            )
        previous_apex = point_time
        insertion = bisect_right(phone_starts, point_time + 1e-7) - 1
        candidate_indices = {
            index
            for index in (insertion - 1, insertion, insertion + 1)
            if 0 <= index < len(phone_tier.intervals)
            and phone_tier.intervals[index][0] - 1e-7
            <= point_time
            <= phone_tier.intervals[index][1] + 1e-7
        }
        if point_label.strip():
            point_phone, _ = _normalize_phone(point_label)
            candidate_indices = {
                index
                for index in candidate_indices
                if _normalize_phone(phone_tier.intervals[index][2])[0]
                == point_phone
            }
        if len(candidate_indices) != 1:
            raise _invalid(
                "A phone apex point does not bind unambiguously to one matching interval",
                index=point_index,
                candidate_count=len(candidate_indices),
            )
        event_index = candidate_indices.pop()
        if event_index in apex_by_phone:
            raise _invalid(
                "A phone interval contains multiple apex points",
                index=point_index,
            )
        apex_by_phone[event_index] = point_time

    word_starts = (
        [interval[0] for interval in word_tier.intervals]
        if word_tier is not None
        else []
    )
    events: list[PhoneEvent] = []
    previous_end_tick = 0
    for index, (start, end, source_label) in enumerate(phone_tier.intervals):
        start_tick = _ticks(start)
        end_tick = _ticks(end)
        if end_tick <= start_tick or start_tick < previous_end_tick:
            raise _invalid("Phone intervals collapse or overlap on the 48 kHz clock", index=index)
        previous_end_tick = end_tick
        phone, stress = _normalize_phone(source_label)
        apex_tick = (start_tick + end_tick) // 2
        apex_reviewed = False
        point_time = apex_by_phone.get(index)
        if point_time is not None:
            apex_tick = _ticks(point_time)
            # A point-tier mark is only independent review evidence when the
            # operator explicitly attested the complete annotation set.
            apex_reviewed = independently_reviewed
        midpoint = 0.5 * (start + end)
        word: str | None = None
        if word_tier is not None:
            word_index = bisect_right(word_starts, midpoint + 1e-7) - 1
            if 0 <= word_index < len(word_tier.intervals):
                word_start, word_end, label = word_tier.intervals[word_index]
                if (
                    word_start - 1e-7 <= midpoint <= word_end + 1e-7
                    and label.strip()
                ):
                    word = label.strip()
        if phone in _SILENCE:
            manner, place, voiced, rounded = "silence", "none", None, None
        else:
            manner, place, voiced, rounded = _PHONE_FEATURES.get(
                phone, ("unknown", "unknown", None, None)
            )
        events.append(
            PhoneEvent(
                event_id=f"phone_{index + 1:06d}",
                phone=phone,
                source_label=source_label,
                word=word,
                start_tick=start_tick,
                apex_tick=apex_tick,
                end_tick=end_tick,
                apex_reviewed=apex_reviewed,
                stress=stress,
                manner=manner,
                place=place,
                voiced=voiced,
                rounded=rounded,
            )
        )
    if not events:
        raise _invalid("TextGrid phones tier is empty")
    duration_ticks = _ticks(duration_seconds)
    if events[-1].end_tick > duration_ticks + 1:
        raise _invalid("Phone events extend beyond the normalized audio")
    return PhoneAnnotationSet(
        events=tuple(events),
        duration_ticks=duration_ticks,
        source_textgrid_sha256=sha256_digest(raw).hexdigest(),
        source_audio_sha256=_sha256_file(media_path),
        phone_tier=phone_tier.name,
        word_tier=word_tier.name if word_tier is not None else None,
        apex_tier=apex_tier.name if apex_tier is not None else None,
        independently_reviewed=independently_reviewed,
        reviewer=reviewer.strip() if reviewer is not None else None,
    )


def _sha256_file(path: Path) -> str:
    digest = sha256_digest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_phone_events(path: str | Path, annotations: PhoneAnnotationSet) -> Path:
    document = annotations.as_dict()
    payload_bytes = len(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ) + 1
    if payload_bytes > MAX_SERIALIZED_PHONE_EVENT_BYTES:
        raise _invalid(
            "Serialized phone-event evidence exceeds the bounded artifact size",
            bytes=payload_bytes,
            maximum=MAX_SERIALIZED_PHONE_EVENT_BYTES,
        )
    return write_json(path, document)


def evaluate_bilabial_timing(
    annotations: PhoneAnnotationSet,
    *,
    timestamps_seconds: np.ndarray,
    lip_gap_interocular: np.ndarray,
    contact_threshold_interocular: float,
) -> dict[str, Any]:
    """Evaluate bilabial timing without pretending interval midpoints are labels."""

    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    gaps = np.asarray(lip_gap_interocular, dtype=np.float64)
    if (
        timestamps.ndim != 1
        or gaps.shape != timestamps.shape
        or len(timestamps) < 2
        or not np.isfinite(timestamps).all()
        or not np.isfinite(gaps).all()
        or timestamps[0] < 0
        or np.any(np.diff(timestamps) <= 0)
        or not math.isfinite(contact_threshold_interocular)
        or contact_threshold_interocular <= 0
    ):
        raise _invalid("Bilabial evaluation inputs are invalid")
    ticks = np.rint(timestamps * TICKS_PER_SECOND).astype(np.int64)
    predicted_contact = gaps <= contact_threshold_interocular
    event_reports: list[dict[str, Any]] = []
    reviewed_errors: list[float] = []
    reviewed_onset_errors: list[float] = []
    reviewed_release_errors: list[float] = []
    found = 0
    for event in (item for item in annotations.events if item.is_bilabial):
        inside = (ticks >= event.start_tick) & (ticks <= event.end_tick)
        indices = np.flatnonzero(inside)
        if not len(indices):
            event_reports.append(
                {
                    "id": event.event_id,
                    "phone": event.phone,
                    "scored": False,
                    "failure": "no_animation_sample_inside_event",
                }
            )
            continue
        best_index = int(indices[np.argmin(gaps[indices])])
        contact_indices = indices[predicted_contact[indices]]
        contact_found = bool(len(contact_indices))
        found += int(contact_found)
        predicted_apex_tick = int(ticks[best_index])
        apex_error_ms = abs(predicted_apex_tick - event.apex_tick) / 48.0
        onset_tick = int(ticks[contact_indices[0]]) if contact_found else None
        release_tick = int(ticks[contact_indices[-1]]) if contact_found else None
        onset_error_ms = (
            abs(onset_tick - event.start_tick) / 48.0 if onset_tick is not None else None
        )
        release_error_ms = (
            abs(release_tick - event.end_tick) / 48.0 if release_tick is not None else None
        )
        if event.apex_reviewed:
            reviewed_errors.append(apex_error_ms)
        if annotations.independently_reviewed and contact_found:
            reviewed_onset_errors.append(float(onset_error_ms))
            reviewed_release_errors.append(float(release_error_ms))
        event_reports.append(
            {
                "id": event.event_id,
                "phone": event.phone,
                "word": event.word,
                "scored": True,
                "apex_reviewed": event.apex_reviewed,
                "contact_found": contact_found,
                "predicted_apex_tick": predicted_apex_tick,
                "apex_error_ms": apex_error_ms,
                "contact_onset_tick": onset_tick,
                "contact_release_tick": release_tick,
                "onset_error_ms": onset_error_ms,
                "release_error_ms": release_error_ms,
                "minimum_gap_interocular": float(gaps[best_index]),
            }
        )

    def stats(values: list[float]) -> dict[str, float | None]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "median_ms": float(np.median(array)) if len(array) else None,
            "p95_ms": float(np.percentile(array, 95)) if len(array) else None,
        }

    bilabial_count = sum(event.is_bilabial for event in annotations.events)
    apex_stats = stats(reviewed_errors)
    onset_stats = stats(reviewed_onset_errors)
    release_stats = stats(reviewed_release_errors)
    failures: list[str] = []
    if not annotations.production_review_complete:
        failures.append("independent_reviewed_phone_apexes")
    if bilabial_count < 100:
        failures.append("minimum_bilabial_event_count")
    if apex_stats["median_ms"] is None or apex_stats["median_ms"] > 1000 / 30:
        failures.append("bilabial_apex_median")
    if apex_stats["p95_ms"] is None or apex_stats["p95_ms"] > 2000 / 30:
        failures.append("bilabial_apex_p95")
    if onset_stats["median_ms"] is None or onset_stats["median_ms"] > 40:
        failures.append("bilabial_onset_median")
    if onset_stats["p95_ms"] is None or onset_stats["p95_ms"] > 80:
        failures.append("bilabial_onset_p95")
    if release_stats["median_ms"] is None or release_stats["median_ms"] > 40:
        failures.append("bilabial_release_median")
    if release_stats["p95_ms"] is None or release_stats["p95_ms"] > 80:
        failures.append("bilabial_release_p95")
    # This first evidence phase intentionally scores only bilabials. A timing
    # report must not become a production approval until the other required
    # articulators and false-contact negatives are independently evaluated.
    failures.extend(
        (
            "labiodental_geometry_not_evaluated",
            "tongue_contact_not_evaluated",
            "false_contact_not_evaluated",
        )
    )
    return {
        "schema_version": PHONE_TIMING_REPORT_SCHEMA_VERSION,
        "annotation_bindings": {
            "textgrid_sha256": annotations.source_textgrid_sha256,
            "audio_sha256": annotations.source_audio_sha256,
        },
        "contact_threshold_interocular": contact_threshold_interocular,
        "bilabial_event_count": bilabial_count,
        "bilabial_contact_found_count": found,
        "reviewed_apex_event_count": len(reviewed_errors),
        "apex_timing": apex_stats,
        "onset_timing": onset_stats,
        "release_timing": release_stats,
        "events": event_reports,
        "production_gate": {"passed": not failures, "failures": failures},
        "claims": {
            "interval_midpoint_scored_as_reviewed_apex": False,
            "labiodental_geometry_evaluated": False,
            "tongue_contact_evaluated": False,
            "production_validated": not failures,
        },
    }


__all__ = [
    "MAX_COPIED_WORD_BYTES",
    "MAX_PHONE_LABEL_BYTES",
    "MAX_SERIALIZED_PHONE_EVENT_BYTES",
    "MAX_WORD_LABEL_BYTES",
    "PHONE_EVENT_SCHEMA_VERSION",
    "PHONE_TIMING_REPORT_SCHEMA_VERSION",
    "PhoneAnnotationSet",
    "PhoneEvent",
    "TICKS_PER_SECOND",
    "evaluate_bilabial_timing",
    "load_textgrid_phone_events",
    "write_phone_events",
]
