from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.phone_events import (
    TICKS_PER_SECOND,
    evaluate_bilabial_timing,
    load_textgrid_phone_events,
    write_phone_events,
)


def _textgrid(*, apex: bool = True, overlap: bool = False) -> str:
    second_start = 0.05 if overlap else 0.10
    apex_tier = (
        '''
    item [3]:
        class = "TextTier"
        name = "phone_apex"
        xmin = 0
        xmax = 0.5
        points: size = 1
        points [1]:
            number = 0.18
            mark = "P"
'''
        if apex
        else ""
    )
    return f'''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.5
tiers? <exists>
size = {3 if apex else 2}
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 0.5
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = "sil"
        intervals [2]:
            xmin = {second_start}
            xmax = 0.3
            text = "P"
        intervals [3]:
            xmin = 0.3
            xmax = 0.5
            text = "IY1"
    item [2]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 0.5
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "pea"
{apex_tier}'''


def _files(tmp_path: Path, text: str) -> tuple[Path, Path]:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"immutable normalized audio")
    grid = tmp_path / "phones.TextGrid"
    grid.write_text(text, encoding="utf-8")
    return audio, grid


def test_reviewed_textgrid_retains_phone_features_words_and_apex(tmp_path: Path) -> None:
    audio, grid = _files(tmp_path, _textgrid())
    annotations = load_textgrid_phone_events(
        grid,
        audio_path=audio,
        duration_seconds=0.5,
        independently_reviewed=True,
        reviewer="Test phonetic annotator",
    )

    assert len(annotations.events) == 3
    assert annotations.events[0].is_silence
    bilabial = annotations.events[1]
    assert bilabial.phone == "P"
    assert bilabial.place == "bilabial"
    assert bilabial.word == "pea"
    assert bilabial.apex_tick == round(0.18 * TICKS_PER_SECOND)
    assert bilabial.apex_reviewed is True
    assert annotations.events[2].phone == "IY"
    assert annotations.events[2].stress == 1
    assert annotations.production_review_complete is False
    # The vowel has no reviewed apex. A point tier must cover every
    # articulatory event before the whole annotation set is production-complete.

    written = write_phone_events(tmp_path / "events.json", annotations)
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["bindings"]["audio_sha256"] == annotations.source_audio_sha256
    assert document["claims"]["apex_midpoint_is_independent_evidence"] is False
    assert document["review"]["production_review_complete"] is False


def test_interval_midpoint_is_diagnostic_and_never_reviewed(tmp_path: Path) -> None:
    audio, grid = _files(tmp_path, _textgrid(apex=False))
    annotations = load_textgrid_phone_events(
        grid,
        audio_path=audio,
        duration_seconds=0.5,
        independently_reviewed=True,
        reviewer="Reviewer",
    )

    bilabial = annotations.events[1]
    assert bilabial.apex_tick == round(0.20 * TICKS_PER_SECOND)
    assert bilabial.apex_reviewed is False
    assert annotations.production_review_complete is False


def test_explicit_apex_without_review_attestation_is_not_reviewed(
    tmp_path: Path,
) -> None:
    audio, grid = _files(tmp_path, _textgrid(apex=True))
    annotations = load_textgrid_phone_events(
        grid,
        audio_path=audio,
        duration_seconds=0.5,
    )

    bilabial = annotations.events[1]
    assert bilabial.apex_tick == round(0.18 * TICKS_PER_SECOND)
    assert bilabial.apex_reviewed is False
    assert annotations.production_review_complete is False


def test_textgrid_overlap_and_unreviewed_reviewer_claim_fail_closed(tmp_path: Path) -> None:
    audio, grid = _files(tmp_path, _textgrid(overlap=True))
    with pytest.raises(AutoAnimError, match="overlap") as overlap:
        load_textgrid_phone_events(grid, audio_path=audio, duration_seconds=0.5)
    assert overlap.value.code == "PHONE_EVIDENCE_INVALID"

    grid.write_text(_textgrid(), encoding="utf-8")
    with pytest.raises(AutoAnimError, match="reviewer"):
        load_textgrid_phone_events(
            grid,
            audio_path=audio,
            duration_seconds=0.5,
            independently_reviewed=True,
        )


def test_bilabial_evaluator_scores_geometry_but_withholds_production(tmp_path: Path) -> None:
    audio, grid = _files(tmp_path, _textgrid())
    annotations = load_textgrid_phone_events(
        grid,
        audio_path=audio,
        duration_seconds=0.5,
        independently_reviewed=True,
        reviewer="Reviewer",
    )
    timestamps = np.arange(51, dtype=np.float64) / 100.0
    gap = np.full(51, 0.08, dtype=np.float64)
    gap[10:31] = 0.004
    gap[18] = 0.001

    report = evaluate_bilabial_timing(
        annotations,
        timestamps_seconds=timestamps,
        lip_gap_interocular=gap,
        contact_threshold_interocular=0.006,
    )

    event = next(item for item in report["events"] if item.get("phone") == "P")
    assert event["contact_found"] is True
    assert event["apex_error_ms"] == pytest.approx(0.0)
    assert report["reviewed_apex_event_count"] == 1
    assert report["production_gate"]["passed"] is False
    assert "minimum_bilabial_event_count" in report["production_gate"]["failures"]
    assert "labiodental_geometry_not_evaluated" in report["production_gate"][
        "failures"
    ]
    assert report["claims"]["labiodental_geometry_evaluated"] is False


def test_bilabial_apex_error_uses_exact_30_fps_sample_ticks(tmp_path: Path) -> None:
    audio, grid = _files(tmp_path, _textgrid())
    annotations = load_textgrid_phone_events(
        grid,
        audio_path=audio,
        duration_seconds=0.5,
        independently_reviewed=True,
        reviewer="Reviewer",
    )
    timestamps = np.arange(16, dtype=np.float64) / 30.0
    gap = np.full(16, 0.08, dtype=np.float64)
    gap[5] = 0.001  # 166.667 ms versus the reviewed 180 ms apex.

    report = evaluate_bilabial_timing(
        annotations,
        timestamps_seconds=timestamps,
        lip_gap_interocular=gap,
        contact_threshold_interocular=0.006,
    )

    event = next(item for item in report["events"] if item.get("phone") == "P")
    assert event["predicted_apex_tick"] == 8_000
    assert event["apex_error_ms"] == pytest.approx(13.3333333333)


def test_unbound_apex_point_is_rejected(tmp_path: Path) -> None:
    audio, grid = _files(
        tmp_path,
        _textgrid().replace('number = 0.18', 'number = 0.45').replace(
            'mark = "P"', 'mark = "P"'
        ),
    )
    with pytest.raises(AutoAnimError, match="apex"):
        load_textgrid_phone_events(
            grid,
            audio_path=audio,
            duration_seconds=0.5,
        )


def test_unlabeled_apex_on_phone_boundary_is_rejected_as_ambiguous(
    tmp_path: Path,
) -> None:
    audio, grid = _files(
        tmp_path,
        _textgrid().replace('number = 0.18', 'number = 0.3').replace(
            'mark = "P"', 'mark = ""'
        ),
    )
    with pytest.raises(AutoAnimError, match="unambiguously"):
        load_textgrid_phone_events(
            grid,
            audio_path=audio,
            duration_seconds=0.5,
        )


def test_mfa_spn_is_unknown_spoken_noise_not_silence(tmp_path: Path) -> None:
    audio, grid = _files(
        tmp_path,
        _textgrid(apex=False).replace('text = "IY1"', 'text = "spn"'),
    )
    annotations = load_textgrid_phone_events(
        grid,
        audio_path=audio,
        duration_seconds=0.5,
        independently_reviewed=True,
        reviewer="Reviewer",
    )

    spoken_noise = annotations.events[2]
    assert spoken_noise.phone == "SPN"
    assert spoken_noise.is_silence is False
    assert spoken_noise.manner == "unknown"
    assert annotations.production_review_complete is False


def test_bracketed_alignment_silence_aliases_are_silence(tmp_path: Path) -> None:
    audio, grid = _files(
        tmp_path,
        _textgrid(apex=False)
        .replace('text = "SIL"', 'text = "[SIL]"')
        .replace('text = "P"', 'text = "[SP]"')
        .replace('text = "IY1"', 'text = "[PAU]"'),
    )
    annotations = load_textgrid_phone_events(
        grid,
        audio_path=audio,
        duration_seconds=0.5,
    )

    assert [event.phone for event in annotations.events] == ["SIL", "SP", "PAU"]
    assert all(event.is_silence for event in annotations.events)
