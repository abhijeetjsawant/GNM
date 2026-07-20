from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import autoanim_gnm.phone_articulation as phone_articulation_module
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.phone_articulation import (
    PHONE_ARTICULATION_REPORT_SCHEMA_VERSION,
    ArticulationCalibration,
    ArticulationGateThresholds,
    ArticulationGeometry,
    MAX_ARTICULATION_EVENTS,
    evaluate_phone_articulation,
    measure_articulation_geometry,
    measure_articulation_geometry_from_controls,
    summarize_phone_articulation,
)
from autoanim_gnm.phone_events import PhoneAnnotationSet, PhoneEvent, TICKS_PER_SECOND
from autoanim_gnm.oral_validation import validate_oral_frames
from autoanim_gnm.rig import ControlRig
from autoanim_gnm.semantic_decoder import ExpressionDecoder


def _event(
    index: int,
    phone: str,
    start: float,
    end: float,
    *,
    manner: str,
    place: str,
    rounded: bool,
) -> PhoneEvent:
    return PhoneEvent(
        event_id=f"phone_{index:06d}",
        phone=phone,
        source_label=phone,
        word=None,
        start_tick=round(start * TICKS_PER_SECOND),
        apex_tick=round(0.5 * (start + end) * TICKS_PER_SECOND),
        end_tick=round(end * TICKS_PER_SECOND),
        apex_reviewed=True,
        stress=None,
        manner=manner,
        place=place,
        voiced=False,
        rounded=rounded,
    )


def _annotations() -> PhoneAnnotationSet:
    events = (
        _event(1, "P", 0.50, 1.00, manner="stop", place="bilabial", rounded=False),
        _event(
            2,
            "F",
            1.25,
            1.75,
            manner="fricative",
            place="labiodental",
            rounded=False,
        ),
        _event(3, "T", 2.00, 2.50, manner="stop", place="alveolar", rounded=False),
        _event(4, "UW", 2.75, 3.25, manner="vowel", place="vowel", rounded=True),
    )
    return PhoneAnnotationSet(
        events=events,
        duration_ticks=round(4.0 * TICKS_PER_SECOND),
        source_textgrid_sha256="1" * 64,
        source_audio_sha256="2" * 64,
        phone_tier="phones",
        word_tier="words",
        apex_tier="phone_apex",
        independently_reviewed=True,
        reviewer="Independent phonetics reviewer",
    )


def _diagnostic_calibration() -> ArticulationCalibration:
    return ArticulationCalibration(
        bilabial_gap_interocular=0.01,
        labiodental_gap_interocular=0.01,
        tongue_upper_teeth_gap_interocular=0.01,
        rounded_mouth_width_interocular=0.50,
    )


def _perfect_series() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    timestamps = np.arange(121, dtype=np.float64) / 30.0
    series = {
        "lip": np.full(len(timestamps), 0.08, dtype=np.float64),
        "labiodental": np.full(len(timestamps), 0.08, dtype=np.float64),
        "tongue": np.full(len(timestamps), 0.08, dtype=np.float64),
        "width": np.full(len(timestamps), 0.62, dtype=np.float64),
    }
    for event, key in zip(
        _annotations().events,
        ("lip", "labiodental", "tongue", "width"),
        strict=True,
    ):
        start = event.start_tick / TICKS_PER_SECOND
        end = event.end_tick / TICKS_PER_SECOND
        active = (timestamps >= start - 1.0e-9) & (timestamps <= end + 1.0e-9)
        series[key][active] = 0.005 if key != "width" else 0.45
        apex = int(np.argmin(np.abs(timestamps - event.apex_tick / TICKS_PER_SECOND)))
        series[key][apex] = 0.001 if key != "width" else 0.40
    return timestamps, series


def test_synthetic_phone_spans_can_pass_only_the_diagnostic_proxy_gate() -> None:
    timestamps, series = _perfect_series()
    report = evaluate_phone_articulation(
        _annotations(),
        timestamps_seconds=timestamps,
        lip_gap_interocular=series["lip"],
        labiodental_gap_interocular=series["labiodental"],
        tongue_upper_teeth_gap_interocular=series["tongue"],
        mouth_width_interocular=series["width"],
        calibration=_diagnostic_calibration(),
        gates=ArticulationGateThresholds(minimum_events_per_family=1),
    )

    assert report["schema_version"] == PHONE_ARTICULATION_REPORT_SCHEMA_VERSION
    assert report["phone_span_proxy_gate"] == {"passed": True, "failures": []}
    assert report["production_gate"]["passed"] is False
    assert "independent_articulation_state_annotations_not_implemented" in report[
        "production_gate"
    ]["failures"]
    assert report["claims"]["phone_span_proxy_gate_passed"] is True
    assert report["claims"]["phone_articulation_gate_passed"] is False
    assert report["claims"]["production_validated"] is False
    for family in ("bilabial", "labiodental", "tongue_upper_teeth", "rounded"):
        evidence = report["families"][family]
        assert evidence["event_count"] == 1
        assert evidence["proxy_active_event_count"] == 1
        assert evidence["proxy_active_event_recall"] == 1.0
        assert evidence["classification"]["f1"] >= 0.90
        assert evidence["classification"]["false_positive_fraction"] == 0.0
        assert evidence[
            "minimum_proxy_signal_vs_reviewed_phone_apex_error_frames"
        ]["median"] <= 0.5 + 1.0e-9
        assert evidence["phone_span_duration_error_frames"]["median"] <= 1.0 + 1.0e-9


def test_diagnostic_calibration_and_small_set_fail_closed() -> None:
    timestamps, series = _perfect_series()
    report = evaluate_phone_articulation(
        _annotations(),
        timestamps_seconds=timestamps,
        lip_gap_interocular=series["lip"],
        labiodental_gap_interocular=series["labiodental"],
        tongue_upper_teeth_gap_interocular=series["tongue"],
        mouth_width_interocular=series["width"],
        calibration=ArticulationCalibration(
            bilabial_gap_interocular=0.01,
            labiodental_gap_interocular=0.01,
            tongue_upper_teeth_gap_interocular=0.01,
            rounded_mouth_width_interocular=0.50,
        ),
    )

    failures = report["phone_span_proxy_gate"]["failures"]
    assert "bilabial_minimum_event_count" in failures
    assert "labiodental_minimum_event_count" in failures
    assert report["calibration"]["production_eligible"] is False
    assert report["production_gate"]["passed"] is False
    assert report["claims"]["production_validated"] is False


def test_false_contact_outside_the_phone_family_fails_classification() -> None:
    timestamps, series = _perfect_series()
    series["lip"][0:5] = 0.001
    report = evaluate_phone_articulation(
        _annotations(),
        timestamps_seconds=timestamps,
        lip_gap_interocular=series["lip"],
        labiodental_gap_interocular=series["labiodental"],
        tongue_upper_teeth_gap_interocular=series["tongue"],
        mouth_width_interocular=series["width"],
        calibration=_diagnostic_calibration(),
        gates=ArticulationGateThresholds(minimum_events_per_family=1),
    )

    bilabial = report["families"]["bilabial"]["classification"]
    assert bilabial["false_positive_frames"] == 5
    assert "bilabial_f1" in report["phone_span_proxy_gate"]["failures"]
    assert "bilabial_false_positive_fraction" in report["phone_span_proxy_gate"][
        "failures"
    ]


def test_articulation_arrays_and_calibration_claims_are_fail_closed() -> None:
    timestamps, series = _perfect_series()
    series["tongue"][4] = np.nan
    with pytest.raises(AutoAnimError, match="frame-aligned") as error:
        evaluate_phone_articulation(
            _annotations(),
            timestamps_seconds=timestamps,
            lip_gap_interocular=series["lip"],
            labiodental_gap_interocular=series["labiodental"],
            tongue_upper_teeth_gap_interocular=series["tongue"],
            mouth_width_interocular=series["width"],
            calibration=_diagnostic_calibration(),
        )
    assert error.value.code == "PHONE_ARTICULATION_INVALID"

    with pytest.raises(AutoAnimError, match="unsupported"):
        ArticulationCalibration(
            bilabial_gap_interocular=0.01,
            labiodental_gap_interocular=0.01,
            tongue_upper_teeth_gap_interocular=0.01,
            rounded_mouth_width_interocular=0.5,
            source="artist_approved_character_profile",
        )
    with pytest.raises(AutoAnimError, match="false-positive"):
        ArticulationGateThresholds(false_positive_fraction=0.0)
    with pytest.raises(AutoAnimError, match="duration median"):
        ArticulationGateThresholds(
            duration_median_frames=3.0,
            duration_p95_frames=2.0,
        )


def test_proxy_run_boundaries_are_not_clipped_to_the_phone_guard() -> None:
    timestamps, series = _perfect_series()
    # The P proxy begins two frames before the phone. One frame is within the
    # contextual guard and one is outside, so a clipped implementation could
    # incorrectly report an acceptable one-frame onset.
    series["lip"][13:15] = 0.001
    report = evaluate_phone_articulation(
        _annotations(),
        timestamps_seconds=timestamps,
        lip_gap_interocular=series["lip"],
        labiodental_gap_interocular=series["labiodental"],
        tongue_upper_teeth_gap_interocular=series["tongue"],
        mouth_width_interocular=series["width"],
        calibration=_diagnostic_calibration(),
        gates=ArticulationGateThresholds(minimum_events_per_family=1),
    )

    event = report["families"]["bilabial"]["events"][0]
    assert event["proxy_run_vs_phone_start_bias_ms"] == pytest.approx(-200.0 / 3.0)
    assert event["proxy_run_vs_phone_start_absolute_error_ms"] == pytest.approx(
        200.0 / 3.0
    )
    assert "bilabial_onset_median" in report["phone_span_proxy_gate"]["failures"]


def test_track_edge_proxy_runs_are_censored_not_scored_as_boundaries() -> None:
    timestamps, series = _perfect_series()
    series["lip"][:] = 0.001
    report = evaluate_phone_articulation(
        _annotations(),
        timestamps_seconds=timestamps,
        lip_gap_interocular=series["lip"],
        labiodental_gap_interocular=series["labiodental"],
        tongue_upper_teeth_gap_interocular=series["tongue"],
        mouth_width_interocular=series["width"],
        calibration=_diagnostic_calibration(),
        gates=ArticulationGateThresholds(minimum_events_per_family=1),
    )

    event = report["families"]["bilabial"]["events"][0]
    assert event["proxy_run_left_censored"] is True
    assert event["proxy_run_right_censored"] is True
    assert event["proxy_run_vs_phone_start_absolute_error_ms"] is None
    assert event["proxy_run_vs_phone_end_absolute_error_ms"] is None
    assert report["families"]["bilabial"]["phone_span_duration_error_ms"] == {
        "median": None,
        "p95": None,
    }


@pytest.mark.parametrize(
    ("timestamps_mutator", "series_mutator", "message"),
    (
        (lambda values: values.__setitem__(4, values[3] + 0.02), None, "exact 30 or 60"),
        (lambda values: values.__setitem__(-1, 1.0e300), None, "tick range"),
        (None, lambda values: values["labiodental"].__setitem__(4, -0.1), "frame-aligned"),
    ),
)
def test_irregular_overflow_and_negative_metric_inputs_fail_closed(
    timestamps_mutator, series_mutator, message: str
) -> None:
    timestamps, series = _perfect_series()
    if timestamps_mutator is not None:
        timestamps_mutator(timestamps)
    if series_mutator is not None:
        series_mutator(series)
    with pytest.raises(AutoAnimError, match=message):
        evaluate_phone_articulation(
            _annotations(),
            timestamps_seconds=timestamps,
            lip_gap_interocular=series["lip"],
            labiodental_gap_interocular=series["labiodental"],
            tongue_upper_teeth_gap_interocular=series["tongue"],
            mouth_width_interocular=series["width"],
            calibration=_diagnostic_calibration(),
        )


def test_real_gnm_geometry_measurement_uses_complete_oral_components() -> None:
    adapter = GNMAdapter()
    rig = ControlRig(
        adapter,
        ExpressionDecoder("gnm/shape/data/semantic_sampler/expression_decoder_model.h5"),
    )
    expressions = (rig.viseme("X"), rig.viseme("F"))
    frames = np.stack([adapter.mesh(expression=value) for value in expressions])
    landmarks = np.stack([rig.compact_landmarks(value) for value in expressions])

    measured = measure_articulation_geometry(frames, landmarks, adapter=adapter)
    streamed = measure_articulation_geometry_from_controls(
        np.stack(expressions),
        np.zeros(adapter.identity_dim, dtype=np.float32),
        landmarks,
        adapter=adapter,
        batch_size=1,
    )

    assert measured.labiodental_gap_interocular.shape == (2,)
    assert measured.mouth_width_interocular.shape == (2,)
    assert np.isfinite(measured.labiodental_gap_interocular).all()
    assert measured.mouth_width_interocular[1] < measured.mouth_width_interocular[0]
    assert measured.labiodental_gap_interocular.flags.writeable is False
    assert np.allclose(
        streamed.labiodental_gap_interocular,
        measured.labiodental_gap_interocular,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.allclose(
        streamed.mouth_width_interocular,
        measured.mouth_width_interocular,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_current_fv_and_tongue_full_surface_proxies_are_not_promoted() -> None:
    adapter = GNMAdapter()
    rig = ControlRig(
        adapter,
        ExpressionDecoder("gnm/shape/data/semantic_sampler/expression_decoder_model.h5"),
    )
    expressions = np.stack((rig.viseme("X"), rig.viseme("G"), rig.viseme("H")))
    frames = np.stack([adapter.mesh(expression=value) for value in expressions])
    landmarks = np.stack([rig.compact_landmarks(value) for value in expressions])
    proximity = measure_articulation_geometry(frames, landmarks, adapter=adapter)
    oral = validate_oral_frames(
        frames,
        adapter=adapter,
        timestamps=np.arange(3, dtype=np.float64) / 30.0,
    )

    # The current semantic prototypes move these unsigned full-surface minima
    # away from their nominal targets. They are telemetry, not qualifiers.
    assert proximity.labiodental_gap_interocular[1] > proximity.labiodental_gap_interocular[0]
    assert oral.tongue_upper_teeth_gap_interocular[2] > oral.tongue_upper_teeth_gap_interocular[0]
    assert _diagnostic_calibration().production_eligible is False


def test_event_detail_is_bounded_and_removed_from_manifest_summary() -> None:
    timestamps, series = _perfect_series()
    report = evaluate_phone_articulation(
        _annotations(),
        timestamps_seconds=timestamps,
        lip_gap_interocular=series["lip"],
        labiodental_gap_interocular=series["labiodental"],
        tongue_upper_teeth_gap_interocular=series["tongue"],
        mouth_width_interocular=series["width"],
        calibration=_diagnostic_calibration(),
        gates=ArticulationGateThresholds(minimum_events_per_family=1),
    )
    summary = summarize_phone_articulation(report)
    assert "events" in report["families"]["bilabial"]
    assert "events" not in summary["families"]["bilabial"]
    assert summary["families"]["bilabial"]["event_detail_count"] == 1

    excessive = replace(
        _annotations(),
        events=(_annotations().events[0],) * (MAX_ARTICULATION_EVENTS + 1),
    )
    with pytest.raises(AutoAnimError, match="bounded diagnostic event count"):
        evaluate_phone_articulation(
            excessive,
            timestamps_seconds=timestamps,
            lip_gap_interocular=series["lip"],
            labiodental_gap_interocular=series["labiodental"],
            tongue_upper_teeth_gap_interocular=series["tongue"],
            mouth_width_interocular=series["width"],
            calibration=_diagnostic_calibration(),
        )


def test_ten_minute_60fps_control_measurement_stays_batched(monkeypatch) -> None:
    frame_count = 10 * 60 * 60
    expression = np.broadcast_to(
        np.zeros((1, 383), dtype=np.float32), (frame_count, 383)
    )
    landmarks = np.zeros((1, 68, 3), dtype=np.float64)
    landmarks[0, 36, 0] = -0.03
    landmarks[0, 45, 0] = 0.03
    landmarks[0, 48, 0] = -0.02
    landmarks[0, 54, 0] = 0.02
    landmarks = np.broadcast_to(landmarks, (frame_count, 68, 3))

    class FakeAdapter:
        expression_dim = 383
        identity_dim = 253
        maximum_batch = 0

        def mesh(self, *, identity, expression):
            self.maximum_batch = max(self.maximum_batch, len(expression))
            return np.zeros((len(expression), 130, 3), dtype=np.float32)

    adapter = FakeAdapter()

    def bounded_measure(frames, points, *, adapter):
        assert len(frames) <= 64
        return ArticulationGeometry(
            labiodental_gap_interocular=np.zeros(len(frames), dtype=np.float64),
            mouth_width_interocular=np.ones(len(frames), dtype=np.float64),
        )

    monkeypatch.setattr(
        phone_articulation_module,
        "measure_articulation_geometry",
        bounded_measure,
    )
    measured = measure_articulation_geometry_from_controls(
        expression,
        np.zeros(253, dtype=np.float32),
        landmarks,
        adapter=adapter,
    )

    assert adapter.maximum_batch == 64
    assert measured.labiodental_gap_interocular.shape == (frame_count,)
    assert measured.mouth_width_interocular.shape == (frame_count,)
