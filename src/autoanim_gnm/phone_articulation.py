"""Fail-closed, multi-articulator diagnostics for phone-timed GNM motion.

The legacy phone timing report intentionally measures only bilabial lip gap.
This module adds the broader A1 evidence lane without changing animation.  It
measures four geometry signals against a bound phone timeline:

* inner-lip closure for /p b m/;
* lower-lip/upper-teeth proximity for /f v/;
* tongue/upper-teeth proximity for dental and alveolar phones; and
* mouth narrowing for rounded phones.

The last three are coarse unsigned proxies, not proof of a particular contact
surface or visible articulation. Normal phone intervals are acoustic context,
not independently labelled articulatory onset/contact/release truth. This
module therefore cannot emit production approval. It records proposed proxy
metrics while the required contact-state corpus, character-bound target
surfaces, protrusion measurement, and verified calibration format are absent.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy
from scipy.spatial import cKDTree

from .errors import AutoAnimError
from .gnm_adapter import GNMAdapter
from .phone_events import PhoneAnnotationSet, PhoneEvent, TICKS_PER_SECOND


PHONE_ARTICULATION_REPORT_SCHEMA_VERSION = "autoanim.phone-articulation-report/1.0"
PHONE_ARTICULATION_VERIFIER_ALGORITHM = (
    "autoanim.phone-articulation-report/1.0-diagnostic"
)
_TONGUE_UPPER_CONTACT_PHONES = frozenset(("T", "D", "N", "L", "TH", "DH"))
MAX_ARTICULATION_EVENTS = 10_000


def _invalid(message: str, **details: object) -> AutoAnimError:
    return AutoAnimError("PHONE_ARTICULATION_INVALID", message, details)


@dataclass(frozen=True, slots=True)
class ArticulationCalibration:
    """Thresholds plus the evidence needed to interpret them.

    Only the diagnostic profile exists in this schema. A future production
    profile must be introduced through a separate canonical artifact loader
    that verifies actual bytes, character/identity/rig/model bindings, target
    surfaces, and an immutable approval record. Caller-supplied booleans are
    deliberately not accepted as trust evidence.
    """

    bilabial_gap_interocular: float
    labiodental_gap_interocular: float
    tongue_upper_teeth_gap_interocular: float
    rounded_mouth_width_interocular: float
    source: str = "diagnostic_default"

    def __post_init__(self) -> None:
        values = (
            self.bilabial_gap_interocular,
            self.labiodental_gap_interocular,
            self.tongue_upper_teeth_gap_interocular,
            self.rounded_mouth_width_interocular,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise _invalid("Articulation thresholds must be finite and positive")
        if self.source != "diagnostic_default":
            raise _invalid("Articulation calibration source is unsupported")

    @property
    def production_eligible(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "profile_sha256": None,
            "artist_approved": False,
            "identity_and_rig_binding_verified": False,
            "production_eligible": self.production_eligible,
            "thresholds": {
                "bilabial_gap_interocular": self.bilabial_gap_interocular,
                "labiodental_gap_interocular": self.labiodental_gap_interocular,
                "tongue_upper_teeth_gap_interocular": (
                    self.tongue_upper_teeth_gap_interocular
                ),
                "rounded_mouth_width_interocular": (
                    self.rounded_mouth_width_interocular
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class ArticulationGeometry:
    """One finite, frame-aligned geometry signal per articulation family."""

    labiodental_gap_interocular: np.ndarray
    mouth_width_interocular: np.ndarray


@dataclass(frozen=True, slots=True)
class ArticulationGateThresholds:
    """Proposed diagnostic thresholds, not corpus-qualified release values."""

    minimum_events_per_family: int = 100
    apex_median_frames: float = 1.0
    apex_p95_frames: float = 2.0
    duration_median_frames: float = 1.0
    duration_p95_frames: float = 2.0
    boundary_median_ms: float = 40.0
    boundary_p95_ms: float = 80.0
    bilabial_f1: float = 0.90
    labiodental_f1: float = 0.85
    tongue_f1: float = 0.85
    rounded_f1: float = 0.85
    false_positive_fraction: float = 0.01
    transition_guard_frames: int = 1

    def __post_init__(self) -> None:
        rates = (
            self.bilabial_f1,
            self.labiodental_f1,
            self.tongue_f1,
            self.rounded_f1,
        )
        if self.minimum_events_per_family < 1 or self.transition_guard_frames < 0:
            raise _invalid("Articulation gate counts must be non-negative")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in rates):
            raise _invalid("Articulation rate thresholds must lie in [0,1]")
        if not (
            math.isfinite(self.false_positive_fraction)
            and 0.0 < self.false_positive_fraction <= 1.0
        ):
            raise _invalid(
                "Strict false-positive fraction threshold must lie in (0,1]"
            )
        timing = (
            self.apex_median_frames,
            self.apex_p95_frames,
            self.duration_median_frames,
            self.duration_p95_frames,
            self.boundary_median_ms,
            self.boundary_p95_ms,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in timing):
            raise _invalid("Articulation timing thresholds must be finite and positive")
        if self.apex_median_frames > self.apex_p95_frames:
            raise _invalid("Articulation apex median threshold may not exceed p95")
        if self.duration_median_frames > self.duration_p95_frames:
            raise _invalid("Articulation duration median threshold may not exceed p95")
        if self.boundary_median_ms > self.boundary_p95_ms:
            raise _invalid("Articulation boundary median threshold may not exceed p95")


def diagnostic_articulation_calibration(
    *,
    bilabial_gap_interocular: float,
    neutral_landmarks: np.ndarray,
    rounded_landmarks: np.ndarray,
) -> ArticulationCalibration:
    """Build the deterministic, explicitly unapproved instrumentation profile."""

    neutral = np.asarray(neutral_landmarks, dtype=np.float64)
    rounded = np.asarray(rounded_landmarks, dtype=np.float64)
    if (
        neutral.shape != (68, 3)
        or rounded.shape != (68, 3)
        or not np.isfinite(neutral).all()
        or not np.isfinite(rounded).all()
    ):
        raise _invalid("Diagnostic articulation calibration requires two finite 68-point poses")

    def width(points: np.ndarray) -> float:
        interocular = float(np.linalg.norm(points[36] - points[45]))
        if interocular <= 1.0e-6:
            raise _invalid("Diagnostic articulation calibration has a degenerate face scale")
        return float(np.linalg.norm(points[48] - points[54]) / interocular)

    neutral_width = width(neutral)
    rounded_width = width(rounded)
    if rounded_width >= neutral_width:
        raise _invalid("Diagnostic rounded prototype does not narrow the mouth")
    return ArticulationCalibration(
        bilabial_gap_interocular=bilabial_gap_interocular,
        # These are deliberately conservative proximity diagnostics. They are
        # not claimed as calibrated contact thresholds.
        labiodental_gap_interocular=0.012,
        tongue_upper_teeth_gap_interocular=0.010,
        rounded_mouth_width_interocular=0.5 * (neutral_width + rounded_width),
    )


def measure_articulation_geometry(
    frames: np.ndarray,
    landmarks: np.ndarray,
    *,
    adapter: GNMAdapter,
) -> ArticulationGeometry:
    """Measure robust labiodental proximity and mouth width on evaluated GNM."""

    vertices = np.asarray(frames, dtype=np.float32)
    points = np.asarray(landmarks, dtype=np.float64)
    if (
        vertices.ndim != 3
        or vertices.shape[1:] != (adapter.model.num_vertices, 3)
        or points.shape != (len(vertices), 68, 3)
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or not np.isfinite(points).all()
    ):
        raise _invalid("Articulation geometry must be finite, complete GNM frames")
    interocular = np.linalg.norm(points[:, 36] - points[:, 45], axis=1)
    if np.any(interocular <= 1.0e-6):
        raise _invalid("Articulation geometry has a degenerate interocular scale")
    lower_lip = np.flatnonzero(adapter.vertex_group("lower_lip") > 0.5)
    upper_teeth = np.flatnonzero(
        (adapter.vertex_group("teeth") > 0.5)
        & (adapter.vertex_group("upper_teeth_and_gums") > 0.5)
    )
    if len(lower_lip) < 20 or len(upper_teeth) < 100:
        raise _invalid("Required lower-lip or upper-teeth geometry is absent")
    labiodental = np.empty(len(vertices), dtype=np.float64)
    for frame_index, frame in enumerate(vertices):
        distances, _ = cKDTree(frame[upper_teeth]).query(
            frame[lower_lip], k=1, workers=1
        )
        # The first percentile is less sensitive than one closest vertex while
        # still responding to a local lower-lip/upper-incisor approach.
        labiodental[frame_index] = float(np.percentile(distances, 1)) / interocular[
            frame_index
        ]
    width = np.linalg.norm(points[:, 48] - points[:, 54], axis=1) / interocular
    labiodental.setflags(write=False)
    width.setflags(write=False)
    return ArticulationGeometry(
        labiodental_gap_interocular=labiodental,
        mouth_width_interocular=width,
    )


def measure_articulation_geometry_from_controls(
    expression: np.ndarray,
    identity: np.ndarray,
    landmarks: np.ndarray,
    *,
    adapter: GNMAdapter,
    batch_size: int = 64,
) -> ArticulationGeometry:
    """Stream GNM control evaluation while retaining only two scalar tracks."""

    expressions = np.asarray(expression, dtype=np.float32)
    identity_value = np.asarray(identity, dtype=np.float32)
    points = np.asarray(landmarks, dtype=np.float64)
    if (
        expressions.ndim != 2
        or expressions.shape[1] != adapter.expression_dim
        or not len(expressions)
        or identity_value.shape != (adapter.identity_dim,)
        or points.shape != (len(expressions), 68, 3)
        or not np.isfinite(expressions).all()
        or not np.isfinite(identity_value).all()
        or not np.isfinite(points).all()
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise _invalid("Articulation controls must be finite, complete, and bounded")
    labiodental = np.empty(len(expressions), dtype=np.float64)
    width = np.empty(len(expressions), dtype=np.float64)
    for start in range(0, len(expressions), batch_size):
        stop = min(start + batch_size, len(expressions))
        identity_batch = np.broadcast_to(
            identity_value, (stop - start, adapter.identity_dim)
        )
        frames = adapter.mesh(
            identity=identity_batch,
            expression=expressions[start:stop],
        )
        measured = measure_articulation_geometry(
            frames,
            points[start:stop],
            adapter=adapter,
        )
        labiodental[start:stop] = measured.labiodental_gap_interocular
        width[start:stop] = measured.mouth_width_interocular
    labiodental.setflags(write=False)
    width.setflags(write=False)
    return ArticulationGeometry(
        labiodental_gap_interocular=labiodental,
        mouth_width_interocular=width,
    )


def articulation_evidence_bindings(
    *,
    controls_path: str | Path,
    identity: np.ndarray,
    gnm_asset_path: str | Path,
    landmark_regressor_path: str | Path,
    expression_decoder_path: str | Path,
    character_revision_manifest_sha256: str | None,
) -> dict[str, str | None]:
    """Bind diagnostics to exact controls, identity, GNM, decoder, and character."""

    identity_value = np.asarray(identity, dtype="<f4")
    if identity_value.shape != (253,) or not np.isfinite(identity_value).all():
        raise _invalid("Articulation binding identity is invalid")

    def file_digest(value: str | Path) -> str:
        path = Path(value)
        if not path.is_file():
            raise _invalid("Articulation binding asset is absent", path=str(path))
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if character_revision_manifest_sha256 is not None and (
        len(character_revision_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in character_revision_manifest_sha256)
    ):
        raise _invalid("Character revision binding digest is invalid")
    verifier_sources = tuple(
        Path(__file__).with_name(name)
        for name in (
            "gnm_adapter.py",
            "oral_validation.py",
            "phone_articulation.py",
            "phone_events.py",
        )
    )
    verifier_bundle = sha256()
    for source in verifier_sources:
        verifier_bundle.update(source.name.encode("utf-8"))
        verifier_bundle.update(bytes.fromhex(file_digest(source)))
    return {
        "controls_sha256": file_digest(controls_path),
        "identity_f32le_sha256": sha256(identity_value.tobytes(order="C")).hexdigest(),
        "gnm_head_asset_sha256": file_digest(gnm_asset_path),
        "landmark_regressor_sha256": file_digest(landmark_regressor_path),
        "expression_decoder_sha256": file_digest(expression_decoder_path),
        "character_revision_manifest_sha256": character_revision_manifest_sha256,
        "verifier_algorithm": PHONE_ARTICULATION_VERIFIER_ALGORITHM,
        "verifier_source_sha256": file_digest(Path(__file__)),
        "verifier_bundle_sha256": verifier_bundle.hexdigest(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def _validate_series(
    timestamps_seconds: np.ndarray,
    series: Mapping[str, np.ndarray],
    *,
    duration_ticks: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int]:
    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    if (
        timestamps.ndim != 1
        or len(timestamps) < 2
        or not np.isfinite(timestamps).all()
        or timestamps[0] != 0.0
        or np.any(np.diff(timestamps) <= 0.0)
    ):
        raise _invalid("Articulation timestamps are invalid")
    if not isinstance(duration_ticks, int) or duration_ticks <= 0:
        raise _invalid("Articulation annotation duration is invalid")
    if timestamps[-1] > np.iinfo(np.int64).max / TICKS_PER_SECOND:
        raise _invalid("Articulation timestamps exceed the exact tick range")
    ticks = np.rint(timestamps * TICKS_PER_SECOND).astype(np.int64)
    tick_steps = np.diff(ticks)
    allowed_frame_ticks = (TICKS_PER_SECOND // 60, TICKS_PER_SECOND // 30)
    if (
        ticks[0] != 0
        or len(np.unique(tick_steps)) != 1
        or int(tick_steps[0]) not in allowed_frame_ticks
        or int(ticks[-1]) > duration_ticks
        or duration_ticks - int(ticks[-1]) > int(tick_steps[0])
    ):
        raise _invalid("Articulation timestamps must be a complete exact 30 or 60 fps clock")
    validated: dict[str, np.ndarray] = {}
    for name, value in series.items():
        array = np.asarray(value, dtype=np.float64)
        if (
            array.shape != timestamps.shape
            or not np.isfinite(array).all()
            or np.any(array < 0.0)
            or (name == "mouth_width_interocular" and np.any(array <= 0.0))
        ):
            raise _invalid("Articulation metric is not finite and frame-aligned", metric=name)
        validated[name] = array
    return timestamps, ticks, validated, int(tick_steps[0])


def _stats(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)) if len(array) else None,
        "p95": float(np.percentile(array, 95)) if len(array) else None,
    }


def _event_family(event: PhoneEvent, family: str) -> bool:
    if family == "bilabial":
        return event.is_bilabial
    if family == "labiodental":
        return event.is_labiodental
    if family == "tongue_upper_teeth":
        # S/Z share the alveolar place label but require a groove rather than
        # full tongue contact, so they must not be scored as closure targets.
        return event.phone in _TONGUE_UPPER_CONTACT_PHONES
    if family == "rounded":
        return event.rounded is True
    raise AssertionError(family)


def _classification(
    predicted: np.ndarray,
    expected: np.ndarray,
) -> dict[str, float | int | None]:
    true_positive = int(np.count_nonzero(predicted & expected))
    false_positive = int(np.count_nonzero(predicted & ~expected))
    false_negative = int(np.count_nonzero(~predicted & expected))
    negative_frames = int(np.count_nonzero(~expected))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else None
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "true_positive_frames": true_positive,
        "false_positive_frames": false_positive,
        "false_negative_frames": false_negative,
        "negative_frames": negative_frames,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_fraction": (
            false_positive / negative_frames if negative_frames else None
        ),
    }


def _family_report(
    family: str,
    *,
    annotations: PhoneAnnotationSet,
    ticks: np.ndarray,
    signal: np.ndarray,
    threshold: float,
    guard_ticks: int,
    frame_ticks: int,
) -> dict[str, Any]:
    target_events = tuple(
        event
        for event in annotations.events
        if not event.is_silence and _event_family(event, family)
    )
    if len(target_events) > MAX_ARTICULATION_EVENTS:
        raise _invalid(
            "Articulation family exceeds the bounded diagnostic event count",
            family=family,
            maximum=MAX_ARTICULATION_EVENTS,
        )
    # Build the contextual phone-span mask in O(frames + events). This is not
    # independent contact-state truth and is labelled accordingly in output.
    expected_delta = np.zeros(len(ticks) + 1, dtype=np.int32)
    for event in target_events:
        left = int(np.searchsorted(ticks, event.start_tick - guard_ticks, side="left"))
        right = int(np.searchsorted(ticks, event.end_tick + guard_ticks, side="right"))
        if left < right:
            expected_delta[left] += 1
            expected_delta[right] -= 1
    expected = np.cumsum(expected_delta[:-1]) > 0
    predicted = signal <= threshold
    transitions = np.diff(np.pad(predicted.astype(np.int8), (1, 1)))
    run_starts = np.flatnonzero(transitions == 1)
    run_ends = np.flatnonzero(transitions == -1)
    event_reports: list[dict[str, Any]] = []
    reviewed_apex_errors_ms: list[float] = []
    reviewed_onset_errors_ms: list[float] = []
    reviewed_release_errors_ms: list[float] = []
    reviewed_duration_errors_ms: list[float] = []
    found_count = 0
    for event in target_events:
        left = int(np.searchsorted(ticks, event.start_tick - guard_ticks, side="left"))
        right = int(np.searchsorted(ticks, event.end_tick + guard_ticks, side="right"))
        if left >= right:
            event_reports.append(
                {
                    "id": event.event_id,
                    "phone": event.phone,
                    "scored": False,
                    "failure": "no_animation_sample_inside_guarded_event",
                }
            )
            continue
        indices = np.arange(left, right, dtype=np.int64)
        best_index = int(indices[np.argmin(signal[indices])])
        predicted_indices = indices[predicted[indices]]
        proxy_active = bool(len(predicted_indices))
        found_count += int(proxy_active)
        onset_tick: int | None = None
        release_tick: int | None = None
        left_censored: bool | None = None
        right_censored: bool | None = None
        if proxy_active:
            # Select a component that intersects the contextual phone window,
            # then measure its UNCLIPPED global boundaries. This exposes early
            # or persistent proxy activation instead of hiding it at the guard.
            selected_index = int(
                predicted_indices[np.argmin(np.abs(predicted_indices - best_index))]
            )
            run_index = int(np.searchsorted(run_starts, selected_index, side="right") - 1)
            if run_index < 0 or selected_index >= int(run_ends[run_index]):
                raise _invalid("Predicted articulation component indexing failed")
            onset_tick = int(ticks[int(run_starts[run_index])])
            release_tick = int(ticks[int(run_ends[run_index] - 1)])
            left_censored = bool(run_starts[run_index] == 0)
            right_censored = bool(run_ends[run_index] == len(ticks))
        predicted_apex_tick = int(ticks[best_index])
        apex_error_ms = abs(predicted_apex_tick - event.apex_tick) / 48.0
        onset_error_ms = (
            abs(onset_tick - event.start_tick) / 48.0
            if onset_tick is not None and not left_censored
            else None
        )
        release_error_ms = (
            abs(release_tick - event.end_tick) / 48.0
            if release_tick is not None and not right_censored
            else None
        )
        onset_bias_ms = (
            (onset_tick - event.start_tick) / 48.0
            if onset_tick is not None and not left_censored
            else None
        )
        release_bias_ms = (
            (release_tick - event.end_tick) / 48.0
            if release_tick is not None and not right_censored
            else None
        )
        if event.apex_reviewed:
            reviewed_apex_errors_ms.append(apex_error_ms)
        if annotations.independently_reviewed and proxy_active:
            if not left_censored:
                reviewed_onset_errors_ms.append(float(onset_error_ms))
            if not right_censored:
                reviewed_release_errors_ms.append(float(release_error_ms))
            if not left_censored and not right_censored:
                reviewed_duration_errors_ms.append(
                    abs(
                        (release_tick - onset_tick)
                        - (event.end_tick - event.start_tick)
                    )
                    / 48.0
                )
        event_reports.append(
            {
                "id": event.event_id,
                "phone": event.phone,
                "word": event.word,
                "scored": True,
                "apex_reviewed": event.apex_reviewed,
                "proxy_active_in_context": proxy_active,
                "minimum_proxy_signal_tick": predicted_apex_tick,
                "minimum_proxy_signal_vs_phone_apex_error_ms": apex_error_ms,
                "proxy_run_onset_tick": onset_tick,
                "proxy_run_release_tick": release_tick,
                "proxy_run_left_censored": left_censored,
                "proxy_run_right_censored": right_censored,
                "proxy_run_vs_phone_start_absolute_error_ms": onset_error_ms,
                "proxy_run_vs_phone_end_absolute_error_ms": release_error_ms,
                "proxy_run_vs_phone_start_bias_ms": onset_bias_ms,
                "proxy_run_vs_phone_end_bias_ms": release_bias_ms,
                "minimum_signal_interocular": float(signal[best_index]),
            }
        )
    apex_ms = _stats(reviewed_apex_errors_ms)
    onset_ms = _stats(reviewed_onset_errors_ms)
    release_ms = _stats(reviewed_release_errors_ms)
    duration_ms = _stats(reviewed_duration_errors_ms)
    milliseconds_per_frame = frame_ticks / 48.0
    return {
        "family": family,
        "reference_labels": "phone_span_context_not_articulation_state_truth",
        "event_count": len(target_events),
        "proxy_active_event_count": found_count,
        "proxy_active_event_recall": (
            found_count / len(target_events) if target_events else None
        ),
        "reviewed_phone_apex_event_count": len(reviewed_apex_errors_ms),
        "threshold_interocular": threshold,
        "classification": {
            **_classification(predicted, expected),
            "negative_label_source": "outside_contextual_phone_spans_not_independently_labelled",
        },
        "minimum_proxy_signal_vs_reviewed_phone_apex_error_ms": apex_ms,
        "minimum_proxy_signal_vs_reviewed_phone_apex_error_frames": {
            key: value / milliseconds_per_frame if value is not None else None
            for key, value in apex_ms.items()
        },
        "phone_span_onset_error_ms": onset_ms,
        "phone_span_release_error_ms": release_ms,
        "phone_span_duration_error_ms": duration_ms,
        "phone_span_duration_error_frames": {
            key: value / milliseconds_per_frame if value is not None else None
            for key, value in duration_ms.items()
        },
        "events": event_reports,
    }


def evaluate_phone_articulation(
    annotations: PhoneAnnotationSet,
    *,
    timestamps_seconds: np.ndarray,
    lip_gap_interocular: np.ndarray,
    labiodental_gap_interocular: np.ndarray,
    tongue_upper_teeth_gap_interocular: np.ndarray,
    mouth_width_interocular: np.ndarray,
    calibration: ArticulationCalibration,
    evidence_bindings: Mapping[str, str | None] | None = None,
    gates: ArticulationGateThresholds = ArticulationGateThresholds(),
) -> dict[str, Any]:
    """Evaluate four phone-conditioned geometry lanes without authoring motion."""

    timestamps, ticks, series, frame_ticks = _validate_series(
        timestamps_seconds,
        {
            "lip_gap_interocular": lip_gap_interocular,
            "labiodental_gap_interocular": labiodental_gap_interocular,
            "tongue_upper_teeth_gap_interocular": tongue_upper_teeth_gap_interocular,
            "mouth_width_interocular": mouth_width_interocular,
        },
        duration_ticks=annotations.duration_ticks,
    )
    guard_ticks = gates.transition_guard_frames * frame_ticks
    family_inputs = {
        "bilabial": (
            series["lip_gap_interocular"],
            calibration.bilabial_gap_interocular,
        ),
        "labiodental": (
            series["labiodental_gap_interocular"],
            calibration.labiodental_gap_interocular,
        ),
        "tongue_upper_teeth": (
            series["tongue_upper_teeth_gap_interocular"],
            calibration.tongue_upper_teeth_gap_interocular,
        ),
        "rounded": (
            series["mouth_width_interocular"],
            calibration.rounded_mouth_width_interocular,
        ),
    }
    families = {
        family: _family_report(
            family,
            annotations=annotations,
            ticks=ticks,
            signal=signal,
            threshold=threshold,
            guard_ticks=guard_ticks,
            frame_ticks=frame_ticks,
        )
        for family, (signal, threshold) in family_inputs.items()
    }
    proxy_failures: list[str] = []
    if not annotations.production_review_complete:
        proxy_failures.append("independent_reviewed_phone_boundaries_and_apexes")
    f1_thresholds = {
        "bilabial": gates.bilabial_f1,
        "labiodental": gates.labiodental_f1,
        "tongue_upper_teeth": gates.tongue_f1,
        "rounded": gates.rounded_f1,
    }
    for family, report in families.items():
        if report["event_count"] < gates.minimum_events_per_family:
            proxy_failures.append(f"{family}_minimum_event_count")
        classification = report["classification"]
        f1 = classification["f1"]
        if f1 is None or f1 < f1_thresholds[family]:
            proxy_failures.append(f"{family}_f1")
        event_recall = report["proxy_active_event_recall"]
        if event_recall is None or event_recall < f1_thresholds[family]:
            proxy_failures.append(f"{family}_proxy_active_event_recall")
        false_positive = classification["false_positive_fraction"]
        if false_positive is None or false_positive >= gates.false_positive_fraction:
            proxy_failures.append(f"{family}_false_positive_fraction")
        apex = report[
            "minimum_proxy_signal_vs_reviewed_phone_apex_error_frames"
        ]
        if (
            apex["median"] is None
            or apex["median"] > gates.apex_median_frames + 1.0e-9
        ):
            proxy_failures.append(f"{family}_apex_median")
        if (
            apex["p95"] is None
            or apex["p95"] > gates.apex_p95_frames + 1.0e-9
        ):
            proxy_failures.append(f"{family}_apex_p95")
        duration_error = report["phone_span_duration_error_frames"]
        if (
            duration_error["median"] is None
            or duration_error["median"] > gates.duration_median_frames + 1.0e-9
        ):
            proxy_failures.append(f"{family}_duration_median")
        if (
            duration_error["p95"] is None
            or duration_error["p95"] > gates.duration_p95_frames + 1.0e-9
        ):
            proxy_failures.append(f"{family}_duration_p95")
        for boundary in (
            "phone_span_onset_error_ms",
            "phone_span_release_error_ms",
        ):
            timing = report[boundary]
            label = "onset" if "onset" in boundary else "release"
            if timing["median"] is None or timing["median"] > gates.boundary_median_ms:
                proxy_failures.append(f"{family}_{label}_median")
            if timing["p95"] is None or timing["p95"] > gates.boundary_p95_ms:
                proxy_failures.append(f"{family}_{label}_p95")

    production_failures = [
        "independent_articulation_state_annotations_not_implemented",
        "verified_character_articulation_profile_not_implemented",
        "validated_labiodental_and_tongue_target_surfaces_not_implemented",
        "lip_protrusion_measurement_not_implemented",
        "independent_perceptual_readability_not_implemented",
    ]
    if proxy_failures:
        production_failures.append("phone_span_proxy_gate_failed")

    return {
        "schema_version": PHONE_ARTICULATION_REPORT_SCHEMA_VERSION,
        "status": (
            "phone_span_proxy_metrics_failed_not_production_qualified"
            if proxy_failures
            else "phone_span_proxy_metrics_passed_not_production_qualified"
        ),
        "annotation_bindings": {
            "textgrid_sha256": annotations.source_textgrid_sha256,
            "audio_sha256": annotations.source_audio_sha256,
        },
        "annotation_semantics": {
            "phone_intervals": "context_only_not_articulation_state_truth",
            "outside_phone_intervals": "not_independently_labelled_negative_states",
            "apex_points": (
                "reviewed_points_when_present"
                if annotations.apex_tier is not None
                else "interval_midpoints_not_independently_reviewed"
            ),
        },
        "evidence_bindings": dict(evidence_bindings or {}),
        "timebase": {
            "ticks_per_second": TICKS_PER_SECOND,
            "frame_count": len(timestamps),
            "frame_ticks": frame_ticks,
            "fps": TICKS_PER_SECOND // frame_ticks,
            "transition_guard_frames": gates.transition_guard_frames,
            "transition_guard_ticks": guard_ticks,
        },
        "calibration": calibration.as_dict(),
        "gate_thresholds": {
            "scope": "proposed_diagnostic_phone_span_proxy_thresholds",
            "minimum_events_per_family": gates.minimum_events_per_family,
            "apex_median_frames": gates.apex_median_frames,
            "apex_p95_frames": gates.apex_p95_frames,
            "phone_span_duration_median_frames": gates.duration_median_frames,
            "phone_span_duration_p95_frames": gates.duration_p95_frames,
            "boundary_median_ms": gates.boundary_median_ms,
            "boundary_p95_ms": gates.boundary_p95_ms,
            "f1": f1_thresholds,
            "proxy_active_event_recall": f1_thresholds,
            "false_positive_fraction_strictly_below": gates.false_positive_fraction,
        },
        "families": families,
        "phone_span_proxy_gate": {
            "passed": not proxy_failures,
            "failures": proxy_failures,
        },
        "production_gate": {
            "passed": False,
            "failures": production_failures,
        },
        "claims": {
            "animation_authored_by_annotations": False,
            "inner_lip_closure_measured": True,
            "labiodental_proximity_measured": True,
            "tongue_upper_teeth_proximity_measured": True,
            "rounded_mouth_narrowing_measured": True,
            "exact_surface_intersection_validated": False,
            "tongue_target_surface_validated": False,
            "lip_protrusion_validated": False,
            "perceptual_readability_validated": False,
            "phone_span_proxy_gate_passed": not proxy_failures,
            "phone_articulation_gate_passed": False,
            "production_validated": False,
        },
        "limitations": [
            "Phone intervals are acoustic context, not independently labelled articulatory state intervals or negative frames.",
            "Labiodental and tongue metrics are unsigned full-surface proximity proxies with no validated positive/negative prototype separation.",
            "Dental and alveolar phones are conflated against upper-teeth geometry; target surfaces are not anatomical qualifiers.",
            "Rounded-phone evidence measures mouth narrowing, not lip protrusion or a full artist target.",
            "Frame classification uses one-frame phone-span dilation and can only be interpreted as diagnostic context agreement.",
            "The proposed numerical thresholds are not calibrated on a held-out, speaker-balanced corpus.",
            "Independent contact-state labels, perceptual readability, and animator approval are outside this report.",
        ],
    }


def summarize_phone_articulation(report: Mapping[str, Any]) -> dict[str, Any]:
    """Strip per-event rows for the signed job manifest; the artifact keeps them."""

    summary = copy.deepcopy(dict(report))
    families = summary.get("families")
    if not isinstance(families, dict):
        raise _invalid("Articulation report families are absent")
    for family in families.values():
        if not isinstance(family, dict) or not isinstance(family.get("events"), list):
            raise _invalid("Articulation report event detail is invalid")
        family["event_detail_count"] = len(family["events"])
        family.pop("events")
    summary["event_detail_artifact"] = "phone-articulation-report.json"
    return summary


__all__ = [
    "PHONE_ARTICULATION_REPORT_SCHEMA_VERSION",
    "PHONE_ARTICULATION_VERIFIER_ALGORITHM",
    "ArticulationCalibration",
    "ArticulationGateThresholds",
    "ArticulationGeometry",
    "MAX_ARTICULATION_EVENTS",
    "articulation_evidence_bindings",
    "diagnostic_articulation_calibration",
    "evaluate_phone_articulation",
    "measure_articulation_geometry",
    "measure_articulation_geometry_from_controls",
    "summarize_phone_articulation",
]
