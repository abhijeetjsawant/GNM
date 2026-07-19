"""Fail-closed qualification of an existing GNM lipsync control track.

This module deliberately does not generate or modify animation.  It binds an
independently authored phonetic-event tier and artist-approved GNM landmark
targets to exact source, character, identity, rig, timebase, and provenance
artifacts, then evaluates a retained ``controls.npz`` through
``lipsync_quality.evaluate_lipsync_quality``.

The evidence declaration is necessary but not magically sufficient: the
software can verify bound-by-hash artifacts and explicit independence
statements, but it cannot independently prove a human annotator's identity or
workflow.  Missing, self-scored, mismatched, malformed, or unapproved evidence
therefore fails closed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from .artifacts import sha256
from .lipsync_quality import (
    LipsyncQualityReport,
    QualityThresholds,
    TimingAnnotation,
    evaluate_lipsync_quality,
)
from .rig import ControlRig


PROFILE_SCHEMA = "autoanim.lipsync-qualification/1.0"
REPORT_SCHEMA = "autoanim.lipsync-qualification-report/1.0"
_MAX_PROFILE_BYTES = 8 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_ALLOWED_ANNOTATION_METHODS = frozenset(
    ("manual_phonetic_annotation", "manual_performance_capture_annotation")
)
_QUALITY_THRESHOLD_FIELDS = (
    "mouth_step_max_interocular",
    "speech_active_stationary_fraction",
    "neutral_return_frames",
    "false_silence_motion_ratio_p95",
    "target_contrast_median",
    "target_contrast_p10",
    "timing_error_median_frames",
    "timing_error_p95_frames",
    "minimum_independent_events",
)


class LipsyncQualificationError(ValueError):
    """A profile, evidence binding, or retained control track is invalid."""

    def __init__(self, code: str, message: str, *, field: str | None = None):
        super().__init__(message)
        self.code = code
        self.field = field


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    artifact_id: str
    sha256: str


@dataclass(frozen=True, slots=True)
class QualificationBinding:
    source_audio_sha256: str
    character_manifest_sha256: str
    identity_artifact_sha256: str
    identity_array_sha256: str
    rig_sha256: str


@dataclass(frozen=True, slots=True)
class QualificationTimebase:
    units: str
    fps_numerator: int
    fps_denominator: int
    frame_count: int
    timestamp_origin_seconds: float

    @property
    def fps(self) -> float:
        return self.fps_numerator / self.fps_denominator


@dataclass(frozen=True, slots=True)
class ProfileProvenance:
    curator_id: str
    created_at: str
    protocol: str


@dataclass(frozen=True, slots=True)
class AnnotationProvenance:
    annotator_id: str
    annotator_organization: str
    created_at: str
    method: str
    independent_from_animation_system: bool
    viewed_system_cues: bool
    viewed_generated_animation: bool
    used_system_output_as_timing_source: bool
    evidence_artifact: ArtifactReference


@dataclass(frozen=True, slots=True)
class QualificationEvent:
    event_id: str
    label: str
    start_seconds: float
    apex_seconds: float
    release_seconds: float


@dataclass(frozen=True, slots=True)
class PrototypeProvenance:
    artist_id: str
    artist_organization: str
    created_at: str
    approved_at: str
    authoring_tool: str
    coordinate_space: str
    artist_approved: bool
    source_artifact: ArtifactReference
    approval_artifact: ArtifactReference


@dataclass(frozen=True, slots=True)
class TargetPrototype:
    label: str
    landmarks: np.ndarray
    landmarks_sha256: str
    provenance: PrototypeProvenance


@dataclass(frozen=True, slots=True)
class QualificationEvaluator:
    quality_thresholds: QualityThresholds
    stationary_step_interocular: float
    neutral_tolerance_interocular: float
    silence_guard_frames: int
    timing_search_frames: int


@dataclass(frozen=True, slots=True)
class LipsyncQualificationProfile:
    schema_version: str
    binding: QualificationBinding
    timebase: QualificationTimebase
    profile_provenance: ProfileProvenance
    annotation_provenance: AnnotationProvenance
    annotations: tuple[QualificationEvent, ...]
    target_prototypes: Mapping[str, TargetPrototype]
    evaluator: QualificationEvaluator
    profile_sha256: str

    def required_evidence(self) -> Mapping[str, str]:
        references = [self.annotation_provenance.evidence_artifact]
        for prototype in self.target_prototypes.values():
            references.extend(
                (prototype.provenance.source_artifact, prototype.provenance.approval_artifact)
            )
        required: dict[str, str] = {}
        for reference in references:
            previous = required.setdefault(reference.artifact_id, reference.sha256)
            if previous != reference.sha256:
                raise LipsyncQualificationError(
                    "INVALID_PROVENANCE",
                    f"Evidence artifact {reference.artifact_id!r} has conflicting hashes",
                    field=f"evidence.{reference.artifact_id}",
                )
        return required


@dataclass(frozen=True, slots=True)
class LipsyncQualificationReport:
    profile_sha256: str
    controls_sha256: str
    source_audio_sha256: str
    character_manifest_sha256: str
    identity_artifact_sha256: str
    identity_array_sha256: str
    rig_sha256: str
    evidence_sha256s: Mapping[str, str]
    quality: LipsyncQualityReport
    report_sha256: str

    @property
    def core_quality_gate_passed(self) -> bool:
        return self.quality.production_gate.passed

    @property
    def production_validated(self) -> bool:
        # Schema v1 scores independently annotated pose apexes plus motion
        # hygiene.  It does not yet score event start/release, closure duration,
        # context-dependent transition shape, or blinded perceptual approval.
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA,
            "profile_sha256": self.profile_sha256,
            "controls_sha256": self.controls_sha256,
            "source_audio_sha256": self.source_audio_sha256,
            "character_manifest_sha256": self.character_manifest_sha256,
            "identity_artifact_sha256": self.identity_artifact_sha256,
            "identity_array_sha256": self.identity_array_sha256,
            "rig_sha256": self.rig_sha256,
            "evidence_sha256s": dict(sorted(self.evidence_sha256s.items())),
            "binding_verified": True,
            "independent_annotations_verified": True,
            "artist_prototypes_verified": True,
            "quality": self.quality.as_dict(),
            "core_quality_gate_passed": self.core_quality_gate_passed,
            "sequence_timing_validated": False,
            "perceptual_validation_completed": False,
            "production_validated": self.production_validated,
            "qualification_scope": (
                "independent_apex_pose_and_motion_hygiene_only"
            ),
            "report_sha256": self.report_sha256,
        }


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LipsyncQualificationError(
            "INVALID_PROFILE", "Qualification profile is not canonical JSON"
        ) from exc


def profile_payload_sha256(document: Mapping[str, Any]) -> str:
    """Hash all profile content except the self-referential digest field."""

    if not isinstance(document, Mapping):
        raise LipsyncQualificationError("INVALID_PROFILE", "Profile must be a JSON object")
    payload = deepcopy(dict(document))
    payload.pop("profile_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def seal_profile_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied document with its canonical payload digest set."""

    sealed = deepcopy(dict(document))
    sealed.pop("profile_sha256", None)
    sealed["profile_sha256"] = profile_payload_sha256(sealed)
    return sealed


def _array_sha256(domain: str, value: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.dtype(dtype)))
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise LipsyncQualificationError(
            "INVALID_ARRAY", f"{domain} contains nonfinite values"
        )
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)}))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def landmarks_sha256(landmarks: np.ndarray) -> str:
    """Canonical hash for one 68x3 artist target encoded as little-endian f64."""

    value = np.asarray(landmarks, dtype=np.float64)
    if value.shape != (68, 3) or not np.isfinite(value).all():
        raise LipsyncQualificationError(
            "INVALID_PROTOTYPE", "Prototype landmarks must be finite [68,3]"
        )
    return _array_sha256("autoanim.lipsync-prototype-landmarks/1.0", value, "<f8")


def identity_array_sha256(identity: np.ndarray) -> str:
    """Canonical hash for the evaluated GNM identity coefficient vector."""

    value = np.asarray(identity, dtype=np.float32)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise LipsyncQualificationError(
            "INVALID_IDENTITY", "GNM identity must be one finite vector"
        )
    return _array_sha256("autoanim.gnm-identity-array/1.0", value, "<f4")


def load_identity_artifact(
    path: str | Path, *, expected_dimension: int = 253
) -> np.ndarray:
    """Load one bounded numeric identity artifact for qualification CLI use."""

    source = Path(path)
    if not source.is_file() or source.stat().st_size > 4 * 1024 * 1024:
        raise LipsyncQualificationError(
            "INVALID_IDENTITY", "Identity artifact is missing or exceeds 4 MiB"
        )
    try:
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            if (
                [member.filename for member in members] != ["identity.npy"]
                or sum(member.file_size for member in members) > 4 * 1024 * 1024
            ):
                raise LipsyncQualificationError(
                    "INVALID_IDENTITY",
                    "Identity archive must contain only one bounded identity array",
                )
        with np.load(source, allow_pickle=False) as archive:
            identity = np.asarray(archive["identity"])
    except LipsyncQualificationError:
        raise
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise LipsyncQualificationError(
            "INVALID_IDENTITY", "Identity artifact is not a safe numeric NPZ"
        ) from exc
    if (
        identity.shape != (expected_dimension,)
        or identity.dtype.kind not in "fiu"
        or identity.dtype.itemsize > 8
        or not np.isfinite(identity).all()
    ):
        raise LipsyncQualificationError(
            "INVALID_IDENTITY",
            f"Identity artifact must contain one finite ({expected_dimension},) vector",
        )
    output = identity.astype(np.float32, copy=True)
    output.setflags(write=False)
    return output


def runtime_rig_sha256(rig: ControlRig) -> str:
    """Hash exactly the compact GNM/decoder data used by this evaluator.

    Identity is intentionally excluded and bound separately.  The compact
    landmark template/bases are the complete runtime rig used to reconstruct
    the quality track, while the decoder asset hash binds the semantic rig
    revision associated with the character.
    """

    try:
        adapter = rig.adapter
        decoder_path = Path(rig.decoder.model_path)
        version = str(adapter.model.version.value)
        arrays = (
            ("landmark_indices", adapter.landmark_indices, "<i4"),
            ("landmark_weights", adapter.landmark_weights, "<f4"),
            ("compact_template", adapter.compact_template, "<f4"),
            ("compact_identity_basis", adapter.compact_identity_basis, "<f4"),
            ("compact_expression_basis", adapter.compact_expression_basis, "<f4"),
        )
    except (AttributeError, OSError, ValueError) as exc:
        raise LipsyncQualificationError(
            "INVALID_RIG", "Control rig cannot supply qualification geometry"
        ) from exc
    if not decoder_path.is_file():
        raise LipsyncQualificationError(
            "INVALID_RIG", "Expression decoder artifact is missing", field="rig.decoder"
        )
    digest = hashlib.sha256()
    digest.update(b"autoanim.lipsync-runtime-rig/1.0\0")
    digest.update(
        _canonical_json(
            {
                "gnm_version": version,
                "identity_dim": int(adapter.identity_dim),
                "expression_dim": int(adapter.expression_dim),
                "decoder_sha256": sha256(decoder_path),
            }
        )
    )
    for name, array, dtype in arrays:
        digest.update(b"\0")
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(_array_sha256(f"autoanim.rig-array.{name}/1.0", array, dtype).encode("ascii"))
    return digest.hexdigest()


def _pairs_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise LipsyncQualificationError(
                "DUPLICATE_KEY", f"Duplicate JSON key {key!r}", field=key
            )
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise LipsyncQualificationError("INVALID_NUMBER", f"JSON constant {value!r} is forbidden")


def _expect_object(value: Any, field: str, keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LipsyncQualificationError(
            "INVALID_PROFILE", f"{field} must be an object", field=field
        )
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise LipsyncQualificationError(
            "INVALID_PROFILE",
            f"{field} keys differ (missing={missing}, extra={extra})",
            field=field,
        )
    return value


def _expect_sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise LipsyncQualificationError(
            "INVALID_PROFILE", f"{field} must be an array", field=field
        )
    return value


def _expect_string(value: Any, field: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise LipsyncQualificationError(
            "INVALID_PROFILE", f"{field} must be a nonempty string", field=field
        )
    return value


def _expect_boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise LipsyncQualificationError(
            "INVALID_PROFILE", f"{field} must be boolean", field=field
        )
    return value


def _expect_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LipsyncQualificationError(
            "INVALID_PROFILE", f"{field} must be numeric", field=field
        )
    output = float(value)
    if not np.isfinite(output):
        raise LipsyncQualificationError(
            "INVALID_PROFILE", f"{field} must be finite", field=field
        )
    return output


def _expect_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LipsyncQualificationError(
            "INVALID_PROFILE", f"{field} must be an integer", field=field
        )
    return value


def _expect_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LipsyncQualificationError(
            "INVALID_HASH", f"{field} must be a lowercase SHA-256 digest", field=field
        )
    return value


def _expect_timestamp(value: Any, field: str) -> str:
    timestamp = _expect_string(value, field, maximum=64)
    if not timestamp.endswith("Z"):
        raise LipsyncQualificationError(
            "INVALID_PROVENANCE", f"{field} must be UTC and end in Z", field=field
        )
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise LipsyncQualificationError(
            "INVALID_PROVENANCE", f"{field} is not an ISO-8601 timestamp", field=field
        ) from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise LipsyncQualificationError(
            "INVALID_PROVENANCE", f"{field} must use UTC", field=field
        )
    return timestamp


def _parse_artifact_reference(value: Any, field: str) -> ArtifactReference:
    source = _expect_object(value, field, ("artifact_id", "sha256"))
    artifact_id = _expect_string(source["artifact_id"], f"{field}.artifact_id", maximum=128)
    if _ARTIFACT_ID.fullmatch(artifact_id) is None:
        raise LipsyncQualificationError(
            "INVALID_PROVENANCE",
            f"{field}.artifact_id contains unsupported characters",
            field=f"{field}.artifact_id",
        )
    return ArtifactReference(
        artifact_id=artifact_id,
        sha256=_expect_sha(source["sha256"], f"{field}.sha256"),
    )


def _parse_binding(value: Any) -> QualificationBinding:
    field = "binding"
    source = _expect_object(
        value,
        field,
        (
            "source_audio_sha256",
            "character_manifest_sha256",
            "identity_artifact_sha256",
            "identity_array_sha256",
            "rig_sha256",
        ),
    )
    return QualificationBinding(
        **{name: _expect_sha(source[name], f"{field}.{name}") for name in source}
    )


def _parse_timebase(value: Any) -> QualificationTimebase:
    field = "timebase"
    source = _expect_object(
        value,
        field,
        (
            "units",
            "fps_numerator",
            "fps_denominator",
            "frame_count",
            "timestamp_origin_seconds",
        ),
    )
    units = _expect_string(source["units"], f"{field}.units")
    numerator = _expect_integer(source["fps_numerator"], f"{field}.fps_numerator")
    denominator = _expect_integer(source["fps_denominator"], f"{field}.fps_denominator")
    frame_count = _expect_integer(source["frame_count"], f"{field}.frame_count")
    origin = _expect_number(source["timestamp_origin_seconds"], f"{field}.timestamp_origin_seconds")
    if units != "seconds" or numerator <= 0 or denominator <= 0 or frame_count < 2 or origin != 0.0:
        raise LipsyncQualificationError(
            "INVALID_TIMEBASE",
            "Timebase requires seconds, positive rational FPS, at least two frames, and zero origin",
            field=field,
        )
    fps = numerator / denominator
    if not 12.0 <= fps <= 60.0:
        raise LipsyncQualificationError(
            "INVALID_TIMEBASE", "Qualification FPS must be in [12,60]", field=field
        )
    return QualificationTimebase(units, numerator, denominator, frame_count, origin)


def _parse_profile_provenance(value: Any) -> ProfileProvenance:
    field = "profile_provenance"
    source = _expect_object(value, field, ("curator_id", "created_at", "protocol"))
    protocol = _expect_string(source["protocol"], f"{field}.protocol")
    if protocol != "independent_lipsync_qualification_v1":
        raise LipsyncQualificationError(
            "INVALID_PROVENANCE", "Unsupported qualification protocol", field=f"{field}.protocol"
        )
    return ProfileProvenance(
        curator_id=_expect_string(source["curator_id"], f"{field}.curator_id"),
        created_at=_expect_timestamp(source["created_at"], f"{field}.created_at"),
        protocol=protocol,
    )


def _parse_annotation_provenance(value: Any) -> AnnotationProvenance:
    field = "annotation_provenance"
    source = _expect_object(
        value,
        field,
        (
            "annotator_id",
            "annotator_organization",
            "created_at",
            "method",
            "independent_from_animation_system",
            "viewed_system_cues",
            "viewed_generated_animation",
            "used_system_output_as_timing_source",
            "evidence_artifact",
        ),
    )
    method = _expect_string(source["method"], f"{field}.method")
    independent = _expect_boolean(
        source["independent_from_animation_system"],
        f"{field}.independent_from_animation_system",
    )
    viewed_cues = _expect_boolean(source["viewed_system_cues"], f"{field}.viewed_system_cues")
    viewed_animation = _expect_boolean(
        source["viewed_generated_animation"], f"{field}.viewed_generated_animation"
    )
    used_output = _expect_boolean(
        source["used_system_output_as_timing_source"],
        f"{field}.used_system_output_as_timing_source",
    )
    if method not in _ALLOWED_ANNOTATION_METHODS:
        raise LipsyncQualificationError(
            "INDEPENDENCE_UNPROVEN", "Annotations must use an independent manual method", field=field
        )
    if not independent or viewed_cues or viewed_animation or used_output:
        raise LipsyncQualificationError(
            "INDEPENDENCE_UNPROVEN",
            "Annotations are self-scored or independence is not established",
            field=field,
        )
    return AnnotationProvenance(
        annotator_id=_expect_string(source["annotator_id"], f"{field}.annotator_id"),
        annotator_organization=_expect_string(
            source["annotator_organization"], f"{field}.annotator_organization"
        ),
        created_at=_expect_timestamp(source["created_at"], f"{field}.created_at"),
        method=method,
        independent_from_animation_system=independent,
        viewed_system_cues=viewed_cues,
        viewed_generated_animation=viewed_animation,
        used_system_output_as_timing_source=used_output,
        evidence_artifact=_parse_artifact_reference(
            source["evidence_artifact"], f"{field}.evidence_artifact"
        ),
    )


def _parse_events(value: Any, timebase: QualificationTimebase) -> tuple[QualificationEvent, ...]:
    items = _expect_sequence(value, "annotations")
    events: list[QualificationEvent] = []
    event_ids: set[str] = set()
    last_apex = -1.0
    for index, item in enumerate(items):
        field = f"annotations.{index}"
        source = _expect_object(
            item,
            field,
            ("event_id", "label", "start_seconds", "apex_seconds", "release_seconds"),
        )
        event_id = _expect_string(source["event_id"], f"{field}.event_id", maximum=128)
        label = _expect_string(source["label"], f"{field}.label", maximum=64)
        start = _expect_number(source["start_seconds"], f"{field}.start_seconds")
        apex = _expect_number(source["apex_seconds"], f"{field}.apex_seconds")
        release = _expect_number(source["release_seconds"], f"{field}.release_seconds")
        if event_id in event_ids:
            raise LipsyncQualificationError(
                "INVALID_ANNOTATION", f"Duplicate event id {event_id!r}", field=f"{field}.event_id"
            )
        event_ids.add(event_id)
        frame = int(round(apex * timebase.fps))
        final_timestamp = (timebase.frame_count - 1) / timebase.fps
        if (
            not (0.0 <= start <= apex <= release <= final_timestamp)
            or frame < 0
            or frame >= timebase.frame_count
        ):
            raise LipsyncQualificationError(
                "INVALID_ANNOTATION",
                "Event times must be ordered and the apex must fall on the declared track",
                field=field,
            )
        if apex <= last_apex:
            raise LipsyncQualificationError(
                "INVALID_ANNOTATION", "Events must have strictly increasing apex times", field=field
            )
        last_apex = apex
        events.append(QualificationEvent(event_id, label, start, apex, release))
    return tuple(events)


def _parse_prototypes(value: Any) -> Mapping[str, TargetPrototype]:
    items = _expect_sequence(value, "target_prototypes")
    output: dict[str, TargetPrototype] = {}
    for index, item in enumerate(items):
        field = f"target_prototypes.{index}"
        source = _expect_object(
            item, field, ("label", "landmarks", "landmarks_sha256", "provenance")
        )
        label = _expect_string(source["label"], f"{field}.label", maximum=64)
        if label in output:
            raise LipsyncQualificationError(
                "INVALID_PROTOTYPE", f"Duplicate prototype label {label!r}", field=field
            )
        try:
            landmarks = np.asarray(source["landmarks"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise LipsyncQualificationError(
                "INVALID_PROTOTYPE", "Prototype landmarks must be numeric", field=f"{field}.landmarks"
            ) from exc
        actual_landmarks_hash = landmarks_sha256(landmarks)
        declared_landmarks_hash = _expect_sha(
            source["landmarks_sha256"], f"{field}.landmarks_sha256"
        )
        if actual_landmarks_hash != declared_landmarks_hash:
            raise LipsyncQualificationError(
                "PROTOTYPE_HASH_MISMATCH",
                f"Prototype {label!r} landmark hash does not match",
                field=f"{field}.landmarks_sha256",
            )
        provenance_field = f"{field}.provenance"
        provenance_source = _expect_object(
            source["provenance"],
            provenance_field,
            (
                "artist_id",
                "artist_organization",
                "created_at",
                "approved_at",
                "authoring_tool",
                "coordinate_space",
                "artist_approved",
                "source_artifact",
                "approval_artifact",
            ),
        )
        approved = _expect_boolean(
            provenance_source["artist_approved"], f"{provenance_field}.artist_approved"
        )
        coordinate_space = _expect_string(
            provenance_source["coordinate_space"], f"{provenance_field}.coordinate_space"
        )
        if not approved or coordinate_space != "gnm_head_sparse_68_3d":
            raise LipsyncQualificationError(
                "PROTOTYPE_NOT_APPROVED",
                "Every target must be artist-approved in GNM sparse-68 3D space",
                field=provenance_field,
            )
        landmarks.setflags(write=False)
        provenance = PrototypeProvenance(
            artist_id=_expect_string(
                provenance_source["artist_id"], f"{provenance_field}.artist_id"
            ),
            artist_organization=_expect_string(
                provenance_source["artist_organization"],
                f"{provenance_field}.artist_organization",
            ),
            created_at=_expect_timestamp(
                provenance_source["created_at"], f"{provenance_field}.created_at"
            ),
            approved_at=_expect_timestamp(
                provenance_source["approved_at"], f"{provenance_field}.approved_at"
            ),
            authoring_tool=_expect_string(
                provenance_source["authoring_tool"], f"{provenance_field}.authoring_tool"
            ),
            coordinate_space=coordinate_space,
            artist_approved=approved,
            source_artifact=_parse_artifact_reference(
                provenance_source["source_artifact"], f"{provenance_field}.source_artifact"
            ),
            approval_artifact=_parse_artifact_reference(
                provenance_source["approval_artifact"],
                f"{provenance_field}.approval_artifact",
            ),
        )
        output[label] = TargetPrototype(
            label=label,
            landmarks=landmarks,
            landmarks_sha256=declared_landmarks_hash,
            provenance=provenance,
        )
    return output


def _parse_evaluator(value: Any) -> QualificationEvaluator:
    field = "evaluator"
    source = _expect_object(
        value,
        field,
        (
            "quality_thresholds",
            "stationary_step_interocular",
            "neutral_tolerance_interocular",
            "silence_guard_frames",
            "timing_search_frames",
        ),
    )
    threshold_source = _expect_object(
        source["quality_thresholds"],
        f"{field}.quality_thresholds",
        _QUALITY_THRESHOLD_FIELDS,
    )
    float_fields = set(_QUALITY_THRESHOLD_FIELDS) - {
        "neutral_return_frames",
        "minimum_independent_events",
    }
    threshold_values: dict[str, float | int] = {}
    for name in _QUALITY_THRESHOLD_FIELDS:
        threshold_values[name] = (
            _expect_number(threshold_source[name], f"{field}.quality_thresholds.{name}")
            if name in float_fields
            else _expect_integer(threshold_source[name], f"{field}.quality_thresholds.{name}")
        )
    thresholds = QualityThresholds(**threshold_values)
    production_floor = QualityThresholds()
    bounded_unit = (
        thresholds.speech_active_stationary_fraction,
        thresholds.false_silence_motion_ratio_p95,
        thresholds.target_contrast_median,
        thresholds.target_contrast_p10,
    )
    if (
        thresholds.mouth_step_max_interocular <= 0.0
        or thresholds.neutral_return_frames < 0
        or thresholds.timing_error_median_frames < 0.0
        or thresholds.timing_error_p95_frames < thresholds.timing_error_median_frames
        or thresholds.minimum_independent_events < 3
        or any(not 0.0 <= item <= 1.0 for item in bounded_unit)
        or thresholds.target_contrast_p10 > thresholds.target_contrast_median
        or thresholds.mouth_step_max_interocular
        > production_floor.mouth_step_max_interocular
        or thresholds.speech_active_stationary_fraction
        > production_floor.speech_active_stationary_fraction
        or thresholds.neutral_return_frames > production_floor.neutral_return_frames
        or thresholds.false_silence_motion_ratio_p95
        > production_floor.false_silence_motion_ratio_p95
        or thresholds.target_contrast_median < production_floor.target_contrast_median
        or thresholds.target_contrast_p10 < production_floor.target_contrast_p10
        or thresholds.timing_error_median_frames
        > production_floor.timing_error_median_frames
        or thresholds.timing_error_p95_frames > production_floor.timing_error_p95_frames
        or thresholds.minimum_independent_events
        < production_floor.minimum_independent_events
    ):
        raise LipsyncQualificationError(
            "INVALID_THRESHOLDS", "Quality thresholds are inconsistent", field=field
        )
    stationary = _expect_number(
        source["stationary_step_interocular"], f"{field}.stationary_step_interocular"
    )
    neutral = _expect_number(
        source["neutral_tolerance_interocular"], f"{field}.neutral_tolerance_interocular"
    )
    silence_guard = _expect_integer(source["silence_guard_frames"], f"{field}.silence_guard_frames")
    timing_search = _expect_integer(source["timing_search_frames"], f"{field}.timing_search_frames")
    if (
        stationary != 5e-4
        or neutral != 0.015
        or silence_guard != 2
        or timing_search != 6
    ):
        raise LipsyncQualificationError(
            "INVALID_THRESHOLDS",
            "Schema v1 fixes evaluator tolerances/windows to the production defaults",
            field=field,
        )
    return QualificationEvaluator(thresholds, stationary, neutral, silence_guard, timing_search)


def parse_qualification_profile(
    source: str | Path | bytes | bytearray | Mapping[str, Any],
) -> LipsyncQualificationProfile:
    """Parse and fully validate one sealed version-1 qualification profile."""

    if isinstance(source, Mapping):
        document = deepcopy(dict(source))
    else:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise LipsyncQualificationError(
                    "PROFILE_MISSING", "Qualification profile does not exist"
                )
            if path.stat().st_size > _MAX_PROFILE_BYTES:
                raise LipsyncQualificationError(
                    "PROFILE_TOO_LARGE", "Qualification profile exceeds the size limit"
                )
            payload = path.read_bytes()
        else:
            payload = bytes(source)
            if len(payload) > _MAX_PROFILE_BYTES:
                raise LipsyncQualificationError(
                    "PROFILE_TOO_LARGE", "Qualification profile exceeds the size limit"
                )
        try:
            document = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_pairs_without_duplicates,
                parse_constant=_reject_constant,
            )
        except LipsyncQualificationError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LipsyncQualificationError(
                "INVALID_JSON", "Qualification profile is not strict UTF-8 JSON"
            ) from exc

    root = _expect_object(
        document,
        "profile",
        (
            "schema_version",
            "binding",
            "timebase",
            "profile_provenance",
            "annotation_provenance",
            "annotations",
            "target_prototypes",
            "evaluator",
            "profile_sha256",
        ),
    )
    schema = _expect_string(root["schema_version"], "schema_version")
    if schema != PROFILE_SCHEMA:
        raise LipsyncQualificationError(
            "UNSUPPORTED_SCHEMA", f"Unsupported qualification schema {schema!r}"
        )
    declared_profile_hash = _expect_sha(root["profile_sha256"], "profile_sha256")
    actual_profile_hash = profile_payload_sha256(root)
    if declared_profile_hash != actual_profile_hash:
        raise LipsyncQualificationError(
            "PROFILE_HASH_MISMATCH", "Qualification profile payload hash does not match"
        )

    timebase = _parse_timebase(root["timebase"])
    evaluator = _parse_evaluator(root["evaluator"])
    events = _parse_events(root["annotations"], timebase)
    prototypes = _parse_prototypes(root["target_prototypes"])
    labels = {event.label for event in events}
    if len(events) < evaluator.quality_thresholds.minimum_independent_events:
        raise LipsyncQualificationError(
            "INSUFFICIENT_EVIDENCE",
            "Profile contains fewer events than its production threshold",
            field="annotations",
        )
    if not labels or labels != set(prototypes):
        raise LipsyncQualificationError(
            "INCOMPLETE_PROTOTYPES",
            "Annotation labels and artist-approved prototype labels must match exactly",
            field="target_prototypes",
        )

    profile = LipsyncQualificationProfile(
        schema_version=schema,
        binding=_parse_binding(root["binding"]),
        timebase=timebase,
        profile_provenance=_parse_profile_provenance(root["profile_provenance"]),
        annotation_provenance=_parse_annotation_provenance(root["annotation_provenance"]),
        annotations=events,
        target_prototypes=prototypes,
        evaluator=evaluator,
        profile_sha256=declared_profile_hash,
    )
    profile.required_evidence()
    return profile


def _verify_file_binding(path: str | Path, expected: str, field: str) -> str:
    artifact = Path(path)
    if not artifact.is_file():
        raise LipsyncQualificationError(
            "BINDING_MISSING", f"Bound artifact {field} is missing", field=field
        )
    actual = sha256(artifact)
    if actual != expected:
        raise LipsyncQualificationError(
            "BINDING_MISMATCH", f"Bound artifact {field} does not match", field=field
        )
    return actual


def _verify_evidence_artifacts(
    profile: LipsyncQualificationProfile,
    artifacts: Mapping[str, str | Path],
) -> Mapping[str, str]:
    required = profile.required_evidence()
    if set(artifacts) != set(required):
        missing = sorted(set(required) - set(artifacts))
        extra = sorted(set(artifacts) - set(required))
        raise LipsyncQualificationError(
            "EVIDENCE_MISSING",
            f"Provenance artifact set differs (missing={missing}, extra={extra})",
            field="provenance_artifacts",
        )
    verified: dict[str, str] = {}
    for artifact_id, expected in sorted(required.items()):
        path = Path(artifacts[artifact_id])
        if not path.is_file():
            raise LipsyncQualificationError(
                "EVIDENCE_MISSING",
                f"Provenance artifact {artifact_id!r} is missing",
                field=f"provenance_artifacts.{artifact_id}",
            )
        actual = sha256(path)
        if actual != expected:
            raise LipsyncQualificationError(
                "EVIDENCE_MISMATCH",
                f"Provenance artifact {artifact_id!r} does not match",
                field=f"provenance_artifacts.{artifact_id}",
            )
        verified[artifact_id] = actual
    return verified


def _load_controls(
    controls_path: str | Path,
    profile: LipsyncQualificationProfile,
    rig: ControlRig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(controls_path)
    if not path.is_file():
        raise LipsyncQualificationError("CONTROLS_MISSING", "Controls artifact is missing")
    frame_count = profile.timebase.frame_count
    expression_dim = rig.adapter.expression_dim
    required_members = {
        "expression.npy",
        "timestamps.npy",
        "fps.npy",
        "speech_activity.npy",
    }
    maximum_uncompressed_bytes = (
        8 * (frame_count * (expression_dim + 2) + 1) + 1024 * 1024
    )
    if path.stat().st_size > maximum_uncompressed_bytes:
        raise LipsyncQualificationError(
            "INVALID_CONTROLS", "Controls archive exceeds its declared track bound"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or set(names) != required_members:
                raise LipsyncQualificationError(
                    "INVALID_CONTROLS",
                    "Controls archive members must match the numeric schema exactly",
                )
            if sum(member.file_size for member in members) > maximum_uncompressed_bytes:
                raise LipsyncQualificationError(
                    "INVALID_CONTROLS", "Controls archive expands beyond its declared track bound"
                )
        with np.load(path, allow_pickle=False) as archive:
            required = {"expression", "timestamps", "fps", "speech_activity"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise LipsyncQualificationError(
                    "INVALID_CONTROLS", f"Controls artifact is missing arrays: {missing}"
                )
            expression = np.asarray(archive["expression"])
            timestamps = np.asarray(archive["timestamps"])
            fps_value = np.asarray(archive["fps"])
            activity = np.asarray(archive["speech_activity"])
    except LipsyncQualificationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        raise LipsyncQualificationError(
            "INVALID_CONTROLS", "Controls artifact is not a safe numeric NPZ"
        ) from exc

    if (
        expression.shape != (frame_count, expression_dim)
        or expression.dtype.kind not in "fiu"
        or expression.dtype.itemsize > 8
        or not np.isfinite(expression).all()
        or np.max(np.abs(expression), initial=0.0) > 3.000001
    ):
        raise LipsyncQualificationError(
            "INVALID_CONTROLS",
            f"expression must be finite [{frame_count},{expression_dim}] in [-3,3]",
            field="controls.expression",
        )
    if (
        fps_value.shape != ()
        or fps_value.dtype.kind not in "fiu"
        or fps_value.dtype.itemsize > 8
        or not np.isfinite(fps_value)
    ):
        raise LipsyncQualificationError(
            "INVALID_CONTROLS", "fps must be one finite numeric scalar", field="controls.fps"
        )
    if abs(float(fps_value) - profile.timebase.fps) > 1e-9:
        raise LipsyncQualificationError(
            "TIMEBASE_MISMATCH", "Controls FPS differs from qualification profile", field="controls.fps"
        )
    expected_timestamps = (
        profile.timebase.timestamp_origin_seconds
        + np.arange(frame_count, dtype=np.float64) / profile.timebase.fps
    )
    if (
        timestamps.shape != (frame_count,)
        or timestamps.dtype.kind not in "fiu"
        or timestamps.dtype.itemsize > 8
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps.astype(np.float64)) <= 0.0)
        or not np.allclose(
            timestamps.astype(np.float64), expected_timestamps, rtol=0.0, atol=2e-6
        )
    ):
        raise LipsyncQualificationError(
            "TIMEBASE_MISMATCH",
            "Control timestamps do not match the declared exact output clock",
            field="controls.timestamps",
        )
    if (
        activity.shape != (frame_count,)
        or activity.dtype.kind not in "fiu"
        or activity.dtype.itemsize > 8
        or not np.isfinite(activity).all()
        or np.any(activity < 0.0)
        or np.any(activity > 1.0)
    ):
        raise LipsyncQualificationError(
            "INVALID_CONTROLS",
            "speech_activity must provide one finite [0,1] value per frame",
            field="controls.speech_activity",
        )
    return (
        expression.astype(np.float32, copy=False),
        timestamps.astype(np.float64, copy=False),
        activity.astype(np.float32, copy=False),
    )


def _report_payload(
    *,
    profile: LipsyncQualificationProfile,
    controls_hash: str,
    evidence: Mapping[str, str],
    quality: LipsyncQualityReport,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "profile_sha256": profile.profile_sha256,
        "controls_sha256": controls_hash,
        **asdict(profile.binding),
        "evidence_sha256s": dict(sorted(evidence.items())),
        "binding_verified": True,
        "independent_annotations_verified": True,
        "artist_prototypes_verified": True,
        "quality": quality.as_dict(),
        "core_quality_gate_passed": quality.production_gate.passed,
        "sequence_timing_validated": False,
        "perceptual_validation_completed": False,
        "production_validated": False,
        "qualification_scope": "independent_apex_pose_and_motion_hygiene_only",
    }


def evaluate_controls_qualification(
    profile_source: str | Path | bytes | bytearray | Mapping[str, Any] | LipsyncQualificationProfile,
    *,
    controls_path: str | Path,
    source_audio_path: str | Path,
    character_manifest_path: str | Path,
    identity_artifact_path: str | Path,
    provenance_artifacts: Mapping[str, str | Path],
    rig: ControlRig,
) -> LipsyncQualificationReport:
    """Evaluate an existing controls track only after every binding verifies."""

    profile = (
        profile_source
        if isinstance(profile_source, LipsyncQualificationProfile)
        else parse_qualification_profile(profile_source)
    )
    binding = profile.binding
    source_hash = _verify_file_binding(
        source_audio_path, binding.source_audio_sha256, "source_audio"
    )
    character_hash = _verify_file_binding(
        character_manifest_path,
        binding.character_manifest_sha256,
        "character_manifest",
    )
    identity_artifact_hash = _verify_file_binding(
        identity_artifact_path,
        binding.identity_artifact_sha256,
        "identity_artifact",
    )
    actual_identity_hash = identity_array_sha256(rig.identity)
    if actual_identity_hash != binding.identity_array_sha256:
        raise LipsyncQualificationError(
            "BINDING_MISMATCH",
            "Runtime identity coefficients differ from the qualification profile",
            field="identity_array",
        )
    actual_rig_hash = runtime_rig_sha256(rig)
    if actual_rig_hash != binding.rig_sha256:
        raise LipsyncQualificationError(
            "BINDING_MISMATCH",
            "Runtime GNM/decoder rig differs from the qualification profile",
            field="rig",
        )
    evidence = _verify_evidence_artifacts(profile, provenance_artifacts)
    expression, _timestamps, speech_activity = _load_controls(controls_path, profile, rig)

    landmarks = np.stack([rig.compact_landmarks(frame) for frame in expression])
    neutral = rig.compact_landmarks(
        np.zeros(rig.adapter.expression_dim, dtype=np.float32)
    )
    quality = evaluate_lipsync_quality(
        landmarks,
        neutral,
        speech_activity,
        fps=profile.timebase.fps,
        annotations=tuple(
            TimingAnnotation(event.apex_seconds, event.label) for event in profile.annotations
        ),
        annotations_are_independent=True,
        target_prototypes={
            label: prototype.landmarks
            for label, prototype in profile.target_prototypes.items()
        },
        thresholds=profile.evaluator.quality_thresholds,
        stationary_step_interocular=profile.evaluator.stationary_step_interocular,
        neutral_tolerance_interocular=profile.evaluator.neutral_tolerance_interocular,
        silence_guard_frames=profile.evaluator.silence_guard_frames,
        timing_search_frames=profile.evaluator.timing_search_frames,
    )
    controls_hash = sha256(controls_path)
    payload = _report_payload(
        profile=profile,
        controls_hash=controls_hash,
        evidence=evidence,
        quality=quality,
    )
    report_hash = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return LipsyncQualificationReport(
        profile_sha256=profile.profile_sha256,
        controls_sha256=controls_hash,
        source_audio_sha256=source_hash,
        character_manifest_sha256=character_hash,
        identity_artifact_sha256=identity_artifact_hash,
        identity_array_sha256=actual_identity_hash,
        rig_sha256=actual_rig_hash,
        evidence_sha256s=evidence,
        quality=quality,
        report_sha256=report_hash,
    )
