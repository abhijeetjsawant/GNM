"""Fail-closed I0 identity-capture evidence contract.

This module validates provenance declarations and the geometry of two supplied
calibrated camera bundles.  It deliberately does *not* evaluate an independent
scan, recompute camera calibration from raw target observations, or approve a
character.  Consequently neither a synthetic nor a real profile can set
``asset_identity_validated`` or ``production_validated`` in schema version 1.

The distinction is intentional: a complete declaration is useful evidence,
but declarations and camera coverage are not a likeness measurement.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePath
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .camera_bundle import CalibratedCameraBundle


PROFILE_SCHEMA_VERSION = "autoanim.identity-capture-qualification/1.0"
REPORT_SCHEMA_VERSION = "autoanim.identity-capture-qualification-report/1.0"
THRESHOLD_VERSION = "autoanim.identity-i0-contract-thresholds/1.0"
CAMERA_YAW_METHOD = "camera_center_azimuth_about_declared_world_origin_xz_v1"
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MIN_FIT_VIEWS = 5
MIN_HELD_OUT_VIEWS = 2
MIN_CAMERA_CENTER_YAW_SPAN_DEGREES = 120.0

FIXTURE_CLASSES = frozenset(("synthetic", "real_consented_subject"))
REQUIRED_CONSENT_SCOPES = (
    "biometric_capture",
    "3d_likeness_reconstruction",
    "derived_asset_storage",
    "commercial_animation",
    "reviewer_access",
)
REQUIRED_REVIEW_SCOPES = (
    "capture_neutrality",
    "held_out_geometry",
    "scan_alignment",
    "identity_likeness",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class IdentityQualificationError(ValueError):
    """The I0 profile, report, or supplied camera evidence is invalid."""

    def __init__(self, code: str, message: str, *, field: str | None = None):
        super().__init__(message)
        self.code = code
        self.field = field


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    artifact_id: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {"artifact_id": self.artifact_id, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ViewSource:
    view_index: int
    image_name: str
    artifact: ArtifactReference

    def as_dict(self) -> dict[str, Any]:
        return {
            "view_index": self.view_index,
            "image_name": self.image_name,
            **self.artifact.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class CaptureSession:
    role: str
    session_id: str
    acquired_at: str
    camera_bundle: ArtifactReference
    view_sources: tuple[ViewSource, ...]
    declared_fit_view_indices: tuple[int, ...]
    declared_held_out_view_indices: tuple[int, ...]
    declared_camera_center_yaw_span_degrees: float
    yaw_method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "session_id": self.session_id,
            "acquired_at": self.acquired_at,
            "camera_bundle": self.camera_bundle.as_dict(),
            "view_sources": [value.as_dict() for value in self.view_sources],
            "declared_fit_view_indices": list(self.declared_fit_view_indices),
            "declared_held_out_view_indices": list(
                self.declared_held_out_view_indices
            ),
            "declared_camera_center_yaw_span_degrees": (
                self.declared_camera_center_yaw_span_degrees
            ),
            "yaw_method": self.yaw_method,
        }


@dataclass(frozen=True, slots=True)
class SubjectBinding:
    pseudonymous_subject_id: str
    session_ids: tuple[str, ...]
    scan_acquisition_id: str
    same_subject_attested: bool
    attester_id: str
    evidence: ArtifactReference

    def as_dict(self) -> dict[str, Any]:
        return {
            "pseudonymous_subject_id": self.pseudonymous_subject_id,
            "session_ids": list(self.session_ids),
            "scan_acquisition_id": self.scan_acquisition_id,
            "same_subject_attested": self.same_subject_attested,
            "attester_id": self.attester_id,
            "evidence": self.evidence.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ConsentDescriptor:
    pseudonymous_subject_id: str
    attester_id: str
    scopes: tuple[str, ...]
    valid_from: str
    expires_at: str
    revoked: bool
    evidence: ArtifactReference

    def as_dict(self) -> dict[str, Any]:
        return {
            "pseudonymous_subject_id": self.pseudonymous_subject_id,
            "attester_id": self.attester_id,
            "scopes": list(self.scopes),
            "valid_from": self.valid_from,
            "expires_at": self.expires_at,
            "revoked": self.revoked,
            "evidence": self.evidence.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class IndependentScanDescriptor:
    acquisition_id: str
    pseudonymous_subject_id: str
    acquired_at: str
    units: str
    scan_artifact: ArtifactReference
    provenance_evidence: ArtifactReference
    independent_from_reconstruction: bool
    used_evaluation_photos: bool
    used_candidate_mesh_as_geometry_source: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "acquisition_id": self.acquisition_id,
            "pseudonymous_subject_id": self.pseudonymous_subject_id,
            "acquired_at": self.acquired_at,
            "units": self.units,
            "scan_artifact": self.scan_artifact.as_dict(),
            "provenance_evidence": self.provenance_evidence.as_dict(),
            "independent_from_reconstruction": (
                self.independent_from_reconstruction
            ),
            "used_evaluation_photos": self.used_evaluation_photos,
            "used_candidate_mesh_as_geometry_source": (
                self.used_candidate_mesh_as_geometry_source
            ),
        }


@dataclass(frozen=True, slots=True)
class ReviewerApproval:
    reviewer_id: str
    organization: str
    reviewed_at: str
    scopes: tuple[str, ...]
    decision: str
    independent_from_capture_and_fit: bool
    evidence: ArtifactReference

    def as_dict(self) -> dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "organization": self.organization,
            "reviewed_at": self.reviewed_at,
            "scopes": list(self.scopes),
            "decision": self.decision,
            "independent_from_capture_and_fit": (
                self.independent_from_capture_and_fit
            ),
            "evidence": self.evidence.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class IdentityQualificationProfile:
    schema_version: str
    threshold_version: str
    declared_fixture_class: str
    created_at: str
    subject_binding: SubjectBinding
    consent: ConsentDescriptor
    sessions: tuple[CaptureSession, ...]
    independent_scan: IndependentScanDescriptor
    reviewers: tuple[ReviewerApproval, ...]
    profile_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "threshold_version": self.threshold_version,
            "declared_fixture_class": self.declared_fixture_class,
            "created_at": self.created_at,
            "subject_binding": self.subject_binding.as_dict(),
            "consent": self.consent.as_dict(),
            "sessions": [value.as_dict() for value in self.sessions],
            "independent_scan": self.independent_scan.as_dict(),
            "reviewers": [value.as_dict() for value in self.reviewers],
            "profile_sha256": self.profile_sha256,
        }

    @property
    def sessions_by_role(self) -> Mapping[str, CaptureSession]:
        return {session.role: session for session in self.sessions}


def _canonical_json(value: Any, *, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise IdentityQualificationError(
            "INVALID_DOCUMENT", f"{label} is not finite canonical JSON"
        ) from exc


def _payload_sha256(document: Mapping[str, Any], digest_field: str, label: str) -> str:
    payload = deepcopy(dict(document))
    payload.pop(digest_field, None)
    return hashlib.sha256(_canonical_json(payload, label=label)).hexdigest()


def profile_payload_sha256(document: Mapping[str, Any]) -> str:
    return _payload_sha256(document, "profile_sha256", "Identity profile")


def report_payload_sha256(document: Mapping[str, Any]) -> str:
    return _payload_sha256(document, "report_sha256", "Identity report")


def _duplicate_free_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IdentityQualificationError(
                "DUPLICATE_KEY", f"Duplicate JSON member {key!r}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise IdentityQualificationError(
        "NONFINITE_NUMBER", f"Non-finite JSON number {value!r} is forbidden"
    )


def _read_document(
    source: str | Path | bytes | bytearray | Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        document = deepcopy(dict(source))
        encoded = _canonical_json(document, label=label)
        if not encoded or len(encoded) > MAX_DOCUMENT_BYTES:
            raise IdentityQualificationError(
                "DOCUMENT_SIZE", f"{label} size is outside the accepted bounds"
            )
        return document
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise IdentityQualificationError(
                "DOCUMENT_MISSING", f"{label} does not exist"
            )
        size = path.stat().st_size
        if size <= 0 or size > MAX_DOCUMENT_BYTES:
            raise IdentityQualificationError(
                "DOCUMENT_SIZE", f"{label} size is outside the accepted bounds"
            )
        payload = path.read_bytes()
        if not payload or len(payload) > MAX_DOCUMENT_BYTES:
            raise IdentityQualificationError(
                "DOCUMENT_SIZE", f"{label} size is outside the accepted bounds"
            )
    else:
        payload = bytes(source)
        if not payload or len(payload) > MAX_DOCUMENT_BYTES:
            raise IdentityQualificationError(
                "DOCUMENT_SIZE", f"{label} size is outside the accepted bounds"
            )
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_free_pairs,
            parse_constant=_reject_constant,
        )
    except IdentityQualificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise IdentityQualificationError(
            "INVALID_JSON", f"{label} must be strict UTF-8 JSON"
        ) from exc
    if not isinstance(document, dict):
        raise IdentityQualificationError(
            "INVALID_DOCUMENT", f"{label} root must be an object"
        )
    return document


def _object(value: Any, field: str, keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IdentityQualificationError(
            "INVALID_FIELD", f"{field} must be an object", field=field
        )
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise IdentityQualificationError(
            "INVALID_FIELDS",
            f"{field} has missing or unknown fields",
            field=field,
        )
    return value


def _sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise IdentityQualificationError(
            "INVALID_FIELD", f"{field} must be an array", field=field
        )
    return value


def _string(
    value: Any, field: str, *, maximum: int = 300, identifier: bool = False
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise IdentityQualificationError(
            "INVALID_FIELD", f"{field} must be a non-empty trimmed string", field=field
        )
    if len(value) > maximum or (identifier and _IDENTIFIER.fullmatch(value) is None):
        raise IdentityQualificationError(
            "INVALID_FIELD", f"{field} has an invalid format or length", field=field
        )
    return value


def _identifier_value(value: Any, field: str) -> str:
    return _string(value, field, maximum=160, identifier=True)


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IdentityQualificationError(
            "INVALID_DIGEST", f"{field} must be a lowercase SHA-256 digest", field=field
        )
    return value


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise IdentityQualificationError(
            "INVALID_FIELD", f"{field} must be boolean", field=field
        )
    return value


def _integer(value: Any, field: str) -> int:
    if type(value) is not int:
        raise IdentityQualificationError(
            "INVALID_FIELD", f"{field} must be an integer", field=field
        )
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IdentityQualificationError(
            "INVALID_FIELD", f"{field} must be numeric", field=field
        )
    try:
        result = float(value)
    except OverflowError as exc:
        raise IdentityQualificationError(
            "NONFINITE_NUMBER", f"{field} must be finite", field=field
        ) from exc
    if not math.isfinite(result):
        raise IdentityQualificationError(
            "NONFINITE_NUMBER", f"{field} must be finite", field=field
        )
    return result


def _timestamp(value: Any, field: str) -> tuple[str, datetime]:
    text = _string(value, field, maximum=20)
    if _TIMESTAMP.fullmatch(text) is None:
        raise IdentityQualificationError(
            "INVALID_TIMESTAMP",
            f"{field} must be UTC with whole-second YYYY-MM-DDTHH:MM:SSZ format",
            field=field,
        )
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise IdentityQualificationError(
            "INVALID_TIMESTAMP", f"{field} is not a valid timestamp", field=field
        ) from exc
    return text, parsed


def _artifact(value: Any, field: str) -> ArtifactReference:
    source = _object(value, field, ("artifact_id", "sha256"))
    return ArtifactReference(
        _identifier_value(source["artifact_id"], f"{field}.artifact_id"),
        _sha(source["sha256"], f"{field}.sha256"),
    )


def _scopes(value: Any, field: str, expected: Sequence[str]) -> tuple[str, ...]:
    items = _sequence(value, field)
    if any(not isinstance(item, str) for item in items):
        raise IdentityQualificationError(
            "INVALID_SCOPE", f"{field} must contain strings", field=field
        )
    if tuple(items) != tuple(expected):
        raise IdentityQualificationError(
            "INVALID_SCOPE",
            f"{field} must exactly match the schema-v1 ordered scope set",
            field=field,
        )
    return tuple(items)


def _indices(value: Any, field: str) -> tuple[int, ...]:
    items = _sequence(value, field)
    output = tuple(_integer(item, f"{field}.{index}") for index, item in enumerate(items))
    if any(item < 0 for item in output) or len(set(output)) != len(output):
        raise IdentityQualificationError(
            "INVALID_VIEW_SET", f"{field} must contain unique nonnegative indices", field=field
        )
    if tuple(sorted(output)) != output:
        raise IdentityQualificationError(
            "INVALID_VIEW_SET", f"{field} must be sorted", field=field
        )
    return output


def _parse_session(value: Any, index: int) -> CaptureSession:
    field = f"sessions.{index}"
    source = _object(
        value,
        field,
        (
            "role",
            "session_id",
            "acquired_at",
            "camera_bundle",
            "view_sources",
            "declared_fit_view_indices",
            "declared_held_out_view_indices",
            "declared_camera_center_yaw_span_degrees",
            "yaw_method",
        ),
    )
    role = _string(source["role"], f"{field}.role", maximum=16)
    if role not in {"primary", "repeat"}:
        raise IdentityQualificationError(
            "INVALID_SESSION", f"{field}.role must be primary or repeat", field=field
        )
    acquired_at, _ = _timestamp(source["acquired_at"], f"{field}.acquired_at")
    view_items = _sequence(source["view_sources"], f"{field}.view_sources")
    views: list[ViewSource] = []
    for view_index, raw_view in enumerate(view_items):
        view_field = f"{field}.view_sources.{view_index}"
        item = _object(
            raw_view,
            view_field,
            ("view_index", "image_name", "artifact_id", "sha256"),
        )
        declared_index = _integer(item["view_index"], f"{view_field}.view_index")
        image_name = _string(item["image_name"], f"{view_field}.image_name", maximum=255)
        if declared_index != view_index or PurePath(image_name).name != image_name:
            raise IdentityQualificationError(
                "INVALID_VIEW_SOURCE",
                f"{view_field} must be contiguous and use a basename",
                field=view_field,
            )
        views.append(
            ViewSource(
                view_index,
                image_name,
                ArtifactReference(
                    _identifier_value(item["artifact_id"], f"{view_field}.artifact_id"),
                    _sha(item["sha256"], f"{view_field}.sha256"),
                ),
            )
        )
    fit_indices = _indices(
        source["declared_fit_view_indices"], f"{field}.declared_fit_view_indices"
    )
    held_indices = _indices(
        source["declared_held_out_view_indices"],
        f"{field}.declared_held_out_view_indices",
    )
    if len(fit_indices) < MIN_FIT_VIEWS or len(held_indices) < MIN_HELD_OUT_VIEWS:
        raise IdentityQualificationError(
            "INSUFFICIENT_VIEWS",
            f"{field} requires at least {MIN_FIT_VIEWS} fit and {MIN_HELD_OUT_VIEWS} held-out views",
            field=field,
        )
    if set(fit_indices) & set(held_indices):
        raise IdentityQualificationError(
            "VIEW_LEAKAGE", f"{field} fit and held-out views overlap", field=field
        )
    expected_indices = tuple(range(len(views)))
    if tuple(sorted((*fit_indices, *held_indices))) != expected_indices:
        raise IdentityQualificationError(
            "INVALID_VIEW_SET",
            f"{field} declarations must partition every view exactly once",
            field=field,
        )
    hashes = [item.artifact.sha256 for item in views]
    identifiers = [item.artifact.artifact_id for item in views]
    if len(set(hashes)) != len(hashes) or len(set(identifiers)) != len(identifiers):
        raise IdentityQualificationError(
            "DUPLICATE_SOURCE", f"{field} view sources must be distinct", field=field
        )
    declared_yaw = _number(
        source["declared_camera_center_yaw_span_degrees"],
        f"{field}.declared_camera_center_yaw_span_degrees",
    )
    if not MIN_CAMERA_CENTER_YAW_SPAN_DEGREES <= declared_yaw <= 360.0:
        raise IdentityQualificationError(
            "INSUFFICIENT_YAW",
            f"{field} declared yaw span must be in [{MIN_CAMERA_CENTER_YAW_SPAN_DEGREES},360]",
            field=field,
        )
    yaw_method = _string(source["yaw_method"], f"{field}.yaw_method", maximum=100)
    if yaw_method != CAMERA_YAW_METHOD:
        raise IdentityQualificationError(
            "INVALID_YAW", f"{field} uses an unsupported yaw method", field=field
        )
    return CaptureSession(
        role=role,
        session_id=_identifier_value(source["session_id"], f"{field}.session_id"),
        acquired_at=acquired_at,
        camera_bundle=_artifact(source["camera_bundle"], f"{field}.camera_bundle"),
        view_sources=tuple(views),
        declared_fit_view_indices=fit_indices,
        declared_held_out_view_indices=held_indices,
        declared_camera_center_yaw_span_degrees=declared_yaw,
        yaw_method=yaw_method,
    )


def _parse_subject(value: Any) -> SubjectBinding:
    field = "subject_binding"
    source = _object(
        value,
        field,
        (
            "pseudonymous_subject_id",
            "session_ids",
            "scan_acquisition_id",
            "same_subject_attested",
            "attester_id",
            "evidence",
        ),
    )
    session_ids = tuple(
        _identifier_value(item, f"{field}.session_ids.{index}")
        for index, item in enumerate(_sequence(source["session_ids"], f"{field}.session_ids"))
    )
    if len(session_ids) != 2 or len(set(session_ids)) != 2:
        raise IdentityQualificationError(
            "SUBJECT_UNBOUND", "Subject binding requires two distinct session IDs", field=field
        )
    if _boolean(source["same_subject_attested"], f"{field}.same_subject_attested") is not True:
        raise IdentityQualificationError(
            "SUBJECT_UNBOUND", "Same-subject attestation is required", field=field
        )
    return SubjectBinding(
        pseudonymous_subject_id=_identifier_value(
            source["pseudonymous_subject_id"], f"{field}.pseudonymous_subject_id"
        ),
        session_ids=session_ids,
        scan_acquisition_id=_identifier_value(
            source["scan_acquisition_id"], f"{field}.scan_acquisition_id"
        ),
        same_subject_attested=True,
        attester_id=_identifier_value(source["attester_id"], f"{field}.attester_id"),
        evidence=_artifact(source["evidence"], f"{field}.evidence"),
    )


def _parse_consent(value: Any, created: datetime) -> ConsentDescriptor:
    field = "consent"
    source = _object(
        value,
        field,
        (
            "pseudonymous_subject_id",
            "attester_id",
            "scopes",
            "valid_from",
            "expires_at",
            "revoked",
            "evidence",
        ),
    )
    valid_from, valid = _timestamp(source["valid_from"], f"{field}.valid_from")
    expires_at, expires = _timestamp(source["expires_at"], f"{field}.expires_at")
    revoked = _boolean(source["revoked"], f"{field}.revoked")
    if revoked or not valid <= created < expires:
        raise IdentityQualificationError(
            "CONSENT_INACTIVE",
            "Consent must be active and unrevoked at profile creation",
            field=field,
        )
    return ConsentDescriptor(
        pseudonymous_subject_id=_identifier_value(
            source["pseudonymous_subject_id"], f"{field}.pseudonymous_subject_id"
        ),
        attester_id=_identifier_value(source["attester_id"], f"{field}.attester_id"),
        scopes=_scopes(source["scopes"], f"{field}.scopes", REQUIRED_CONSENT_SCOPES),
        valid_from=valid_from,
        expires_at=expires_at,
        revoked=False,
        evidence=_artifact(source["evidence"], f"{field}.evidence"),
    )


def _parse_scan(value: Any) -> IndependentScanDescriptor:
    field = "independent_scan"
    source = _object(
        value,
        field,
        (
            "acquisition_id",
            "pseudonymous_subject_id",
            "acquired_at",
            "units",
            "scan_artifact",
            "provenance_evidence",
            "independent_from_reconstruction",
            "used_evaluation_photos",
            "used_candidate_mesh_as_geometry_source",
        ),
    )
    acquired_at, _ = _timestamp(source["acquired_at"], f"{field}.acquired_at")
    units = _string(source["units"], f"{field}.units", maximum=16)
    independent = _boolean(
        source["independent_from_reconstruction"],
        f"{field}.independent_from_reconstruction",
    )
    used_photos = _boolean(
        source["used_evaluation_photos"], f"{field}.used_evaluation_photos"
    )
    used_candidate = _boolean(
        source["used_candidate_mesh_as_geometry_source"],
        f"{field}.used_candidate_mesh_as_geometry_source",
    )
    if units != "meters" or not independent or used_photos or used_candidate:
        raise IdentityQualificationError(
            "SCAN_NOT_INDEPENDENT",
            "Scan must be metric and independent from photos and candidate geometry",
            field=field,
        )
    return IndependentScanDescriptor(
        acquisition_id=_identifier_value(
            source["acquisition_id"], f"{field}.acquisition_id"
        ),
        pseudonymous_subject_id=_identifier_value(
            source["pseudonymous_subject_id"], f"{field}.pseudonymous_subject_id"
        ),
        acquired_at=acquired_at,
        units=units,
        scan_artifact=_artifact(source["scan_artifact"], f"{field}.scan_artifact"),
        provenance_evidence=_artifact(
            source["provenance_evidence"], f"{field}.provenance_evidence"
        ),
        independent_from_reconstruction=True,
        used_evaluation_photos=False,
        used_candidate_mesh_as_geometry_source=False,
    )


def _parse_reviewer(value: Any, index: int) -> ReviewerApproval:
    field = f"reviewers.{index}"
    source = _object(
        value,
        field,
        (
            "reviewer_id",
            "organization",
            "reviewed_at",
            "scopes",
            "decision",
            "independent_from_capture_and_fit",
            "evidence",
        ),
    )
    reviewed_at, _ = _timestamp(source["reviewed_at"], f"{field}.reviewed_at")
    decision = _string(source["decision"], f"{field}.decision", maximum=16)
    independent = _boolean(
        source["independent_from_capture_and_fit"],
        f"{field}.independent_from_capture_and_fit",
    )
    if decision != "approved" or not independent:
        raise IdentityQualificationError(
            "REVIEW_NOT_APPROVED",
            "Every reviewer must independently approve all I0 scopes",
            field=field,
        )
    return ReviewerApproval(
        reviewer_id=_identifier_value(source["reviewer_id"], f"{field}.reviewer_id"),
        organization=_string(source["organization"], f"{field}.organization", maximum=160),
        reviewed_at=reviewed_at,
        scopes=_scopes(source["scopes"], f"{field}.scopes", REQUIRED_REVIEW_SCOPES),
        decision=decision,
        independent_from_capture_and_fit=True,
        evidence=_artifact(source["evidence"], f"{field}.evidence"),
    )


def load_identity_qualification_profile(
    source: str | Path | bytes | bytearray | Mapping[str, Any],
) -> IdentityQualificationProfile:
    """Load and completely validate one bounded I0.0 profile."""

    document = _read_document(source, label="Identity qualification profile")
    root = _object(
        document,
        "profile",
        (
            "schema_version",
            "threshold_version",
            "declared_fixture_class",
            "created_at",
            "subject_binding",
            "consent",
            "sessions",
            "independent_scan",
            "reviewers",
            "profile_sha256",
        ),
    )
    schema = _string(root["schema_version"], "schema_version")
    thresholds = _string(root["threshold_version"], "threshold_version")
    if schema != PROFILE_SCHEMA_VERSION or thresholds != THRESHOLD_VERSION:
        raise IdentityQualificationError(
            "UNSUPPORTED_SCHEMA", "Unsupported identity qualification schema or thresholds"
        )
    declared_fixture_class = _string(
        root["declared_fixture_class"], "declared_fixture_class", maximum=40
    )
    if declared_fixture_class not in FIXTURE_CLASSES:
        raise IdentityQualificationError(
            "INVALID_FIXTURE",
            "declared_fixture_class is unsupported",
            field="declared_fixture_class",
        )
    created_at, created = _timestamp(root["created_at"], "created_at")
    sessions = tuple(
        _parse_session(value, index)
        for index, value in enumerate(_sequence(root["sessions"], "sessions"))
    )
    if len(sessions) != 2 or tuple(session.role for session in sessions) != (
        "primary",
        "repeat",
    ):
        raise IdentityQualificationError(
            "INVALID_SESSIONS", "Exactly ordered primary and repeat sessions are required"
        )
    if len({session.session_id for session in sessions}) != 2:
        raise IdentityQualificationError(
            "INVALID_SESSIONS", "Primary and repeat session IDs must differ"
        )
    all_sources = [
        view.artifact
        for session in sessions
        for view in session.view_sources
    ]
    if len({item.sha256 for item in all_sources}) != len(all_sources):
        raise IdentityQualificationError(
            "DUPLICATE_SOURCE", "Primary and repeat source images must all be distinct"
        )
    if len({item.artifact_id for item in all_sources}) != len(all_sources):
        raise IdentityQualificationError(
            "DUPLICATE_SOURCE", "Primary and repeat source artifact IDs must all be distinct"
        )
    subject = _parse_subject(root["subject_binding"])
    consent = _parse_consent(root["consent"], created)
    scan = _parse_scan(root["independent_scan"])
    if subject.session_ids != tuple(session.session_id for session in sessions):
        raise IdentityQualificationError(
            "SUBJECT_UNBOUND", "Subject evidence does not bind the ordered session IDs"
        )
    subject_id = subject.pseudonymous_subject_id
    if (
        consent.pseudonymous_subject_id != subject_id
        or scan.pseudonymous_subject_id != subject_id
        or subject.scan_acquisition_id != scan.acquisition_id
    ):
        raise IdentityQualificationError(
            "SUBJECT_UNBOUND", "Subject, consent, sessions, and scan are not identically bound"
        )
    reviewers = tuple(
        _parse_reviewer(value, index)
        for index, value in enumerate(_sequence(root["reviewers"], "reviewers"))
    )
    if len(reviewers) < 2:
        raise IdentityQualificationError(
            "INSUFFICIENT_REVIEWS", "At least two independent approvals are required"
        )
    normalized_reviewers = [reviewer.reviewer_id.casefold() for reviewer in reviewers]
    if len(set(normalized_reviewers)) != len(normalized_reviewers):
        raise IdentityQualificationError(
            "DUPLICATE_REVIEWER", "Reviewer identities must be distinct"
        )
    if len({reviewer.evidence.sha256 for reviewer in reviewers}) != len(reviewers):
        raise IdentityQualificationError(
            "DUPLICATE_REVIEWER",
            "Independent reviewers must retain distinct approval records",
        )
    session_times = [
        _timestamp(session.acquired_at, f"sessions.{index}.acquired_at")[1]
        for index, session in enumerate(sessions)
    ]
    scan_time = _timestamp(scan.acquired_at, "independent_scan.acquired_at")[1]
    review_times = [
        _timestamp(reviewer.reviewed_at, f"reviewers.{index}.reviewed_at")[1]
        for index, reviewer in enumerate(reviewers)
    ]
    evidence_ready = max((*session_times, scan_time))
    if (
        any(value > created for value in (*session_times, scan_time))
        or any(value < evidence_ready or value > created for value in review_times)
    ):
        raise IdentityQualificationError(
            "INVALID_EVIDENCE_TIME",
            "Capture and scan must precede reviews, and all evidence must precede the profile",
        )
    consent_valid_from = _timestamp(consent.valid_from, "consent.valid_from")[1]
    consent_expires_at = _timestamp(consent.expires_at, "consent.expires_at")[1]
    consent_bound_events = (*session_times, scan_time, *review_times, created)
    if any(
        value < consent_valid_from or value >= consent_expires_at
        for value in consent_bound_events
    ):
        raise IdentityQualificationError(
            "CONSENT_INACTIVE",
            "Declared consent must cover capture, scan, review, and profile timestamps",
            field="consent",
        )
    evidence = [
        subject.evidence,
        consent.evidence,
        scan.scan_artifact,
        scan.provenance_evidence,
        *(reviewer.evidence for reviewer in reviewers),
    ]
    if len({item.artifact_id for item in evidence}) != len(evidence):
        raise IdentityQualificationError(
            "DUPLICATE_EVIDENCE", "Evidence artifact IDs must be unique"
        )
    declared_hash = _sha(root["profile_sha256"], "profile_sha256")
    actual_hash = profile_payload_sha256(root)
    if declared_hash != actual_hash:
        raise IdentityQualificationError(
            "PROFILE_HASH_MISMATCH", "Identity profile payload hash does not match"
        )
    return IdentityQualificationProfile(
        schema_version=schema,
        threshold_version=thresholds,
        declared_fixture_class=declared_fixture_class,
        created_at=created_at,
        subject_binding=subject,
        consent=consent,
        sessions=sessions,
        independent_scan=scan,
        reviewers=reviewers,
        profile_sha256=declared_hash,
    )


def build_identity_qualification_profile(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash and validate a profile payload that omits ``profile_sha256``."""

    if not isinstance(payload, Mapping):
        raise IdentityQualificationError("INVALID_DOCUMENT", "Profile payload must be an object")
    document = deepcopy(dict(payload))
    if "profile_sha256" in document:
        raise IdentityQualificationError(
            "INVALID_FIELDS", "Profile builder payload must omit profile_sha256"
        )
    document["profile_sha256"] = profile_payload_sha256(document)
    return load_identity_qualification_profile(document).as_dict()


def _minimal_circular_span_degrees(angles: Sequence[float]) -> float:
    values = np.mod(np.asarray(tuple(angles), dtype=np.float64), 2.0 * np.pi)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise IdentityQualificationError(
            "CAMERA_GEOMETRY_INVALID", "Camera azimuths are unavailable or non-finite"
        )
    ordered = np.sort(values)
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 2.0 * np.pi)))
    return float(np.degrees(2.0 * np.pi - np.max(gaps)))


def camera_center_yaw_span_degrees(bundle: CalibratedCameraBundle) -> float:
    """Return fit-camera azimuth coverage around the declared world origin.

    I0.0 does not yet recompute a physical subject origin.  Therefore the method
    is named explicitly and real profiles remain non-authorizing.  Camera-bundle
    v2 will replace this declared-origin boundary with raw calibration evidence.
    """

    angles: list[float] = []
    for index in bundle.fit_indices:
        transform = bundle.views[index].world_to_camera
        center = -transform[:3, :3].T @ transform[:3, 3]
        radius = float(np.linalg.norm(center[[0, 2]]))
        if not np.isfinite(center).all() or radius <= 1e-8:
            raise IdentityQualificationError(
                "CAMERA_GEOMETRY_INVALID",
                "Fit camera center is not separated from the declared world origin",
            )
        angles.append(math.atan2(float(center[0]), float(center[2])))
    return _minimal_circular_span_degrees(angles)


def _bundle_semantic_sha256(bundle: CalibratedCameraBundle) -> str:
    return hashlib.sha256(
        _canonical_json(bundle.as_dict(), label="Camera bundle semantics")
    ).hexdigest()


_FAILURE_REMEDIATION = {
    "CAMERA_BUNDLE_HASH_MISMATCH": "Use the exact camera-bundle artifact bound by the profile.",
    "VIEW_SOURCE_COUNT_MISMATCH": "Bind exactly one ordered, distinct source image to every bundle view.",
    "VIEW_SOURCE_BINDING_MISMATCH": "Make profile view names and indices exactly match the calibrated bundle.",
    "FIT_VIEW_DECLARATION_MISMATCH": "Make the profile fit-view indices exactly match the bound bundle.",
    "FIT_VIEW_COUNT_FAILED": "Retain at least five fit views in the bound bundle.",
    "HELD_OUT_VIEW_DECLARATION_MISMATCH": "Make the profile held-out indices exactly match the bound bundle.",
    "HELD_OUT_VIEW_COUNT_FAILED": "Retain at least two untouched held-out views in the bound bundle.",
    "CALIBRATION_METADATA_GATE_FAILED": "Recalibrate the capture and pass the declared v1 metadata gates.",
    "CAMERA_CENTER_YAW_SPAN_FAILED": "Capture wider fit-camera coverage around the subject; at least 120 degrees is required.",
    "DECLARED_CAMERA_YAW_MISMATCH": "Regenerate the profile from the exact bundle; declared and recomputed yaw must agree.",
    "RAW_CALIBRATION_NOT_RECOMPUTED": "Add camera-bundle v2 raw target observations and independently recompute calibration.",
    "SCAN_METRICS_NOT_RECOMPUTED": "Evaluate the retained independent metric scan with rigid, no-scale alignment.",
    "REPEAT_GEOMETRY_NOT_RECOMPUTED": "Recompute primary-versus-repeat metric neutral-surface stability.",
    "FIXTURE_CLASS_NOT_INDEPENDENTLY_RESOLVED": "Independently verify whether retained evidence comes from a real consented subject or a synthetic fixture.",
    "SYNTHETIC_FIXTURE": "Repeat the protocol with a consented real subject, independent scan, and genuine reviews.",
}
_SESSION_FAILURE_CODES = frozenset(
    (
        "CAMERA_BUNDLE_HASH_MISMATCH",
        "VIEW_SOURCE_COUNT_MISMATCH",
        "VIEW_SOURCE_BINDING_MISMATCH",
        "FIT_VIEW_DECLARATION_MISMATCH",
        "FIT_VIEW_COUNT_FAILED",
        "HELD_OUT_VIEW_DECLARATION_MISMATCH",
        "HELD_OUT_VIEW_COUNT_FAILED",
        "CALIBRATION_METADATA_GATE_FAILED",
        "CAMERA_CENTER_YAW_SPAN_FAILED",
        "DECLARED_CAMERA_YAW_MISMATCH",
    )
)
_DERIVED_SESSION_FAILURE_CODES = frozenset(
    (
        "FIT_VIEW_COUNT_FAILED",
        "HELD_OUT_VIEW_COUNT_FAILED",
        "CALIBRATION_METADATA_GATE_FAILED",
        "CAMERA_CENTER_YAW_SPAN_FAILED",
        "DECLARED_CAMERA_YAW_MISMATCH",
    )
)


def build_identity_qualification_report(
    profile_source: (
        str
        | Path
        | bytes
        | bytearray
        | Mapping[str, Any]
        | IdentityQualificationProfile
    ),
    *,
    camera_bundles: Mapping[str, CalibratedCameraBundle],
) -> dict[str, Any]:
    """Evaluate the I0.0 contract without making a likeness-quality claim."""

    profile = load_identity_qualification_profile(
        profile_source.as_dict()
        if isinstance(profile_source, IdentityQualificationProfile)
        else profile_source
    )
    if set(camera_bundles) != {"primary", "repeat"}:
        raise IdentityQualificationError(
            "CAMERA_BUNDLES_MISSING", "Exactly primary and repeat camera bundles are required"
        )
    failures: list[str] = []
    session_reports: list[dict[str, Any]] = []
    for session in profile.sessions:
        bundle = camera_bundles[session.role]
        local_failures: list[str] = []
        if bundle.source_sha256 != session.camera_bundle.sha256:
            local_failures.append("CAMERA_BUNDLE_HASH_MISMATCH")
        if len(session.view_sources) != len(bundle.views):
            local_failures.append("VIEW_SOURCE_COUNT_MISMATCH")
        elif any(
            source.view_index != view.index or source.image_name != view.image_name
            for source, view in zip(session.view_sources, bundle.views, strict=True)
        ):
            local_failures.append("VIEW_SOURCE_BINDING_MISMATCH")
        if session.declared_fit_view_indices != bundle.fit_indices:
            local_failures.append("FIT_VIEW_DECLARATION_MISMATCH")
        if len(bundle.fit_indices) < MIN_FIT_VIEWS:
            local_failures.append("FIT_VIEW_COUNT_FAILED")
        if (
            session.declared_held_out_view_indices != bundle.held_out_indices
        ):
            local_failures.append("HELD_OUT_VIEW_DECLARATION_MISMATCH")
        if len(bundle.held_out_indices) < MIN_HELD_OUT_VIEWS:
            local_failures.append("HELD_OUT_VIEW_COUNT_FAILED")
        if not bundle.declared_calibration_metadata_gate_passed:
            local_failures.append("CALIBRATION_METADATA_GATE_FAILED")
        try:
            recomputed_yaw = camera_center_yaw_span_degrees(bundle)
        except IdentityQualificationError:
            recomputed_yaw = None
            local_failures.append("CAMERA_CENTER_YAW_SPAN_FAILED")
        if (
            recomputed_yaw is None
            or recomputed_yaw + 1e-9 < MIN_CAMERA_CENTER_YAW_SPAN_DEGREES
        ):
            if "CAMERA_CENTER_YAW_SPAN_FAILED" not in local_failures:
                local_failures.append("CAMERA_CENTER_YAW_SPAN_FAILED")
        if (
            recomputed_yaw is None
            or abs(
                recomputed_yaw
                - session.declared_camera_center_yaw_span_degrees
            )
            > 1e-6
        ):
            local_failures.append("DECLARED_CAMERA_YAW_MISMATCH")
        qualified = not local_failures
        session_reports.append(
            {
                "role": session.role,
                "session_id": session.session_id,
                "camera_bundle_artifact_id": session.camera_bundle.artifact_id,
                "camera_bundle_source_sha256": bundle.source_sha256,
                "camera_bundle_semantic_sha256": _bundle_semantic_sha256(bundle),
                "fit_view_indices": list(bundle.fit_indices),
                "held_out_view_indices": list(bundle.held_out_indices),
                "fit_view_count": len(bundle.fit_indices),
                "held_out_view_count": len(bundle.held_out_indices),
                "declared_camera_center_yaw_span_degrees": (
                    session.declared_camera_center_yaw_span_degrees
                ),
                "recomputed_camera_center_yaw_span_degrees": recomputed_yaw,
                "yaw_method": CAMERA_YAW_METHOD,
                "declared_calibration_metadata_gate_passed": (
                    bundle.declared_calibration_metadata_gate_passed
                ),
                "contract_gate_passed": qualified,
                "failures": sorted(set(local_failures)),
            }
        )
        failures.extend(f"{session.role}:{value}" for value in local_failures)

    # These are mandatory schema-v1 blockers, not missing optional polish.
    future_blockers = [
        "RAW_CALIBRATION_NOT_RECOMPUTED",
        "SCAN_METRICS_NOT_RECOMPUTED",
        "REPEAT_GEOMETRY_NOT_RECOMPUTED",
    ]
    future_blockers.append("FIXTURE_CLASS_NOT_INDEPENDENTLY_RESOLVED")
    if profile.declared_fixture_class == "synthetic":
        future_blockers.append("SYNTHETIC_FIXTURE")
    contract_gate_passed = not failures
    all_failures = sorted((*failures, *future_blockers))
    remediation_codes = sorted(
        {
            value.split(":", 1)[-1]
            for value in all_failures
        }
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "threshold_version": THRESHOLD_VERSION,
        "profile_sha256": profile.profile_sha256,
        "declared_fixture_class": profile.declared_fixture_class,
        "fixture_class_resolved": False,
        "thresholds": {
            "minimum_fit_views_per_session": MIN_FIT_VIEWS,
            "minimum_held_out_views_per_session": MIN_HELD_OUT_VIEWS,
            "minimum_camera_center_yaw_span_degrees": (
                MIN_CAMERA_CENTER_YAW_SPAN_DEGREES
            ),
            "yaw_method": CAMERA_YAW_METHOD,
        },
        "bindings": {
            "pseudonymous_subject_id": profile.subject_binding.pseudonymous_subject_id,
            "session_ids": list(profile.subject_binding.session_ids),
            "scan_acquisition_id": profile.independent_scan.acquisition_id,
            "scan_sha256": profile.independent_scan.scan_artifact.sha256,
            "consent_evidence_sha256": profile.consent.evidence.sha256,
            "review_evidence_sha256s": [
                reviewer.evidence.sha256 for reviewer in profile.reviewers
            ],
        },
        "declaration_gates": {
            "same_subject_self_attested": True,
            "consent_declared_active_for_capture_scan_review_and_profile": True,
            "required_consent_scopes_declared": True,
            "independent_metric_scan_self_attested": True,
            "minimum_independent_reviewer_approvals_declared": True,
        },
        "sessions": session_reports,
        "contract_gate_passed": contract_gate_passed,
        "raw_calibration_recomputed": False,
        "scan_metrics_recomputed": False,
        "repeat_geometry_recomputed": False,
        "asset_identity_validated": False,
        "production_validated": False,
        "failures": all_failures,
        "remediation": [
            {"code": code, "action": _FAILURE_REMEDIATION[code]}
            for code in remediation_codes
        ],
        "qualification_scope": (
            "i0_declared_contract_and_camera_coverage_only_no_reality_resolution"
        ),
    }
    report["report_sha256"] = report_payload_sha256(report)
    return load_identity_qualification_report(report)


def load_identity_qualification_report(
    source: str | Path | bytes | bytearray | Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly load a deterministic I0.0 report and enforce non-authorization."""

    document = _read_document(source, label="Identity qualification report")
    root = _object(
        document,
        "report",
        (
            "schema_version",
            "threshold_version",
            "profile_sha256",
            "declared_fixture_class",
            "fixture_class_resolved",
            "thresholds",
            "bindings",
            "declaration_gates",
            "sessions",
            "contract_gate_passed",
            "raw_calibration_recomputed",
            "scan_metrics_recomputed",
            "repeat_geometry_recomputed",
            "asset_identity_validated",
            "production_validated",
            "failures",
            "remediation",
            "qualification_scope",
            "report_sha256",
        ),
    )
    if root["schema_version"] != REPORT_SCHEMA_VERSION or root["threshold_version"] != THRESHOLD_VERSION:
        raise IdentityQualificationError("UNSUPPORTED_SCHEMA", "Unsupported I0 report schema")
    _sha(root["profile_sha256"], "profile_sha256")
    _sha(root["report_sha256"], "report_sha256")
    expected_thresholds = {
        "minimum_fit_views_per_session": MIN_FIT_VIEWS,
        "minimum_held_out_views_per_session": MIN_HELD_OUT_VIEWS,
        "minimum_camera_center_yaw_span_degrees": MIN_CAMERA_CENTER_YAW_SPAN_DEGREES,
        "yaw_method": CAMERA_YAW_METHOD,
    }
    if root["thresholds"] != expected_thresholds:
        raise IdentityQualificationError(
            "INVALID_THRESHOLDS", "I0 report thresholds differ from schema-v1"
        )
    declared_fixture_class = _string(
        root["declared_fixture_class"], "declared_fixture_class", maximum=40
    )
    if declared_fixture_class not in FIXTURE_CLASSES:
        raise IdentityQualificationError("INVALID_FIXTURE", "I0 report fixture is invalid")
    for name in (
        "contract_gate_passed",
        "fixture_class_resolved",
        "raw_calibration_recomputed",
        "scan_metrics_recomputed",
        "repeat_geometry_recomputed",
        "asset_identity_validated",
        "production_validated",
    ):
        _boolean(root[name], name)
    if (
        root["fixture_class_resolved"]
        or root["raw_calibration_recomputed"]
        or root["scan_metrics_recomputed"]
        or root["repeat_geometry_recomputed"]
        or root["asset_identity_validated"]
        or root["production_validated"]
    ):
        raise IdentityQualificationError(
            "UNSUPPORTED_CLAIM", "I0.0 reports cannot authorize identity or production"
        )
    if root["qualification_scope"] != (
        "i0_declared_contract_and_camera_coverage_only_no_reality_resolution"
    ):
        raise IdentityQualificationError(
            "INVALID_SCOPE", "I0 report qualification scope is invalid"
        )
    bindings = _object(
        root["bindings"],
        "bindings",
        (
            "pseudonymous_subject_id",
            "session_ids",
            "scan_acquisition_id",
            "scan_sha256",
            "consent_evidence_sha256",
            "review_evidence_sha256s",
        ),
    )
    _identifier_value(
        bindings["pseudonymous_subject_id"], "bindings.pseudonymous_subject_id"
    )
    session_ids = tuple(
        _identifier_value(value, f"bindings.session_ids.{index}")
        for index, value in enumerate(
            _sequence(bindings["session_ids"], "bindings.session_ids")
        )
    )
    if len(session_ids) != 2 or len(set(session_ids)) != 2:
        raise IdentityQualificationError(
            "INVALID_REPORT", "Report bindings require two distinct session IDs"
        )
    _identifier_value(bindings["scan_acquisition_id"], "bindings.scan_acquisition_id")
    _sha(bindings["scan_sha256"], "bindings.scan_sha256")
    _sha(
        bindings["consent_evidence_sha256"],
        "bindings.consent_evidence_sha256",
    )
    review_hashes = tuple(
        _sha(value, f"bindings.review_evidence_sha256s.{index}")
        for index, value in enumerate(
            _sequence(
                bindings["review_evidence_sha256s"],
                "bindings.review_evidence_sha256s",
            )
        )
    )
    if len(review_hashes) < 2 or len(set(review_hashes)) != len(review_hashes):
        raise IdentityQualificationError(
            "INVALID_REPORT", "Report bindings require distinct reviewer evidence"
        )

    declaration_gates = _object(
        root["declaration_gates"],
        "declaration_gates",
        (
            "same_subject_self_attested",
            "consent_declared_active_for_capture_scan_review_and_profile",
            "required_consent_scopes_declared",
            "independent_metric_scan_self_attested",
            "minimum_independent_reviewer_approvals_declared",
        ),
    )
    if any(
        _boolean(value, f"declaration_gates.{name}") is not True
        for name, value in declaration_gates.items()
    ):
        raise IdentityQualificationError(
            "UNSUPPORTED_CLAIM", "An I0 declaration gate is not established"
        )

    session_values = _sequence(root["sessions"], "sessions")
    if len(session_values) != 2:
        raise IdentityQualificationError(
            "INVALID_REPORT", "I0 report must contain two session results"
        )
    parsed_session_failures: list[str] = []
    parsed_roles: list[str] = []
    for index, value in enumerate(session_values):
        field = f"sessions.{index}"
        session = _object(
            value,
            field,
            (
                "role",
                "session_id",
                "camera_bundle_artifact_id",
                "camera_bundle_source_sha256",
                "camera_bundle_semantic_sha256",
                "fit_view_indices",
                "held_out_view_indices",
                "fit_view_count",
                "held_out_view_count",
                "declared_camera_center_yaw_span_degrees",
                "recomputed_camera_center_yaw_span_degrees",
                "yaw_method",
                "declared_calibration_metadata_gate_passed",
                "contract_gate_passed",
                "failures",
            ),
        )
        role = _string(session["role"], f"{field}.role", maximum=16)
        if role not in {"primary", "repeat"}:
            raise IdentityQualificationError(
                "INVALID_REPORT", f"{field}.role is invalid"
            )
        parsed_roles.append(role)
        session_id = _identifier_value(session["session_id"], f"{field}.session_id")
        if session_id != session_ids[index]:
            raise IdentityQualificationError(
                "INVALID_REPORT", f"{field} does not match its bound session ID"
            )
        _identifier_value(
            session["camera_bundle_artifact_id"],
            f"{field}.camera_bundle_artifact_id",
        )
        _sha(
            session["camera_bundle_source_sha256"],
            f"{field}.camera_bundle_source_sha256",
        )
        _sha(
            session["camera_bundle_semantic_sha256"],
            f"{field}.camera_bundle_semantic_sha256",
        )
        fit_indices = _indices(
            session["fit_view_indices"], f"{field}.fit_view_indices"
        )
        held_indices = _indices(
            session["held_out_view_indices"], f"{field}.held_out_view_indices"
        )
        fit_count = _integer(session["fit_view_count"], f"{field}.fit_view_count")
        held_count = _integer(
            session["held_out_view_count"], f"{field}.held_out_view_count"
        )
        if (
            fit_count < 0
            or held_count < 0
            or fit_count != len(fit_indices)
            or held_count != len(held_indices)
            or set(fit_indices) & set(held_indices)
        ):
            raise IdentityQualificationError(
                "INVALID_REPORT",
                f"{field} view counts and disjoint index sets must agree",
            )
        declared_yaw = _number(
            session["declared_camera_center_yaw_span_degrees"],
            f"{field}.declared_camera_center_yaw_span_degrees",
        )
        recomputed_raw = session["recomputed_camera_center_yaw_span_degrees"]
        recomputed_yaw = (
            None
            if recomputed_raw is None
            else _number(
                recomputed_raw,
                f"{field}.recomputed_camera_center_yaw_span_degrees",
            )
        )
        if (
            not 0.0 <= declared_yaw <= 360.0
            or (recomputed_yaw is not None and not 0.0 <= recomputed_yaw <= 360.0)
            or session["yaw_method"] != CAMERA_YAW_METHOD
        ):
            raise IdentityQualificationError(
                "INVALID_REPORT", f"{field} yaw evidence is invalid"
            )
        calibration_gate = _boolean(
            session["declared_calibration_metadata_gate_passed"],
            f"{field}.declared_calibration_metadata_gate_passed",
        )
        session_gate = _boolean(
            session["contract_gate_passed"], f"{field}.contract_gate_passed"
        )
        local_failures = _sequence(session["failures"], f"{field}.failures")
        derived_failures: set[str] = set()
        if fit_count < MIN_FIT_VIEWS:
            derived_failures.add("FIT_VIEW_COUNT_FAILED")
        if held_count < MIN_HELD_OUT_VIEWS:
            derived_failures.add("HELD_OUT_VIEW_COUNT_FAILED")
        if not calibration_gate:
            derived_failures.add("CALIBRATION_METADATA_GATE_FAILED")
        if (
            recomputed_yaw is None
            or recomputed_yaw + 1e-9 < MIN_CAMERA_CENTER_YAW_SPAN_DEGREES
        ):
            derived_failures.add("CAMERA_CENTER_YAW_SPAN_FAILED")
        if (
            recomputed_yaw is None
            or abs(recomputed_yaw - declared_yaw) > 1e-6
        ):
            derived_failures.add("DECLARED_CAMERA_YAW_MISMATCH")
        reported_derived_failures = set(local_failures) & _DERIVED_SESSION_FAILURE_CODES
        if (
            any(value not in _SESSION_FAILURE_CODES for value in local_failures)
            or local_failures != sorted(set(local_failures))
            or reported_derived_failures != derived_failures
            or session_gate != (not local_failures)
        ):
            raise IdentityQualificationError(
                "INVALID_REPORT",
                f"{field} failures do not match its counts, yaw, calibration, and gate",
            )
        parsed_session_failures.extend(f"{role}:{code}" for code in local_failures)
    if tuple(parsed_roles) != ("primary", "repeat"):
        raise IdentityQualificationError(
            "INVALID_REPORT", "I0 session report order must be primary then repeat"
        )
    failures = root["failures"]
    if (
        not isinstance(failures, list)
        or any(not isinstance(value, str) for value in failures)
        or failures != sorted(set(failures))
    ):
        raise IdentityQualificationError(
            "INVALID_REPORT", "I0 report failures must be sorted and unique"
        )
    mandatory = {
        "FIXTURE_CLASS_NOT_INDEPENDENTLY_RESOLVED",
        "RAW_CALIBRATION_NOT_RECOMPUTED",
        "REPEAT_GEOMETRY_NOT_RECOMPUTED",
        "SCAN_METRICS_NOT_RECOMPUTED",
    }
    if declared_fixture_class == "synthetic":
        mandatory.add("SYNTHETIC_FIXTURE")
    expected_failures = sorted((*parsed_session_failures, *mandatory))
    if failures != expected_failures:
        raise IdentityQualificationError(
            "UNSUPPORTED_CLAIM",
            "I0 report failure set does not match its sessions and mandatory blockers",
        )
    if root["contract_gate_passed"] != (not parsed_session_failures):
        raise IdentityQualificationError(
            "INVALID_REPORT", "I0 report contract gate contradicts its sessions"
        )
    remediation = _sequence(root["remediation"], "remediation")
    expected_codes = sorted({value.split(":", 1)[-1] for value in failures})
    if len(remediation) != len(expected_codes):
        raise IdentityQualificationError(
            "INVALID_REPORT", "I0 remediation does not cover every failure"
        )
    for index, (value, expected_code) in enumerate(
        zip(remediation, expected_codes, strict=True)
    ):
        item = _object(value, f"remediation.{index}", ("code", "action"))
        if (
            item["code"] != expected_code
            or item["action"] != _FAILURE_REMEDIATION[expected_code]
        ):
            raise IdentityQualificationError(
                "INVALID_REPORT", "I0 remediation content is not deterministic"
            )
    actual_hash = report_payload_sha256(root)
    if root["report_sha256"] != actual_hash:
        raise IdentityQualificationError(
            "REPORT_HASH_MISMATCH", "Identity report payload hash does not match"
        )
    # Canonical round-trip drops no information and ensures nested NaN values
    # or unserializable objects cannot hide below fields used above.
    return json.loads(_canonical_json(root, label="Identity report").decode("utf-8"))


def verify_identity_qualification_report_profile(
    profile_source: (
        str
        | Path
        | bytes
        | bytearray
        | Mapping[str, Any]
        | IdentityQualificationProfile
    ),
    report_source: str | Path | bytes | bytearray | Mapping[str, Any],
) -> tuple[IdentityQualificationProfile, dict[str, Any]]:
    """Strictly bind every report declaration to its exact parsed profile.

    This verifier does not load or recompute the camera bundles.  It prevents a
    report from changing profile-bound session IDs, bundle references, view
    partitions, counts, or declared/reported yaw while retaining the profile
    hash.  Raw bundle recomputation remains a separate future evaluator.
    """

    profile = load_identity_qualification_profile(
        profile_source.as_dict()
        if isinstance(profile_source, IdentityQualificationProfile)
        else profile_source
    )
    report = load_identity_qualification_report(report_source)
    bindings = report["bindings"]
    expected_review_hashes = [
        reviewer.evidence.sha256 for reviewer in profile.reviewers
    ]
    if (
        report["profile_sha256"] != profile.profile_sha256
        or report["declared_fixture_class"]
        != profile.declared_fixture_class
        or bindings["pseudonymous_subject_id"]
        != profile.subject_binding.pseudonymous_subject_id
        or bindings["session_ids"]
        != list(profile.subject_binding.session_ids)
        or bindings["scan_acquisition_id"]
        != profile.independent_scan.acquisition_id
        or bindings["scan_sha256"]
        != profile.independent_scan.scan_artifact.sha256
        or bindings["consent_evidence_sha256"]
        != profile.consent.evidence.sha256
        or bindings["review_evidence_sha256s"] != expected_review_hashes
    ):
        raise IdentityQualificationError(
            "REPORT_PROFILE_BINDING_MISMATCH",
            "Identity qualification report is not bound to its exact profile",
            field="bindings",
        )

    report_sessions = report["sessions"]
    if len(report_sessions) != len(profile.sessions):
        raise IdentityQualificationError(
            "REPORT_PROFILE_BINDING_MISMATCH",
            "Identity qualification session counts do not match",
            field="sessions",
        )
    for index, (expected, observed) in enumerate(
        zip(profile.sessions, report_sessions, strict=True)
    ):
        expected_fit = list(expected.declared_fit_view_indices)
        expected_held = list(expected.declared_held_out_view_indices)
        recomputed_yaw = observed["recomputed_camera_center_yaw_span_degrees"]
        if (
            observed["role"] != expected.role
            or observed["session_id"] != expected.session_id
            or observed["camera_bundle_artifact_id"]
            != expected.camera_bundle.artifact_id
            or observed["camera_bundle_source_sha256"]
            != expected.camera_bundle.sha256
            or observed["fit_view_indices"] != expected_fit
            or observed["held_out_view_indices"] != expected_held
            or observed["fit_view_count"] != len(expected_fit)
            or observed["held_out_view_count"] != len(expected_held)
            or abs(
                observed["declared_camera_center_yaw_span_degrees"]
                - expected.declared_camera_center_yaw_span_degrees
            )
            > 1e-9
            or recomputed_yaw is None
            or abs(
                recomputed_yaw
                - expected.declared_camera_center_yaw_span_degrees
            )
            > 1e-6
            or observed["yaw_method"] != expected.yaw_method
        ):
            raise IdentityQualificationError(
                "REPORT_PROFILE_BINDING_MISMATCH",
                f"Report session {index} does not match its exact profile declaration",
                field=f"sessions.{index}",
            )
    return profile, report


__all__ = [
    "CAMERA_YAW_METHOD",
    "FIXTURE_CLASSES",
    "IdentityQualificationError",
    "IdentityQualificationProfile",
    "MAX_DOCUMENT_BYTES",
    "MIN_CAMERA_CENTER_YAW_SPAN_DEGREES",
    "MIN_FIT_VIEWS",
    "MIN_HELD_OUT_VIEWS",
    "PROFILE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "REQUIRED_CONSENT_SCOPES",
    "REQUIRED_REVIEW_SCOPES",
    "THRESHOLD_VERSION",
    "build_identity_qualification_profile",
    "build_identity_qualification_report",
    "camera_center_yaw_span_degrees",
    "load_identity_qualification_profile",
    "load_identity_qualification_report",
    "profile_payload_sha256",
    "report_payload_sha256",
    "verify_identity_qualification_report_profile",
]
