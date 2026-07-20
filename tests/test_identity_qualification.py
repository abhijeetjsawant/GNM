from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from autoanim_gnm.api import create_app
from autoanim_gnm.camera_bundle import (
    CalibratedCameraBundle,
    CalibratedCameraView,
)
from autoanim_gnm.cli import build_parser, main as cli_main
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.identity_qualification import (
    CAMERA_YAW_METHOD,
    MAX_DOCUMENT_BYTES,
    PROFILE_SCHEMA_VERSION,
    REQUIRED_CONSENT_SCOPES,
    REQUIRED_REVIEW_SCOPES,
    THRESHOLD_VERSION,
    IdentityQualificationError,
    build_identity_qualification_profile,
    build_identity_qualification_report,
    camera_center_yaw_span_degrees,
    load_identity_qualification_profile,
    load_identity_qualification_report,
    report_payload_sha256,
    verify_identity_qualification_report_profile,
)
from autoanim_gnm.service import ApplicationService


def _digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _camera_view(
    index: int,
    yaw_degrees: float,
    usage: str,
    *,
    prefix: str,
) -> CalibratedCameraView:
    angle = np.radians(yaw_degrees)
    center = np.asarray((np.sin(angle), 0.0, np.cos(angle)), dtype=np.float64)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, 3] = -center
    return CalibratedCameraView(
        index=index,
        image_name=f"{prefix}-{index}.png",
        role=f"view_{index}",
        usage=usage,
        image_size=(640, 640),
        intrinsics_matrix=np.asarray(
            ((800.0, 0.0, 319.5), (0.0, 800.0, 319.5), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        distortion=np.zeros(5, dtype=np.float64),
        world_to_camera=world_to_camera,
        visibility=np.ones(68, dtype=np.float64),
    )


def _bundle(
    role: str,
    *,
    fit_yaws: tuple[float, ...] = (-70.0, -35.0, 0.0, 35.0, 70.0),
    held_yaws: tuple[float, ...] = (-52.0, 52.0),
    source_hash: str | None = None,
    calibration_rms_px: float = 0.2,
) -> CalibratedCameraBundle:
    yaws = (*fit_yaws, *held_yaws)
    views = tuple(
        _camera_view(
            index,
            yaw,
            "fit" if index < len(fit_yaws) else "held_out",
            prefix=role,
        )
        for index, yaw in enumerate(yaws)
    )
    return CalibratedCameraBundle(
        calibration_rms_px=calibration_rms_px,
        pose_error_degrees=0.3,
        scale_error_fraction=0.002,
        views=views,
        source_sha256=source_hash or _digest(f"{role}-bundle"),
        meters_per_world_unit=1.0,
    )


def _session_payload(
    role: str, bundle: CalibratedCameraBundle, session_id: str
) -> dict:
    return {
        "role": role,
        "session_id": session_id,
        "acquired_at": "2026-07-01T10:00:00Z" if role == "primary" else "2026-07-02T10:00:00Z",
        "camera_bundle": {
            "artifact_id": f"{role}-camera-bundle",
            "sha256": bundle.source_sha256,
        },
        "view_sources": [
            {
                "view_index": view.index,
                "image_name": view.image_name,
                "artifact_id": f"{role}-source-{view.index}",
                "sha256": _digest(f"{role}-source-{view.index}"),
            }
            for view in bundle.views
        ],
        "declared_fit_view_indices": list(bundle.fit_indices),
        "declared_held_out_view_indices": list(bundle.held_out_indices),
        "declared_camera_center_yaw_span_degrees": camera_center_yaw_span_degrees(
            bundle
        ),
        "yaw_method": CAMERA_YAW_METHOD,
    }


def _profile_payload(
    primary: CalibratedCameraBundle,
    repeat: CalibratedCameraBundle,
    *,
    declared_fixture_class: str = "synthetic",
) -> dict:
    primary_id = "session-primary-01"
    repeat_id = "session-repeat-01"
    scan_id = "scan-01"
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "threshold_version": THRESHOLD_VERSION,
        "declared_fixture_class": declared_fixture_class,
        "created_at": "2026-07-10T12:00:00Z",
        "subject_binding": {
            "pseudonymous_subject_id": "subject-001",
            "session_ids": [primary_id, repeat_id],
            "scan_acquisition_id": scan_id,
            "same_subject_attested": True,
            "attester_id": "rights-officer-01",
            "evidence": {
                "artifact_id": "same-subject-evidence",
                "sha256": _digest("same-subject-evidence"),
            },
        },
        "consent": {
            "pseudonymous_subject_id": "subject-001",
            "attester_id": "rights-officer-01",
            "scopes": list(REQUIRED_CONSENT_SCOPES),
            "valid_from": "2026-01-01T00:00:00Z",
            "expires_at": "2027-01-01T00:00:00Z",
            "revoked": False,
            "evidence": {
                "artifact_id": "consent-evidence",
                "sha256": _digest("consent-evidence"),
            },
        },
        "sessions": [
            _session_payload("primary", primary, primary_id),
            _session_payload("repeat", repeat, repeat_id),
        ],
        "independent_scan": {
            "acquisition_id": scan_id,
            "pseudonymous_subject_id": "subject-001",
            "acquired_at": "2026-07-03T10:00:00Z",
            "units": "meters",
            "scan_artifact": {
                "artifact_id": "neutral-scan",
                "sha256": _digest("neutral-scan"),
            },
            "provenance_evidence": {
                "artifact_id": "scan-provenance",
                "sha256": _digest("scan-provenance"),
            },
            "independent_from_reconstruction": True,
            "used_evaluation_photos": False,
            "used_candidate_mesh_as_geometry_source": False,
        },
        "reviewers": [
            {
                "reviewer_id": f"reviewer-{index}",
                "organization": f"Independent Studio {index}",
                "reviewed_at": f"2026-07-0{4 + index}T10:00:00Z",
                "scopes": list(REQUIRED_REVIEW_SCOPES),
                "decision": "approved",
                "independent_from_capture_and_fit": True,
                "evidence": {
                    "artifact_id": f"review-{index}",
                    "sha256": _digest(f"review-{index}"),
                },
            }
            for index in (1, 2)
        ],
    }


def _seal_qualification_job(
    service: ApplicationService,
    tmp_path: Path,
    *,
    profile: dict,
    report: dict,
) -> tuple[str, Path]:
    source = tmp_path / "identity-qualification-source.json"
    source.write_text("{}\n", encoding="utf-8")
    job_id, job_dir, _, manifest = service.store.start(
        "identity_qualification", source, {}
    )
    (job_dir / "identity-profile.json").write_text(
        json.dumps(profile, sort_keys=True), encoding="utf-8"
    )
    (job_dir / "identity-report.json").write_text(
        json.dumps(report, sort_keys=True), encoding="utf-8"
    )
    service.store.finish(
        manifest,
        job_dir,
        {
            "kind": "identity_qualification",
            "artifacts": {
                "identity_qualification_profile": "identity-profile.json",
                "identity_qualification_report": "identity-report.json",
            },
            "warnings": [],
        },
        {},
    )
    return job_id, job_dir


def _seal_performance_job(
    service: ApplicationService, tmp_path: Path
) -> str:
    source = tmp_path / "performance.wav"
    source.write_bytes(b"RIFF")
    job_id, job_dir, _, manifest = service.store.start(
        "audio_animation", source, {}
    )
    (job_dir / "take.glb").write_bytes(b"glTF")
    service.store.finish(
        manifest,
        job_dir,
        {
            "kind": "audio_animation",
            "model": {"character": None},
            "analysis": {"motion_backend": "fallback"},
            "animation": {"production_validated": False},
            "quality": {"production_gate": {"passed": False, "failures": []}},
            "oral_validation": {"production_validated": False},
            "viewer": {"status": "ready", "glb_covers_full_track": True},
            "artifacts": {"glb": "take.glb"},
            "warnings": [],
        },
        {},
    )
    return job_id


@pytest.fixture
def bundles() -> dict[str, CalibratedCameraBundle]:
    return {"primary": _bundle("primary"), "repeat": _bundle("repeat")}


@pytest.fixture
def profile_payload(bundles: dict[str, CalibratedCameraBundle]) -> dict:
    return _profile_payload(bundles["primary"], bundles["repeat"])


def test_synthetic_contract_is_deterministic_but_cannot_authorize(
    bundles: dict[str, CalibratedCameraBundle], profile_payload: dict
) -> None:
    document = build_identity_qualification_profile(profile_payload)
    profile = load_identity_qualification_profile(document)
    first = build_identity_qualification_report(profile, camera_bundles=bundles)
    second = build_identity_qualification_report(document, camera_bundles=bundles)

    assert first == second
    assert first["contract_gate_passed"] is True
    assert first["declared_fixture_class"] == "synthetic"
    assert first["fixture_class_resolved"] is False
    assert set(first["declaration_gates"]) == {
        "same_subject_self_attested",
        "consent_declared_active_for_capture_scan_review_and_profile",
        "required_consent_scopes_declared",
        "independent_metric_scan_self_attested",
        "minimum_independent_reviewer_approvals_declared",
    }
    assert first["raw_calibration_recomputed"] is False
    assert first["scan_metrics_recomputed"] is False
    assert first["repeat_geometry_recomputed"] is False
    assert first["asset_identity_validated"] is False
    assert first["production_validated"] is False
    assert first["failures"] == [
        "FIXTURE_CLASS_NOT_INDEPENDENTLY_RESOLVED",
        "RAW_CALIBRATION_NOT_RECOMPUTED",
        "REPEAT_GEOMETRY_NOT_RECOMPUTED",
        "SCAN_METRICS_NOT_RECOMPUTED",
        "SYNTHETIC_FIXTURE",
    ]
    assert len(first["sessions"]) == 2
    assert all(value["contract_gate_passed"] for value in first["sessions"])
    assert load_identity_qualification_report(first) == first


def test_real_declarations_still_cannot_authorize_before_future_evaluators(
    bundles: dict[str, CalibratedCameraBundle],
) -> None:
    profile = build_identity_qualification_profile(
        _profile_payload(
            bundles["primary"],
            bundles["repeat"],
            declared_fixture_class="real_consented_subject",
        )
    )
    report = build_identity_qualification_report(profile, camera_bundles=bundles)

    assert report["contract_gate_passed"] is True
    assert "SYNTHETIC_FIXTURE" not in report["failures"]
    assert "FIXTURE_CLASS_NOT_INDEPENDENTLY_RESOLVED" in report["failures"]
    assert report["declared_fixture_class"] == "real_consented_subject"
    assert report["fixture_class_resolved"] is False
    assert report["asset_identity_validated"] is False
    assert report["production_validated"] is False


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        (lambda value: value["sessions"][1].__setitem__("session_id", "session-primary-01"), "INVALID_SESSIONS"),
        (
            lambda value: value["sessions"][1]["view_sources"][0].__setitem__(
                "sha256", value["sessions"][0]["view_sources"][0]["sha256"]
            ),
            "DUPLICATE_SOURCE",
        ),
        (lambda value: value["sessions"][0]["declared_fit_view_indices"].pop(), "INSUFFICIENT_VIEWS"),
        (lambda value: value["sessions"][0]["declared_held_out_view_indices"].pop(), "INSUFFICIENT_VIEWS"),
        (lambda value: value["sessions"][0].__setitem__("declared_camera_center_yaw_span_degrees", 119.9), "INSUFFICIENT_YAW"),
        (lambda value: value["subject_binding"].__setitem__("same_subject_attested", False), "SUBJECT_UNBOUND"),
        (lambda value: value["subject_binding"].__setitem__("pseudonymous_subject_id", "other"), "SUBJECT_UNBOUND"),
        (lambda value: value["consent"]["scopes"].pop(), "INVALID_SCOPE"),
        (lambda value: value["consent"].__setitem__("revoked", True), "CONSENT_INACTIVE"),
        (lambda value: value["consent"].__setitem__("expires_at", "2026-01-02T00:00:00Z"), "CONSENT_INACTIVE"),
        (lambda value: value["consent"].__setitem__("valid_from", "2026-07-04T00:00:00Z"), "CONSENT_INACTIVE"),
        (lambda value: value["independent_scan"].__setitem__("units", "millimeters"), "SCAN_NOT_INDEPENDENT"),
        (lambda value: value["independent_scan"].__setitem__("used_evaluation_photos", True), "SCAN_NOT_INDEPENDENT"),
        (lambda value: value["reviewers"].pop(), "INSUFFICIENT_REVIEWS"),
        (lambda value: value["reviewers"][1].__setitem__("reviewer_id", "REVIEWER-1"), "DUPLICATE_REVIEWER"),
        (
            lambda value: value["reviewers"][1]["evidence"].__setitem__(
                "sha256", value["reviewers"][0]["evidence"]["sha256"]
            ),
            "DUPLICATE_REVIEWER",
        ),
        (lambda value: value["reviewers"][0].__setitem__("decision", "rejected"), "REVIEW_NOT_APPROVED"),
        (lambda value: value["reviewers"][0].__setitem__("reviewed_at", "2026-07-01T00:00:00Z"), "INVALID_EVIDENCE_TIME"),
    ),
)
def test_profile_rejects_adversarial_declarations(
    profile_payload: dict, mutation, code: str
) -> None:
    payload = deepcopy(profile_payload)
    mutation(payload)

    with pytest.raises(IdentityQualificationError) as caught:
        build_identity_qualification_profile(payload)
    assert caught.value.code == code


def test_profile_loader_rejects_duplicate_keys_nonfinite_and_bounds(
    tmp_path: Path, profile_payload: dict
) -> None:
    with pytest.raises(IdentityQualificationError) as duplicate:
        load_identity_qualification_profile(
            b'{"schema_version":"one","schema_version":"two"}'
        )
    assert duplicate.value.code == "DUPLICATE_KEY"

    nonfinite = deepcopy(profile_payload)
    nonfinite["sessions"][0]["declared_camera_center_yaw_span_degrees"] = float("nan")
    with pytest.raises(IdentityQualificationError) as invalid_number:
        build_identity_qualification_profile(nonfinite)
    assert invalid_number.value.code == "INVALID_DOCUMENT"

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{" + b" " * MAX_DOCUMENT_BYTES + b"}")
    with pytest.raises(IdentityQualificationError) as bounds:
        load_identity_qualification_profile(oversized)
    assert bounds.value.code == "DOCUMENT_SIZE"


def test_profile_hash_and_unknown_fields_fail_closed(profile_payload: dict) -> None:
    document = build_identity_qualification_profile(profile_payload)
    tampered = deepcopy(document)
    tampered["created_at"] = "2026-07-11T12:00:00Z"
    with pytest.raises(IdentityQualificationError) as digest:
        load_identity_qualification_profile(tampered)
    assert digest.value.code == "PROFILE_HASH_MISMATCH"

    unknown = deepcopy(profile_payload)
    unknown["unexpected"] = True
    with pytest.raises(IdentityQualificationError) as fields:
        build_identity_qualification_profile(unknown)
    assert fields.value.code == "INVALID_FIELDS"


def test_report_records_bundle_binding_and_coverage_failures(
    bundles: dict[str, CalibratedCameraBundle], profile_payload: dict
) -> None:
    profile = build_identity_qualification_profile(profile_payload)
    narrow = _bundle(
        "primary",
        fit_yaws=(-30.0, -15.0, 0.0, 15.0, 30.0),
        source_hash=bundles["primary"].source_sha256,
    )
    bad_bundles = {"primary": narrow, "repeat": bundles["repeat"]}
    report = build_identity_qualification_report(profile, camera_bundles=bad_bundles)

    assert report["contract_gate_passed"] is False
    assert "primary:CAMERA_CENTER_YAW_SPAN_FAILED" in report["failures"]
    assert "primary:DECLARED_CAMERA_YAW_MISMATCH" in report["failures"]
    assert any(
        value["code"] == "CAMERA_CENTER_YAW_SPAN_FAILED"
        for value in report["remediation"]
    )


def test_report_rejects_bundle_hash_and_calibration_metadata_mismatch(
    bundles: dict[str, CalibratedCameraBundle], profile_payload: dict
) -> None:
    profile = build_identity_qualification_profile(profile_payload)
    invalid = _bundle(
        "primary",
        source_hash=_digest("wrong-bundle"),
        calibration_rms_px=0.8,
    )
    report = build_identity_qualification_report(
        profile, camera_bundles={"primary": invalid, "repeat": bundles["repeat"]}
    )

    assert "primary:CAMERA_BUNDLE_HASH_MISMATCH" in report["failures"]
    assert "primary:CALIBRATION_METADATA_GATE_FAILED" in report["failures"]


def test_report_loader_rejects_tamper_claim_escalation_and_unknown_fields(
    bundles: dict[str, CalibratedCameraBundle], profile_payload: dict
) -> None:
    profile = build_identity_qualification_profile(profile_payload)
    report = build_identity_qualification_report(profile, camera_bundles=bundles)

    tampered = deepcopy(report)
    tampered["profile_sha256"] = _digest("other-profile")
    with pytest.raises(IdentityQualificationError) as digest:
        load_identity_qualification_report(tampered)
    assert digest.value.code == "REPORT_HASH_MISMATCH"

    escalated = deepcopy(report)
    escalated["production_validated"] = True
    escalated["report_sha256"] = report_payload_sha256(escalated)
    with pytest.raises(IdentityQualificationError) as claim:
        load_identity_qualification_report(escalated)
    assert claim.value.code == "UNSUPPORTED_CLAIM"

    unknown = deepcopy(report)
    unknown["unexpected"] = True
    unknown["report_sha256"] = report_payload_sha256(unknown)
    with pytest.raises(IdentityQualificationError) as fields:
        load_identity_qualification_report(unknown)
    assert fields.value.code == "INVALID_FIELDS"

    nested_unknown = deepcopy(report)
    nested_unknown["sessions"][0]["unexpected"] = True
    nested_unknown["report_sha256"] = report_payload_sha256(nested_unknown)
    with pytest.raises(IdentityQualificationError) as nested:
        load_identity_qualification_report(nested_unknown)
    assert nested.value.code == "INVALID_FIELDS"

    omitted_blocker = deepcopy(report)
    omitted_blocker["failures"].remove("SCAN_METRICS_NOT_RECOMPUTED")
    omitted_blocker["remediation"] = [
        value
        for value in omitted_blocker["remediation"]
        if value["code"] != "SCAN_METRICS_NOT_RECOMPUTED"
    ]
    omitted_blocker["report_sha256"] = report_payload_sha256(omitted_blocker)
    with pytest.raises(IdentityQualificationError) as omission:
        load_identity_qualification_report(omitted_blocker)
    assert omission.value.code == "UNSUPPORTED_CLAIM"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda report: report["sessions"][0].__setitem__("fit_view_count", 4),
        lambda report: report["sessions"][0].__setitem__(
            "held_out_view_count", 1
        ),
        lambda report: report["sessions"][0].__setitem__(
            "recomputed_camera_center_yaw_span_degrees", 100.0
        ),
        lambda report: report["sessions"][0].__setitem__(
            "declared_calibration_metadata_gate_passed", False
        ),
    ),
)
def test_report_loader_rederives_camera_gate_failures(
    bundles: dict[str, CalibratedCameraBundle],
    profile_payload: dict,
    mutation,
) -> None:
    profile = build_identity_qualification_profile(profile_payload)
    report = build_identity_qualification_report(profile, camera_bundles=bundles)
    forged = deepcopy(report)
    mutation(forged)
    forged["report_sha256"] = report_payload_sha256(forged)

    with pytest.raises(IdentityQualificationError) as caught:
        load_identity_qualification_report(forged)
    assert caught.value.code == "INVALID_REPORT"


def test_profile_report_verifier_rejects_self_hashed_session_rebinding(
    bundles: dict[str, CalibratedCameraBundle], profile_payload: dict
) -> None:
    profile = build_identity_qualification_profile(profile_payload)
    report = build_identity_qualification_report(profile, camera_bundles=bundles)
    for mutation in (
        "session_id",
        "bundle_artifact_id",
        "bundle_hash",
        "fit_partition_and_count",
        "held_partition_and_count",
        "declared_and_reported_yaw",
    ):
        forged = deepcopy(report)
        session = forged["sessions"][0]
        if mutation == "session_id":
            session["session_id"] = "forged-primary-session"
            forged["bindings"]["session_ids"][0] = "forged-primary-session"
        elif mutation == "bundle_artifact_id":
            session["camera_bundle_artifact_id"] = "forged-camera-bundle"
        elif mutation == "bundle_hash":
            session["camera_bundle_source_sha256"] = _digest("forged-bundle")
        elif mutation == "fit_partition_and_count":
            session["fit_view_indices"] = [0, 1, 2, 3, 4, 7]
            session["fit_view_count"] = 6
        elif mutation == "held_partition_and_count":
            session["held_out_view_indices"] = [5, 6, 7]
            session["held_out_view_count"] = 3
        else:
            session["declared_camera_center_yaw_span_degrees"] = 150.0
            session["recomputed_camera_center_yaw_span_degrees"] = 150.0
        forged["report_sha256"] = report_payload_sha256(forged)

        # The standalone report is internally consistent. Only the paired
        # verifier can prove that its declarations were rebound.
        assert load_identity_qualification_report(forged) == forged
        with pytest.raises(IdentityQualificationError) as caught:
            verify_identity_qualification_report_profile(profile, forged)
        assert caught.value.code == "REPORT_PROFILE_BINDING_MISMATCH"


def test_profile_report_verifier_accepts_exact_session_bindings(
    bundles: dict[str, CalibratedCameraBundle], profile_payload: dict
) -> None:
    profile = build_identity_qualification_profile(profile_payload)
    report = build_identity_qualification_report(profile, camera_bundles=bundles)

    loaded_profile, loaded_report = verify_identity_qualification_report_profile(
        profile, report
    )

    assert loaded_profile.as_dict() == profile
    assert loaded_report == report
    for declared, observed in zip(
        loaded_profile.sessions, loaded_report["sessions"], strict=True
    ):
        assert observed["camera_bundle_artifact_id"] == declared.camera_bundle.artifact_id
        assert observed["camera_bundle_source_sha256"] == declared.camera_bundle.sha256
        assert observed["fit_view_indices"] == list(
            declared.declared_fit_view_indices
        )
        assert observed["held_out_view_indices"] == list(
            declared.declared_held_out_view_indices
        )


def test_camera_center_yaw_span_handles_branch_cut_and_rejects_origin() -> None:
    branch = _bundle(
        "branch",
        fit_yaws=(179.0, -179.0, 175.0, -175.0, 178.0),
    )
    assert camera_center_yaw_span_degrees(branch) == pytest.approx(10.0)

    views = list(branch.views)
    source = views[0]
    views[0] = CalibratedCameraView(
        index=source.index,
        image_name=source.image_name,
        role=source.role,
        usage=source.usage,
        image_size=source.image_size,
        intrinsics_matrix=source.intrinsics_matrix,
        distortion=source.distortion,
        world_to_camera=np.eye(4),
        visibility=source.visibility,
    )
    at_origin = CalibratedCameraBundle(
        calibration_rms_px=branch.calibration_rms_px,
        pose_error_degrees=branch.pose_error_degrees,
        scale_error_fraction=branch.scale_error_fraction,
        views=tuple(views),
        source_sha256=branch.source_sha256,
    )
    with pytest.raises(IdentityQualificationError) as caught:
        camera_center_yaw_span_degrees(at_origin)
    assert caught.value.code == "CAMERA_GEOMETRY_INVALID"


def test_profile_file_round_trip_is_canonical(
    tmp_path: Path, profile_payload: dict
) -> None:
    document = build_identity_qualification_profile(profile_payload)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    loaded = load_identity_qualification_profile(path)
    assert loaded.as_dict() == document


def test_sealed_i0_is_inspectable_but_never_authorizes_release_claims(
    tmp_path: Path,
    bundles: dict[str, CalibratedCameraBundle],
    profile_payload: dict,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "missing.task")
    service = app.state.service
    profile = build_identity_qualification_profile(profile_payload)
    report = build_identity_qualification_report(profile, camera_bundles=bundles)
    qualification_job_id, _ = _seal_qualification_job(
        service, tmp_path, profile=profile, report=report
    )

    inspected = service.identity_qualification(qualification_job_id)
    assert inspected["artifacts_verified"] is True
    assert set(inspected["artifacts"]) == {
        "identity_qualification_profile",
        "identity_qualification_report",
    }
    assert inspected["profile"]["session_count"] == 2
    assert inspected["report"]["reported_contract_gate_passed"] is True
    assert inspected["report"]["contract_gate_independently_recomputed"] is False
    assert inspected["report"]["fixture_class_resolved"] is False
    assert inspected["report"]["raw_calibration_recomputed"] is False
    assert inspected["report"]["scan_metrics_recomputed"] is False
    assert inspected["report"]["repeat_geometry_recomputed"] is False
    assert inspected["report"]["asset_identity_validated"] is False
    assert inspected["report"]["production_validated"] is False
    assert inspected["claim_authorizing"] is False

    client = TestClient(app)
    response = client.get(
        f"/api/jobs/{qualification_job_id}/identity-qualification"
    )
    assert response.status_code == 200
    assert response.json() == inspected

    performance_job_id = _seal_performance_job(service, tmp_path)
    readiness = client.get(
        f"/api/jobs/{performance_job_id}/production-readiness",
        params={"identity_qualification_job_id": qualification_job_id},
    )
    assert readiness.status_code == 200
    readiness_report = readiness.json()
    i0 = readiness_report["gates"]["identity"]["evidence"][
        "identity_capture_i0"
    ]
    assert i0["artifacts_verified"] is True
    assert i0["reported_contract_gate_passed"] is True
    assert i0["contract_gate_independently_recomputed"] is False
    assert i0["fixture_class_resolved"] is False
    assert i0["asset_identity_validated"] is False
    assert i0["production_validated"] is False
    assert i0["pbr_validated"] is False
    assert i0["texture_validated"] is False
    assert i0["claim_authorizing"] is False
    assert readiness_report["gates"]["identity"]["passed"] is False
    assert readiness_report["publishable"] is False

    parsed = build_parser().parse_args(
        [
            "inspect-identity-qualification",
            qualification_job_id,
            "--artifacts",
            str(service.store.root),
        ]
    )
    assert parsed.job_id == qualification_job_id
    assert cli_main(
        [
            "--model-path",
            str(tmp_path / "missing.task"),
            "inspect-identity-qualification",
            qualification_job_id,
            "--artifacts",
            str(service.store.root),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["claim_authorizing"] is False

    page = client.get("/")
    assert page.status_code == 200
    assert "at least 5 fit and 2 held-out cameras" in page.text
    assert "at least 3 fit cameras and 1 held-out camera" not in page.text


def test_i0_inspection_rejects_tamper_and_cross_bound_documents(
    tmp_path: Path,
    bundles: dict[str, CalibratedCameraBundle],
    profile_payload: dict,
) -> None:
    service = ApplicationService(
        tmp_path / "jobs", model_path=tmp_path / "missing.task"
    )
    profile = build_identity_qualification_profile(profile_payload)
    report = build_identity_qualification_report(profile, camera_bundles=bundles)
    tampered_job_id, tampered_job_dir = _seal_qualification_job(
        service, tmp_path, profile=profile, report=report
    )
    (tampered_job_dir / "identity-report.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(AutoAnimError) as tampered:
        service.identity_qualification(tampered_job_id)
    assert tampered.value.code == "INTEGRITY_FAILED"

    rebound_report = deepcopy(report)
    rebound_report["sessions"][0]["camera_bundle_source_sha256"] = _digest(
        "rebound-camera-bundle"
    )
    rebound_report["report_sha256"] = report_payload_sha256(rebound_report)
    rebound_job_id, _ = _seal_qualification_job(
        service, tmp_path, profile=profile, report=rebound_report
    )
    with pytest.raises(AutoAnimError) as rebound:
        service.identity_qualification(rebound_job_id)
    assert rebound.value.code == "INTEGRITY_FAILED"
    assert rebound.value.details["qualification_code"] == (
        "REPORT_PROFILE_BINDING_MISMATCH"
    )

    other_payload = _profile_payload(
        bundles["primary"],
        bundles["repeat"],
        declared_fixture_class="real_consented_subject",
    )
    other_profile = build_identity_qualification_profile(other_payload)
    other_report = build_identity_qualification_report(
        other_profile, camera_bundles=bundles
    )
    cross_bound_job_id, _ = _seal_qualification_job(
        service, tmp_path, profile=profile, report=other_report
    )
    with pytest.raises(AutoAnimError) as cross_bound:
        service.identity_qualification(cross_bound_job_id)
    assert cross_bound.value.code == "INTEGRITY_FAILED"

    response = TestClient(
        create_app(
            service.store.root,
            model_path=tmp_path / "missing.task",
        )
    ).get(f"/api/jobs/{cross_bound_job_id}/identity-qualification")
    assert response.status_code == 409
    assert response.json()["code"] == "INTEGRITY_FAILED"


def test_invalid_i0_reference_is_reported_without_changing_readiness(
    tmp_path: Path,
) -> None:
    service = ApplicationService(
        tmp_path / "jobs", model_path=tmp_path / "missing.task"
    )
    performance_job_id = _seal_performance_job(service, tmp_path)

    report = service.production_readiness(
        performance_job_id,
        identity_qualification_job_id="00000000000000000000000000",
    )

    evidence = report["gates"]["identity"]["evidence"]["identity_capture_i0"]
    assert evidence["artifacts_verified"] is False
    assert evidence["resolution_error"] == "JOB_NOT_FOUND"
    assert evidence["claim_authorizing"] is False
    assert report["gates"]["identity"]["passed"] is False
    assert report["publishable"] is False
