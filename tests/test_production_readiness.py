import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from autoanim_gnm.api import create_app
from autoanim_gnm.production_readiness import (
    SCHEMA_VERSION,
    evaluate_production_readiness,
)
from autoanim_gnm.phone_events import (
    PHONE_EVENT_SCHEMA_VERSION,
    PHONE_TIMING_REPORT_SCHEMA_VERSION,
)


def _approved_fixture() -> tuple[dict, dict, dict]:
    character_ref = {
        "character_id": "01abc",
        "revision_id": "01rev",
        "revision_manifest_sha256": "revision-sha",
        "identity_sha256": "identity-sha",
        "runtime_material_sha256s": {
            "base_color": "a",
            "normal": "b",
            "roughness": "c",
            "specular_color": "d",
        },
    }
    performance = {
        "job_id": "01take",
        "kind": "audio_animation",
        "status": "succeeded",
        "model": {
            "character": character_ref,
            "character_pbr_runtime_applied_to_glb": True,
        },
        "analysis": {
            "motion_backend": "learned_a2f",
            "phone_evidence": {
                "present": True,
                "independently_reviewed": True,
                "production_review_complete": True,
            },
        },
        "phone_timing": {"production_gate": {"passed": True, "failures": []}},
        "animation": {"production_validated": True},
        "quality": {"production_gate": {"passed": True, "failures": []}},
        "oral_validation": {
            "all_control_frames_evaluated": True,
            "viewer_structural_reconstruction_validated": True,
            "tongue_control_active_frames": 18,
            "tongue_teeth_collision_risk_frames": 0,
            "lip_order_inversion_risk_frames": 0,
            "production_validated": True,
        },
        "viewer": {"status": "ready", "glb_covers_full_track": True},
        "artifacts": {"glb": {"name": "take.glb"}},
    }
    character = {
        "character_id": "01abc",
        "revision_id": "01rev",
        "_manifest_sha256": "revision-sha",
        "_runtime_material_sha256s": dict(
            character_ref["runtime_material_sha256s"]
        ),
        "gnm": {"identity_sha256": "identity-sha"},
        "source": {"fit_production_validated": True},
        "appearance": {
            "pore_claim_gate_passed": True,
            "relightable_claim_gate_passed": True,
            "production_validated": True,
        },
        "body": {"status": "attached", "head_attachment_validated": True},
        "production_validated": True,
    }
    direction = {
        "job_id": "01direction",
        "kind": "acting_direction",
        "status": "succeeded",
        "source": {"job_id": "01take"},
        "direction": {
            "approval_status": "approved",
            "body_preview_approval_status": "approved",
            "production_validated": True,
        },
    }
    return performance, character, direction


def test_complete_release_evidence_can_pass_every_required_gate() -> None:
    performance, character, direction = _approved_fixture()

    report = evaluate_production_readiness(
        performance,
        performance_manifest_verified=True,
        source_input_verified=True,
        delivery_artifact_verified=True,
        phone_evidence_artifacts_verified=True,
        character_revision=character,
        direction=direction,
        direction_manifest_verified=True,
        require_acting=True,
        require_body=True,
        require_pbr=True,
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["publishable"] is True
    assert report["status"] == "ready"
    assert report["failures"] == []
    assert report["passed_required_gate_count"] == report["required_gate_count"]
    assert all(gate["passed"] for gate in report["gates"].values())


def test_unverified_external_sequence_controls_cannot_enter_production_allowlist() -> None:
    performance, character, _ = _approved_fixture()
    performance["analysis"] = {
        "motion_backend": "unverified_external_sequence_controls_candidate",
        "phone_evidence": {
            "present": True,
            "independently_reviewed": True,
            "production_review_complete": True,
        },
        "sequence_import": {
            "production_qualified": True,
            "worker_authentication_verified": True,
            "sdk_recurrent_state_verified": True,
        },
    }

    report = evaluate_production_readiness(
        performance,
        performance_manifest_verified=True,
        source_input_verified=True,
        delivery_artifact_verified=True,
        phone_evidence_artifacts_verified=True,
        character_revision=character,
    )

    assert report["gates"]["performance"]["passed"] is False
    assert report["publishable"] is False


def test_plausible_audio_take_remains_blocked_without_independent_evidence() -> None:
    performance, _, _ = _approved_fixture()
    performance["model"]["character"] = None
    performance["quality"]["production_gate"] = {
        "passed": False,
        "failures": ["independent_annotations", "timing_error_p95"],
    }
    performance["animation"]["production_validated"] = False
    performance["oral_validation"]["production_validated"] = False

    report = evaluate_production_readiness(
        performance,
        performance_manifest_verified=True,
        source_input_verified=True,
        delivery_artifact_verified=True,
    )

    assert report["publishable"] is False
    assert report["status"] == "blocked"
    assert report["failures"] == [
        "character_revision",
        "identity",
        "appearance",
        "oral_animation",
        "performance",
    ]
    assert report["gates"]["performance"]["evidence"]["quality_failures"] == [
        "independent_annotations",
        "timing_error_p95",
    ]
    assert report["gates"]["acting"]["required"] is False
    assert report["gates"]["body"]["required"] is False
    assert report["advisories"] == ["acting", "body"]


def test_audio_release_gate_requires_hash_verified_phone_artifacts() -> None:
    performance, character, _ = _approved_fixture()

    report = evaluate_production_readiness(
        performance,
        performance_manifest_verified=True,
        source_input_verified=True,
        delivery_artifact_verified=True,
        phone_evidence_artifacts_verified=False,
        character_revision=character,
    )

    assert report["publishable"] is False
    assert report["failures"] == ["performance"]
    assert report["gates"]["performance"]["evidence"][
        "phone_evidence_artifacts_verified"
    ] is False


def test_service_rejects_schema_headers_with_missing_atomic_phone_evidence(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "missing.task")
    service = app.state.service
    source = tmp_path / "source.wav"
    source.write_bytes(b"bound source audio")
    job_id, job_dir, retained, manifest = service.store.start(
        "audio_animation", source, {}
    )
    annotation_path = job_dir / "phone-annotations.TextGrid"
    annotation_path.write_bytes(b"immutable TextGrid evidence")
    audio_hash = hashlib.sha256(retained.read_bytes()).hexdigest()
    textgrid_hash = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    event_document = {
        "schema_version": PHONE_EVENT_SCHEMA_VERSION,
        "bindings": {
            "audio_sha256": audio_hash,
            "textgrid_sha256": textgrid_hash,
        },
        "review": {
            "independently_reviewed": True,
            "production_review_complete": True,
        },
        "event_count": 100,
    }
    timing_document = {
        "schema_version": PHONE_TIMING_REPORT_SCHEMA_VERSION,
        "annotation_bindings": dict(event_document["bindings"]),
        "production_gate": {"passed": False, "failures": ["diagnostic_only"]},
    }
    (job_dir / "phone-events.json").write_text(
        json.dumps(event_document), encoding="utf-8"
    )
    timing_path = job_dir / "phone-timing-report.json"
    timing_path.write_text(json.dumps(timing_document), encoding="utf-8")
    service.store.finish(
        manifest,
        job_dir,
        {
            "kind": "audio_animation",
            "analysis": {
                "motion_backend": "learned_a2f",
                "phone_evidence": {
                    "present": True,
                    "independently_reviewed": True,
                    "production_review_complete": True,
                    "event_count": 100,
                },
            },
            "phone_timing": timing_document,
            "artifacts": {
                "phone_annotations": annotation_path.name,
                "phone_events": "phone-events.json",
                "phone_timing_report": timing_path.name,
            },
            "warnings": [],
        },
        {},
    )

    rejected = service.production_readiness(job_id)
    assert rejected["gates"]["performance"]["evidence"][
        "phone_evidence_artifacts_verified"
    ] is False

    timing_path.write_text("{}", encoding="utf-8")
    tampered = service.production_readiness(job_id)
    assert tampered["gates"]["performance"]["evidence"][
        "phone_evidence_artifacts_verified"
    ] is False


def test_video_gate_requires_subject_and_labeled_neutral_calibration() -> None:
    performance, character, _ = _approved_fixture()
    performance.update(
        {
            "kind": "video_performance",
            "capture": {
                "production_validated": True,
                "performance_evidence_schema_version": (
                    "autoanim.performance-evidence.v2"
                ),
                "performance_evidence_policy": (
                    "observation_only_no_motion_effect"
                ),
            },
            "retargeting": {
                "subject_calibrated": False,
                "neutral_baseline_validated": False,
            },
            "metrics": {
                "face_presence_fraction": 1.0,
                "final_expression_motion_correlation": 0.95,
                "negative_baseline_residual_clipped_fraction": 0.0,
            },
        }
    )
    performance["artifacts"]["performance_evidence"] = {
        "name": "performance-evidence.json"
    }

    report = evaluate_production_readiness(
        performance,
        performance_manifest_verified=True,
        source_input_verified=True,
        delivery_artifact_verified=True,
        performance_evidence_artifact_verified=True,
        character_revision=character,
    )

    assert report["publishable"] is False
    assert report["failures"] == ["performance"]
    assert report["gates"]["performance"]["evidence"]["face_presence_fraction"] == 1.0
    assert report["gates"]["performance"]["passed"] is False

    performance["retargeting"] = {
        "subject_calibrated": True,
        "neutral_baseline_validated": True,
    }
    missing_artifact = evaluate_production_readiness(
        performance,
        performance_manifest_verified=True,
        source_input_verified=True,
        delivery_artifact_verified=True,
        phone_evidence_artifacts_verified=True,
        character_revision=character,
    )
    assert missing_artifact["gates"]["performance"]["passed"] is False
    assert missing_artifact["gates"]["performance"]["evidence"][
        "performance_evidence_artifact_verified"
    ] is False

    performance["capture"].update(
        {
            "observation_v3_schema_version": "autoanim.performance-evidence.v3",
            "observation_v3_arrays_schema_version": (
                "autoanim.pixel-observation/1.0"
            ),
            "observation_v3_policy": (
                "observation_only_pixel_diagnostics_no_motion_effect_v1"
            ),
            "observation_v3_consumed_by_retargeting": False,
            "capture_session_schema_version": "autoanim.capture-session.v1",
        }
    )
    structural_only = evaluate_production_readiness(
        performance,
        performance_manifest_verified=True,
        source_input_verified=True,
        delivery_artifact_verified=True,
        performance_evidence_artifact_verified=True,
        observation_v3_artifacts_verified=True,
        capture_session_artifact_verified=True,
        capture_session_production_claims_verified=False,
        character_revision=character,
    )
    assert structural_only["gates"]["performance"]["passed"] is False
    assert structural_only["gates"]["performance"]["evidence"][
        "capture_session_production_claims_verified"
    ] is False


def test_enabled_audio_visual_repair_is_a_required_unqualified_gate() -> None:
    performance, character, _ = _approved_fixture()
    performance.update(
        {
            "kind": "video_performance",
            "capture": {
                "production_validated": True,
                "performance_evidence_schema_version": (
                    "autoanim.performance-evidence.v2"
                ),
                "performance_evidence_policy": "observation_only_no_motion_effect",
                "observation_v3_schema_version": "autoanim.performance-evidence.v3",
                "observation_v3_arrays_schema_version": (
                    "autoanim.pixel-observation/1.0"
                ),
                "observation_v3_policy": (
                    "observation_only_pixel_diagnostics_no_motion_effect_v1"
                ),
                "observation_v3_consumed_by_retargeting": False,
                "capture_session_schema_version": "autoanim.capture-session.v1",
            },
            "retargeting": {
                "subject_calibrated": True,
                "neutral_baseline_validated": True,
                "audio_visual_repair": {
                    "schemaVersion": "autoanim.audio-visual-repair.v2",
                    "policy": "video_authoritative_conservative_audio_repair_v2",
                    "status": "repaired",
                    "locks": {
                        "upperFaceExact": True,
                        "pupilExact": True,
                        "headPoseAndTranslationExact": True,
                        "sourcePtsAndTimestampsExact": True,
                        "visibleContactProtectedByVisualOwnership": True,
                        "mouthContinuityGeometryValidated": True,
                        "tongueCoefficientContinuityValidated": True,
                    },
                    "claims": {
                        "tongueVisibleValidated": False,
                        "contradictoryMediaValidated": False,
                        "productionValidated": False,
                    },
                },
            },
            "metrics": {},
        }
    )
    performance["artifacts"].update(
        {
            "performance_evidence": {"name": "performance-evidence.json"},
            "audio_visual_source": {"name": "audio-visual-source.json"},
            "audio_visual_repair": {"name": "audio-visual-repair.json"},
            "audio_visual_repair_arrays": {"name": "audio-visual-repair.npz"},
            "audio_visual_source_controls": {"name": "audio-visual-source-controls.npz"},
            "audio_visual_source_arkit_controls": {
                "name": "audio-visual-source-arkit-controls.npz"
            },
            "audio_visual_source_normalized_audio": {"name": "audio-visual-source.wav"},
            "audio_visual_source_raw": {"name": "audio-visual-source-a2f.jsonl"},
            "audio_visual_source_retarget_calibration": {
                "name": "audio-visual-source-retarget-calibration.npz"
            },
            "audio_visual_source_rhubarb": {"name": "audio-visual-source-rhubarb.json"},
            "audio_visual_source_cues": {"name": "audio-visual-source-cues.json"},
            "audio_visual_source_timeline": {"name": "audio-visual-source-timeline.json"},
            "audio_video_timing": {"name": "audio-video-timing.json"},
            "audio_visual_timing_consumption": {
                "name": "audio-visual-timing-consumption.json"
            },
            "performance_revision_chain": {
                "name": "performance-revision-chain.json"
            },
            "audio_visual_repair_qualification": {
                "name": "audio-visual-repair-qualification.json"
            },
        }
    )
    report = evaluate_production_readiness(
        performance,
        performance_manifest_verified=True,
        source_input_verified=True,
        delivery_artifact_verified=True,
        performance_evidence_artifact_verified=True,
        observation_v3_artifacts_verified=True,
        capture_session_artifact_verified=True,
        capture_session_production_claims_verified=True,
        character_revision=character,
    )
    assert report["failures"] == ["audio_visual_repair"]
    assert report["gates"]["audio_visual_repair"]["required"] is True
    assert report["gates"]["audio_visual_repair"]["passed"] is False
    assert report["gates"]["audio_visual_repair"]["evidence"][
        "tongue_visible_validated"
    ] is False

    repair = performance["retargeting"]["audio_visual_repair"]
    repair["claims"] = {
        "tongueVisibleValidated": True,
        "contradictoryMediaValidated": True,
        "artistPreferenceValidated": True,
        "qualificationProfileSha256": "a" * 64,
        "productionValidated": True,
    }
    unverified = evaluate_production_readiness(
        performance,
        performance_manifest_verified=True,
        source_input_verified=True,
        delivery_artifact_verified=True,
        performance_evidence_artifact_verified=True,
        observation_v3_artifacts_verified=True,
        capture_session_artifact_verified=True,
        capture_session_production_claims_verified=True,
        character_revision=character,
    )
    assert unverified["gates"]["audio_visual_repair"]["passed"] is False
    assert unverified["gates"]["audio_visual_repair"]["evidence"][
        "artifact_bytes_verified"
    ] is False
    verified = evaluate_production_readiness(
        performance,
        performance_manifest_verified=True,
        source_input_verified=True,
        delivery_artifact_verified=True,
        performance_evidence_artifact_verified=True,
        observation_v3_artifacts_verified=True,
        capture_session_artifact_verified=True,
        capture_session_production_claims_verified=True,
        audio_visual_repair_artifacts_verified=True,
        character_revision=character,
    )
    assert verified["gates"]["audio_visual_repair"]["passed"] is False
    assert verified["gates"]["audio_visual_repair"]["evidence"][
        "qualification_profile_bytes_verified"
    ] is False
    verified = evaluate_production_readiness(
        performance,
        performance_manifest_verified=True,
        source_input_verified=True,
        delivery_artifact_verified=True,
        performance_evidence_artifact_verified=True,
        observation_v3_artifacts_verified=True,
        capture_session_artifact_verified=True,
        capture_session_production_claims_verified=True,
        audio_visual_repair_artifacts_verified=True,
        audio_visual_repair_qualification_verified=True,
        character_revision=character,
    )
    assert verified["gates"]["audio_visual_repair"]["passed"] is True
    assert "audio_visual_repair" not in verified["failures"]


def test_character_and_material_hashes_must_match_exact_revision() -> None:
    performance, character, direction = _approved_fixture()
    character["gnm"]["identity_sha256"] = "different-identity"
    character["_runtime_material_sha256s"]["normal"] = "different-normal"

    report = evaluate_production_readiness(
        performance,
        performance_manifest_verified=True,
        source_input_verified=True,
        delivery_artifact_verified=True,
        character_revision=character,
        direction=direction,
        direction_manifest_verified=True,
        require_acting=True,
        require_body=True,
    )

    assert report["gates"]["character_revision"]["passed"] is False
    assert report["gates"]["appearance"]["passed"] is False
    assert report["gates"]["appearance"]["evidence"][
        "runtime_map_hashes_match_revision"
    ] is False
    assert report["publishable"] is False


def test_readiness_api_exposes_fail_closed_report(tmp_path: Path) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "missing.task")
    store = app.state.service.store
    source = tmp_path / "input.wav"
    source.write_bytes(b"RIFF")
    job_id, job_dir, _, manifest = store.start("audio_animation", source, {})
    (job_dir / "take.glb").write_bytes(b"glTF")
    finished = store.finish(
        manifest,
        job_dir,
        {
            "kind": "audio_animation",
            "model": {"character": None},
            "analysis": {"motion_backend": "learned_a2f"},
            "animation": {"production_validated": False},
            "quality": {
                "production_gate": {
                    "passed": False,
                    "failures": ["independent_annotations"],
                }
            },
            "oral_validation": {
                "all_control_frames_evaluated": True,
                "viewer_structural_reconstruction_validated": True,
                "production_validated": False,
            },
            "viewer": {"status": "ready", "glb_covers_full_track": True},
            "artifacts": {"glb": "take.glb"},
            "warnings": [],
        },
        {},
    )
    assert finished["job_id"] == job_id

    response = TestClient(app).get(
        f"/api/jobs/{job_id}/production-readiness?require_acting=true"
    )

    assert response.status_code == 200
    report = response.json()
    assert report["publishable"] is False
    assert report["requirements"]["acting"] is True
    assert "acting" in report["failures"]
    assert report["gates"]["delivery"]["passed"] is True

    (job_dir / "input.wav").write_bytes(b"tampered input")
    (job_dir / "take.glb").write_bytes(b"tampered glb")
    tampered = TestClient(app).get(
        f"/api/jobs/{job_id}/production-readiness"
    ).json()
    assert tampered["gates"]["provenance_integrity"]["passed"] is False
    assert tampered["gates"]["delivery"]["passed"] is False
    assert "provenance_integrity" in tampered["failures"]
    assert "delivery" in tampered["failures"]


def test_readiness_api_rejects_unknown_job(tmp_path: Path) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "missing.task")
    response = TestClient(app).get(
        "/api/jobs/00000000000000000000000000/production-readiness"
    )
    assert response.status_code == 404
    assert response.json()["code"] == "JOB_NOT_FOUND"
