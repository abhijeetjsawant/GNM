from pathlib import Path

from fastapi.testclient import TestClient

from autoanim_gnm.api import create_app
from autoanim_gnm.production_readiness import (
    SCHEMA_VERSION,
    evaluate_production_readiness,
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
        "analysis": {"motion_backend": "learned_a2f"},
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
        character_revision=character,
    )
    assert missing_artifact["gates"]["performance"]["passed"] is False
    assert missing_artifact["gates"]["performance"]["evidence"][
        "performance_evidence_artifact_verified"
    ] is False


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
