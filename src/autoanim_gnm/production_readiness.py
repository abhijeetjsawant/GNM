"""Fail-closed release evidence for one character performance take.

This module deliberately does not turn plausible proxy metrics into production
approval.  It normalizes the evidence already written by reconstruction,
audio, video, oral, acting, and export stages into one machine-readable gate.
"""

from __future__ import annotations

from typing import Any

from .audio_visual_repair import (
    AUDIO_VISUAL_REPAIR_POLICY,
    AUDIO_VISUAL_REPAIR_SCHEMA_VERSION,
)
from .phone_articulation import PHONE_ARTICULATION_REPORT_SCHEMA_VERSION


SCHEMA_VERSION = "autoanim.production-readiness/1.2"
_PERFORMANCE_KINDS = frozenset({"audio_animation", "video_performance"})
_PBR_RUNTIME_MAPS = frozenset(
    {"base_color", "normal", "roughness", "specular_color"}
)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _zero_count(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == 0
    )


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _gate(
    *,
    passed: bool,
    summary: str,
    evidence: dict[str, Any],
    remediation: str,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "required": required,
        "passed": bool(passed),
        "summary": summary,
        "evidence": evidence,
        "remediation": remediation,
    }


def evaluate_production_readiness(
    performance: dict[str, Any],
    *,
    performance_manifest_verified: bool = False,
    source_input_verified: bool = False,
    delivery_artifact_verified: bool = False,
    performance_evidence_artifact_verified: bool = False,
    observation_v3_artifacts_verified: bool = False,
    video_capture_run_artifact_verified: bool = False,
    visual_track_artifacts_verified: bool = False,
    capture_session_artifact_verified: bool = False,
    capture_session_production_claims_verified: bool = False,
    phone_evidence_artifacts_verified: bool = False,
    phone_evidence_artifact_failure_reason: str | None = None,
    audio_visual_repair_artifacts_verified: bool = False,
    audio_visual_repair_qualification_verified: bool = False,
    character_revision: dict[str, Any] | None = None,
    character_resolution_error: str | None = None,
    identity_qualification: dict[str, Any] | None = None,
    identity_qualification_job_id: str | None = None,
    identity_qualification_resolution_error: str | None = None,
    direction: dict[str, Any] | None = None,
    direction_manifest_verified: bool = False,
    require_acting: bool = False,
    require_body: bool = False,
    require_pbr: bool = True,
) -> dict[str, Any]:
    """Return one release contract without mutating or approving any asset."""

    kind = performance.get("kind")
    status = performance.get("status")
    model = _mapping(performance.get("model"))
    character_ref = _mapping(model.get("character"))
    revision = _mapping(character_revision)
    appearance = _mapping(revision.get("appearance"))
    oral = _mapping(performance.get("oral_validation"))
    viewer = _mapping(performance.get("viewer"))
    artifacts = _mapping(performance.get("artifacts"))

    expected_character = character_ref.get("character_id")
    expected_revision = character_ref.get("revision_id")
    expected_manifest_hash = character_ref.get("revision_manifest_sha256")
    expected_identity_hash = character_ref.get("identity_sha256")
    resolved_identity_hash = _mapping(revision.get("gnm")).get("identity_sha256")
    exact_character = bool(
        expected_character
        and expected_revision
        and revision.get("character_id") == expected_character
        and revision.get("revision_id") == expected_revision
        and isinstance(expected_manifest_hash, str)
        and revision.get("_manifest_sha256") == expected_manifest_hash
        and isinstance(expected_identity_hash, str)
        and resolved_identity_hash == expected_identity_hash
        and character_resolution_error is None
    )
    identity_qualification_summary = _mapping(identity_qualification)
    identity_qualification_report = _mapping(
        identity_qualification_summary.get("report")
    )
    identity_qualification_evidence = {
        "qualification_job_id": identity_qualification_job_id,
        "artifacts_verified": bool(
            identity_qualification_summary.get("artifacts_verified", False)
        ),
        "resolution_error": identity_qualification_resolution_error,
        "declared_fixture_class": identity_qualification_report.get(
            "declared_fixture_class"
        ),
        "fixture_class_resolved": False,
        "reported_contract_gate_passed": identity_qualification_report.get(
            "reported_contract_gate_passed", False
        ),
        "contract_gate_independently_recomputed": False,
        "raw_calibration_recomputed": identity_qualification_report.get(
            "raw_calibration_recomputed", False
        ),
        "scan_metrics_recomputed": identity_qualification_report.get(
            "scan_metrics_recomputed", False
        ),
        "repeat_geometry_recomputed": identity_qualification_report.get(
            "repeat_geometry_recomputed", False
        ),
        "asset_identity_validated": False,
        "production_validated": False,
        "pbr_validated": False,
        "texture_validated": False,
        "failures": _string_list(identity_qualification_report.get("failures")),
        # I0 schema v1 is declaration/coverage evidence and never authorizes a
        # likeness, material, texture, or production claim.
        "claim_authorizing": False,
    }

    gates: dict[str, dict[str, Any]] = {}
    gates["terminal_take"] = _gate(
        passed=status == "succeeded" and kind in _PERFORMANCE_KINDS,
        summary="The source is a successful immutable performance take.",
        evidence={"job_id": performance.get("job_id"), "kind": kind, "status": status},
        remediation="Run a successful Audio or Video performance job.",
    )
    gates["provenance_integrity"] = _gate(
        passed=performance_manifest_verified and source_input_verified,
        summary="The signed take manifest and retained source bytes match their ledger.",
        evidence={
            "performance_manifest_verified": performance_manifest_verified,
            "source_input_verified": source_input_verified,
        },
        remediation=(
            "Use a sealed job whose retained input still matches its recorded byte count and "
            "SHA-256 digest."
        ),
    )
    gates["character_revision"] = _gate(
        passed=exact_character,
        summary="The take resolves to one exact rights-cleared character revision.",
        evidence={
            "character_id": expected_character,
            "revision_id": expected_revision,
            "revision_manifest_sha256": expected_manifest_hash,
            "identity_sha256": expected_identity_hash,
            "resolution_error": character_resolution_error,
        },
        remediation=(
            "Select a saved, unexpired, unrevoked character revision whose consent grants "
            "the take's intended use."
        ),
    )
    gates["identity"] = _gate(
        passed=exact_character
        and bool(_mapping(revision.get("source")).get("fit_production_validated"))
        and bool(revision.get("production_validated")),
        summary="Identity likeness and hidden geometry passed independent validation.",
        evidence={
            "fit_production_validated": _mapping(revision.get("source")).get(
                "fit_production_validated", False
            ),
            "revision_production_validated": revision.get(
                "production_validated", False
            ),
            "identity_capture_i0": identity_qualification_evidence,
        },
        remediation=(
            "Validate the character against held-out calibrated views or an independent scan, "
            "then record artist/subject approval on a new immutable revision."
        ),
    )

    expected_runtime_map_hashes = _mapping(
        character_ref.get("runtime_material_sha256s")
    )
    resolved_runtime_map_hashes = _mapping(
        revision.get("_runtime_material_sha256s")
    )
    runtime_maps = set(expected_runtime_map_hashes)
    appearance_passed = bool(
        exact_character
        and appearance.get("production_validated")
        and appearance.get("pore_claim_gate_passed")
        and appearance.get("relightable_claim_gate_passed")
        and _PBR_RUNTIME_MAPS.issubset(runtime_maps)
        and expected_runtime_map_hashes == resolved_runtime_map_hashes
        and model.get("character_pbr_runtime_applied_to_glb")
    )
    gates["appearance"] = _gate(
        passed=appearance_passed,
        summary="The exact character has complete validated PBR appearance evidence.",
        evidence={
            "required_runtime_maps": sorted(_PBR_RUNTIME_MAPS),
            "runtime_maps": sorted(runtime_maps),
            "runtime_map_hashes_match_revision": (
                expected_runtime_map_hashes == resolved_runtime_map_hashes
            ),
            "pbr_applied_to_glb": model.get(
                "character_pbr_runtime_applied_to_glb", False
            ),
            "pore_claim_gate_passed": appearance.get("pore_claim_gate_passed", False),
            "relightable_claim_gate_passed": appearance.get(
                "relightable_claim_gate_passed", False
            ),
            "production_validated": appearance.get("production_validated", False),
        },
        remediation=(
            "Attach measured base-color, normal, roughness, and specular maps and pass held-out "
            "pore-frequency and unseen-light validation."
        ),
        required=require_pbr,
    )

    oral_passed = bool(
        oral.get("all_control_frames_evaluated")
        and oral.get("viewer_structural_reconstruction_validated")
        and oral.get("production_validated")
        and _zero_count(oral.get("tongue_teeth_collision_risk_frames"))
        and _zero_count(oral.get("lip_order_inversion_risk_frames"))
    )
    gates["oral_animation"] = _gate(
        passed=oral_passed,
        summary="Every exported frame passed lips, teeth, tongue, and perceptual approval.",
        evidence={
            "all_control_frames_evaluated": oral.get(
                "all_control_frames_evaluated", False
            ),
            "viewer_structural_reconstruction_validated": oral.get(
                "viewer_structural_reconstruction_validated", False
            ),
            "tongue_control_active_frames": oral.get(
                "tongue_control_active_frames", 0
            ),
            "tongue_teeth_collision_risk_frames": oral.get(
                "tongue_teeth_collision_risk_frames", 0
            ),
            "lip_order_inversion_risk_frames": oral.get(
                "lip_order_inversion_risk_frames", 0
            ),
            "production_validated": oral.get("production_validated", False),
        },
        remediation=(
            "Resolve every structural risk and approve tongue visibility, surface collision, "
            "phone timing, mouth aperture, and perceptual speech quality on real footage."
        ),
    )

    if kind == "audio_animation":
        analysis = _mapping(performance.get("analysis"))
        phone_evidence = _mapping(analysis.get("phone_evidence"))
        phone_timing_gate = _mapping(
            _mapping(performance.get("phone_timing")).get("production_gate")
        )
        phone_articulation = _mapping(performance.get("phone_articulation"))
        phone_articulation_gate = _mapping(
            phone_articulation.get("production_gate")
        )
        phone_articulation_present = bool(
            phone_articulation.get("schema_version")
            == PHONE_ARTICULATION_REPORT_SCHEMA_VERSION
        )
        quality_gate = _mapping(_mapping(performance.get("quality")).get("production_gate"))
        animation = _mapping(performance.get("animation"))
        performance_passed = bool(
            analysis.get("motion_backend") == "learned_a2f"
            and phone_evidence.get("present")
            and phone_evidence.get("independently_reviewed")
            and phone_evidence.get("production_review_complete")
            and phone_timing_gate.get("passed")
            and phone_articulation_present
            and phone_articulation_gate.get("passed")
            and phone_evidence_artifacts_verified
            and quality_gate.get("passed")
            and animation.get("production_validated")
        )
        performance_evidence = {
            "motion_backend": analysis.get("motion_backend"),
            "phone_evidence_present": phone_evidence.get("present", False),
            "phone_review_complete": phone_evidence.get(
                "production_review_complete", False
            ),
            "phone_timing_gate_passed": phone_timing_gate.get("passed", False),
            "phone_timing_failures": _string_list(
                phone_timing_gate.get("failures")
            ),
            "phone_articulation_gate_passed": phone_articulation_gate.get(
                "passed", False
            ),
            "phone_articulation_failures": _string_list(
                phone_articulation_gate.get("failures")
            )
            or (
                []
                if phone_articulation_present
                else ["phone_articulation_report_missing_or_legacy_schema"]
            ),
            "phone_evidence_artifacts_verified": phone_evidence_artifacts_verified,
            "phone_evidence_artifact_failure_reason": (
                None
                if phone_evidence_artifacts_verified
                else phone_evidence_artifact_failure_reason
                or "phone_evidence_missing_or_invalid"
            ),
            "independent_quality_gate_passed": quality_gate.get("passed", False),
            "quality_failures": _string_list(quality_gate.get("failures")),
            "animation_production_validated": animation.get(
                "production_validated", False
            ),
        }
        performance_remediation = (
            "Use the learned backend, score independent phone/contact annotations and target "
            "prototypes, then record human perceptual approval for this retarget profile."
        )
    elif kind == "video_performance":
        capture = _mapping(performance.get("capture"))
        retarget = _mapping(performance.get("retargeting"))
        metrics = _mapping(performance.get("metrics"))
        performance_passed = bool(
            capture.get("production_validated")
            and retarget.get("subject_calibrated")
            and retarget.get("neutral_baseline_validated")
            and capture.get("performance_evidence_schema_version")
            == "autoanim.performance-evidence.v2"
            and capture.get("performance_evidence_policy")
            == "observation_only_no_motion_effect"
            and performance_evidence_artifact_verified
            and capture.get("observation_v3_schema_version")
            == "autoanim.performance-evidence.v3"
            and capture.get("observation_v3_arrays_schema_version")
            == "autoanim.pixel-observation/1.0"
            and capture.get("observation_v3_policy")
            == "observation_only_pixel_diagnostics_no_motion_effect_v1"
            and capture.get("observation_v3_consumed_by_retargeting") is False
            and observation_v3_artifacts_verified
            and capture.get("video_capture_run_schema_version")
            == "autoanim.video-capture-run/1.0"
            and video_capture_run_artifact_verified
            and capture.get("visual_track_schema_version")
            == "autoanim.visual-track/1.0"
            and capture.get("visual_track_summary_schema_version")
            == "autoanim.visual-track-summary/1.0"
            and capture.get("visual_track_policy")
            == "shadow_observation_only_no_motion_effect_v1"
            and capture.get("visual_track_motion_authority") == "none"
            and capture.get("visual_track_consumed_by_retargeting") is False
            and capture.get("visual_track_detector_ingress_hashes_verified") is True
            and visual_track_artifacts_verified
            and capture.get("capture_session_schema_version")
            == "autoanim.capture-session.v2"
            and capture_session_artifact_verified
            and capture_session_production_claims_verified
        )
        performance_evidence = {
            "capture_production_validated": capture.get(
                "production_validated", False
            ),
            "subject_calibrated": retarget.get("subject_calibrated", False),
            "neutral_baseline_validated": retarget.get(
                "neutral_baseline_validated", False
            ),
            "performance_evidence_schema_version": capture.get(
                "performance_evidence_schema_version"
            ),
            "performance_evidence_artifact_verified": (
                performance_evidence_artifact_verified
            ),
            "observation_v3_schema_version": capture.get(
                "observation_v3_schema_version"
            ),
            "observation_v3_artifacts_verified": (
                observation_v3_artifacts_verified
            ),
            "observation_v3_consumed_by_retargeting": capture.get(
                "observation_v3_consumed_by_retargeting"
            ),
            "video_capture_run_schema_version": capture.get(
                "video_capture_run_schema_version"
            ),
            "video_capture_run_artifact_verified": (
                video_capture_run_artifact_verified
            ),
            "visual_track_schema_version": capture.get(
                "visual_track_schema_version"
            ),
            "visual_track_summary_schema_version": capture.get(
                "visual_track_summary_schema_version"
            ),
            "visual_track_policy": capture.get("visual_track_policy"),
            "visual_track_motion_authority": capture.get(
                "visual_track_motion_authority"
            ),
            "visual_track_consumed_by_retargeting": capture.get(
                "visual_track_consumed_by_retargeting"
            ),
            "visual_track_detector_ingress_hashes_verified": capture.get(
                "visual_track_detector_ingress_hashes_verified"
            ),
            "visual_track_artifacts_verified": visual_track_artifacts_verified,
            "capture_session_schema_version": capture.get(
                "capture_session_schema_version"
            ),
            "capture_session_artifact_verified": (
                capture_session_artifact_verified
            ),
            "capture_session_production_claims_verified": (
                capture_session_production_claims_verified
            ),
            "face_presence_fraction": metrics.get("face_presence_fraction"),
            "expression_motion_correlation": metrics.get(
                "final_expression_motion_correlation"
            ),
            "negative_baseline_residual_clipped_fraction": metrics.get(
                "negative_baseline_residual_clipped_fraction"
            ),
        }
        performance_remediation = (
            "Capture a labeled subject-neutral calibration, preserve a verified CaptureSession "
            "and regional pixel evidence, then validate expression, microexpression, gaze, head, "
            "mouth, and timing against held-out visual ground truth."
        )
    else:
        performance_passed = False
        performance_evidence = {"kind": kind}
        performance_remediation = "Use a supported Audio or Video performance job."
    gates["performance"] = _gate(
        passed=performance_passed,
        summary="The driving performance passed source-specific production validation.",
        evidence=performance_evidence,
        remediation=performance_remediation,
    )

    retarget = _mapping(performance.get("retargeting"))
    repair = _mapping(retarget.get("audio_visual_repair"))
    repair_status = repair.get("status")
    repair_enabled = bool(
        kind == "video_performance"
        and repair_status not in (None, "disabled")
    )
    repair_claims = _mapping(repair.get("claims"))
    repair_locks = _mapping(repair.get("locks"))
    repair_artifacts_present = all(
        isinstance(artifacts.get(name), dict)
        for name in (
            "audio_visual_source",
            "audio_visual_repair",
            "audio_visual_repair_arrays",
            "audio_visual_source_controls",
            "audio_visual_source_arkit_controls",
            "audio_visual_source_normalized_audio",
            "audio_visual_source_raw",
            "audio_visual_source_retarget_calibration",
            "audio_visual_source_rhubarb",
            "audio_visual_source_cues",
            "audio_visual_source_timeline",
            "audio_video_timing",
            "audio_visual_timing_consumption",
            "performance_revision_chain",
            "audio_visual_repair_qualification",
        )
    )
    repair_passed = bool(
        not repair_enabled
        or (
            repair.get("schemaVersion") == AUDIO_VISUAL_REPAIR_SCHEMA_VERSION
            and repair_status in {"repaired", "exact_noop"}
            and repair.get("policy") == AUDIO_VISUAL_REPAIR_POLICY
            and repair_locks.get("upperFaceExact")
            and repair_locks.get("pupilExact")
            and repair_locks.get("headPoseAndTranslationExact")
            and repair_locks.get("sourcePtsAndTimestampsExact")
            and repair_locks.get("visibleContactProtectedByVisualOwnership")
            and repair_locks.get("mouthContinuityGeometryValidated")
            and repair_locks.get("tongueCoefficientContinuityValidated")
            and repair_claims.get("tongueVisibleValidated")
            and repair_claims.get("contradictoryMediaValidated")
            and repair_claims.get("artistPreferenceValidated")
            and isinstance(
                repair_claims.get("qualificationProfileSha256"), str
            )
            and repair_claims.get("productionValidated")
            and repair_artifacts_present
            and audio_visual_repair_artifacts_verified
            and audio_visual_repair_qualification_verified
        )
    )
    gates["audio_visual_repair"] = _gate(
        passed=repair_passed,
        summary=(
            "Learned audio repair is either disabled or independently qualified against "
            "contradictory media, visible tongue footage, and a hash-bound artist-preference profile."
        ),
        evidence={
            "enabled": repair_enabled,
            "status": repair_status,
            "schema_version": repair.get("schemaVersion"),
            "policy": repair.get("policy"),
            "locks": repair_locks,
            "artifacts_present": repair_artifacts_present,
            "artifact_bytes_verified": audio_visual_repair_artifacts_verified,
            "qualification_profile_bytes_verified": (
                audio_visual_repair_qualification_verified
            ),
            "tongue_visible_validated": repair_claims.get(
                "tongueVisibleValidated", False
            ),
            "contradictory_media_validated": repair_claims.get(
                "contradictoryMediaValidated", False
            ),
            "artist_preference_validated": repair_claims.get(
                "artistPreferenceValidated", False
            ),
            "qualification_profile_sha256": repair_claims.get(
                "qualificationProfileSha256"
            ),
            "production_validated": repair_claims.get(
                "productionValidated", False
            ),
        },
        remediation=(
            "Evaluate the exact repair profile on retained labeled A/V takes, including "
            "contradictory dubbed media and visible tongue gestures; pass phone/contact timing, "
            "oral geometry, and blinded artist-preference gates before approval."
        ),
        required=repair_enabled,
    )

    viewer_passed = bool(
        viewer.get("status") == "ready"
        and viewer.get("glb_covers_full_track")
        and isinstance(artifacts.get("glb"), dict)
        and delivery_artifact_verified
    )
    gates["delivery"] = _gate(
        passed=viewer_passed,
        summary="The sealed interactive export covers the complete source-clocked take.",
        evidence={
            "viewer_status": viewer.get("status"),
            "glb_covers_full_track": viewer.get("glb_covers_full_track", False),
            "glb_artifact_present": isinstance(artifacts.get("glb"), dict),
            "glb_artifact_verified": delivery_artifact_verified,
        },
        remediation="Re-export a complete animated GLB and pass structural reconstruction checks.",
    )

    direction_document = _mapping(direction)
    direction_source = _mapping(direction_document.get("source"))
    direction_contract = _mapping(direction_document.get("direction"))
    direction_linked = bool(
        direction_document.get("status") == "succeeded"
        and direction_document.get("kind") == "acting_direction"
        and direction_source.get("job_id") == performance.get("job_id")
        and direction_manifest_verified
    )
    acting_passed = bool(
        direction_linked
        and direction_contract.get("approval_status") == "approved"
        and direction_contract.get("production_validated")
    )
    gates["acting"] = _gate(
        passed=acting_passed,
        summary="A linked acting plan was artist-approved after deterministic compilation.",
        evidence={
            "direction_job_id": direction_document.get("job_id"),
            "source_job_id": direction_source.get("job_id"),
            "approval_status": direction_contract.get(
                "approval_status",
                direction_contract.get("body_preview_approval_status"),
            ),
            "production_validated": direction_contract.get(
                "production_validated", False
            ),
        },
        remediation=(
            "Generate or attach the acting proposal, edit it, compile it, and record explicit "
            "artist approval; an LLM proposal alone is never approval."
        ),
        required=require_acting,
    )

    body = _mapping(revision.get("body"))
    body_passed = bool(
        exact_character
        and body.get("status") == "attached"
        and body.get("head_attachment_validated")
        and direction_linked
        and direction_contract.get("body_preview_approval_status") == "approved"
        and direction_contract.get("production_validated")
    )
    gates["body"] = _gate(
        passed=body_passed,
        summary="A skinned body, head attachment, contacts, and acting motion were approved.",
        evidence={
            "character_body_status": body.get("status", "not_attached"),
            "head_attachment_validated": body.get(
                "head_attachment_validated", False
            ),
            "body_motion_approval_status": direction_contract.get(
                "body_preview_approval_status"
            ),
        },
        remediation=(
            "Attach a production body rig, solve body/hands/feet, validate the GNM neck seam "
            "and contacts, then approve the compiled motion."
        ),
        required=require_body,
    )

    failures = [
        name
        for name, gate in gates.items()
        if gate["required"] and not gate["passed"]
    ]
    advisories = [
        name
        for name, gate in gates.items()
        if not gate["required"] and not gate["passed"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "performance_job_id": performance.get("job_id"),
        "performance_kind": kind,
        "requirements": {
            "pbr": require_pbr,
            "acting": require_acting,
            "body": require_body,
        },
        "status": "ready" if not failures else "blocked",
        "publishable": not failures,
        "failures": failures,
        "advisories": advisories,
        "passed_required_gate_count": sum(
            1 for gate in gates.values() if gate["required"] and gate["passed"]
        ),
        "required_gate_count": sum(1 for gate in gates.values() if gate["required"]),
        "gates": gates,
        "claim": (
            "All required release evidence is present."
            if not failures
            else "Reviewable output only; missing evidence blocks production publication."
        ),
    }


__all__ = ["SCHEMA_VERSION", "evaluate_production_readiness"]
