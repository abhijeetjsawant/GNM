from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.video_observation import analyze_rgb_frames
from autoanim_gnm.video_sequence_solve import (
    VIDEO_SEQUENCE_CANDIDATE_SCHEMA_VERSION,
    VIDEO_SEQUENCE_SUMMARY_SCHEMA_VERSION,
    _regularized_solve,
    build_shadow_video_sequence_candidate,
    load_verified_video_sequence_summary,
    load_video_sequence_candidate,
    write_video_sequence_candidate,
    write_video_sequence_summary,
)
from autoanim_gnm.visual_track import (
    MOTION_AUTHORITY,
    REGION_NAMES,
    REGION_PROVISIONAL_OBSERVED,
    REGION_UNKNOWN,
    build_visual_track,
    write_visual_track,
    write_visual_track_summary,
)
from autoanim_gnm.visual_track_calibration import (
    VISUAL_TRACK_CALIBRATION_SCHEMA_VERSION,
    build_calibration_evidence,
    build_unavailable_calibration_evidence,
    load_visual_track_calibration,
    write_visual_track_calibration,
)
from autoanim_gnm.visual_track_provider import (
    VISUAL_TRACK_PROVIDER_SCHEMA_VERSION,
    array_sha256,
    build_visual_track_provider_result,
    file_sha256,
    load_visual_track_provider_result,
    write_visual_track_provider_result,
)

from test_video_observation import _capture, _same_frames


def _provider_bundle(tmp_path: Path, count: int = 9):
    capture = _capture(tuple(True for _ in range(count)))
    observations = analyze_rgb_frames(capture, _same_frames(count))
    visual_track = build_visual_track(capture, observations)
    dense = write_visual_track(tmp_path / "visual-track.npz", visual_track)
    summary = write_visual_track_summary(tmp_path / "visual-track.json", visual_track)
    provider = build_visual_track_provider_result(
        visual_track,
        visual_track_v1_sha256=file_sha256(dense),
        visual_track_v1_summary_sha256=file_sha256(summary),
    )
    provider_path = write_visual_track_provider_result(
        tmp_path / "visual-track-provider.npz", provider
    )
    return capture, visual_track, provider, provider_path, dense, summary


def _unavailable_calibration(tmp_path: Path, provider, provider_path: Path):
    evidence = build_unavailable_calibration_evidence(
        provider, provider_result_sha256=file_sha256(provider_path)
    )
    path = write_visual_track_calibration(
        tmp_path / "visual-track-calibration.json", evidence
    )
    return evidence, path


def test_provider_result_binds_exact_pts_hashes_and_keeps_unknowns_unknown(
    tmp_path: Path,
) -> None:
    capture, visual_track, provider, provider_path, dense, summary = _provider_bundle(
        tmp_path
    )
    loaded = load_visual_track_provider_result(
        provider_path,
        expected_visual_track=visual_track,
        expected_visual_track_path=dense,
        expected_visual_track_summary_path=summary,
    )

    assert loaded.metadata["schema_version"] == VISUAL_TRACK_PROVIDER_SCHEMA_VERSION
    assert loaded.metadata["motion_authority"] == MOTION_AUTHORITY
    assert loaded.metadata["consumed_by_retargeting"] is False
    assert loaded.metadata["claims"] == {
        "changes_final_gnm_motion": False,
        "confidence_calibrated": False,
        "occlusion_validated": False,
        "identity_continuity_validated": False,
        "tongue_observed": False,
        "production_validated": False,
    }
    np.testing.assert_array_equal(loaded.source_pts, capture.source_pts)
    assert loaded.evidence_rgb_sha256 == visual_track.evidence_rgb_sha256
    assert loaded.metadata["bindings"]["source_pts_sha256"] == array_sha256(
        capture.source_pts
    )
    assert np.isnan(loaded.region_confidence).all()
    assert np.isnan(loaded.region_occlusion_probability).all()
    assert np.isnan(loaded.region_residual).all()
    head = REGION_NAMES.index("head")
    assert np.all(
        loaded.region_observation_state[:, head] == REGION_PROVISIONAL_OBSERVED
    )
    for name in set(REGION_NAMES) - {"head"}:
        index = REGION_NAMES.index(name)
        assert np.all(loaded.region_observation_state[:, index] == REGION_UNKNOWN)
        assert np.isnan(loaded.region_support_score[:, index]).all()

    summary.write_text("{}\n", encoding="utf-8")
    with pytest.raises(AutoAnimError, match="does not bind"):
        load_visual_track_provider_result(
            provider_path,
            expected_visual_track=visual_track,
            expected_visual_track_path=dense,
            expected_visual_track_summary_path=summary,
        )


def test_provider_loader_rejects_wrong_dtype_and_frame_hash_binding(
    tmp_path: Path,
) -> None:
    _, visual_track, provider, provider_path, _, _ = _provider_bundle(tmp_path)
    with np.load(provider_path, allow_pickle=False) as source:
        payload = {name: np.array(source[name], copy=True) for name in source.files}
    payload["source_pts"] = payload["source_pts"].astype(np.float64)
    wrong_dtype = tmp_path / "wrong-dtype.npz"
    np.savez_compressed(wrong_dtype, **payload)
    with pytest.raises(AutoAnimError, match="dtype"):
        load_visual_track_provider_result(wrong_dtype)

    wrong_hashes = replace(
        provider,
        evidence_rgb_sha256=("f" * 64,) + provider.evidence_rgb_sha256[1:],
    )
    wrong_hash_path = write_visual_track_provider_result(
        tmp_path / "wrong-hash.npz", wrong_hashes
    )
    with pytest.raises(AutoAnimError, match="frame hashes differ"):
        load_visual_track_provider_result(
            wrong_hash_path, expected_visual_track=visual_track
        )


def test_calibration_evidence_reports_ece_brier_and_occlusion_slices(
    tmp_path: Path,
) -> None:
    _, _, provider, provider_path, _, _ = _provider_bundle(tmp_path, count=40)
    shape = (provider.frame_count, len(REGION_NAMES))
    probability = np.full(shape, np.nan, dtype=np.float64)
    target = np.zeros(shape, dtype=np.bool_)
    available = np.zeros(shape, dtype=np.bool_)
    occluded = np.zeros(shape, dtype=np.bool_)
    lips = REGION_NAMES.index("lips_contact")
    frame = np.arange(provider.frame_count)
    target[:, lips] = (frame // 2) % 2 == 1
    occluded[:, lips] = frame % 2 == 0
    available[:, lips] = True
    probability[:, lips] = np.where(target[:, lips], 0.99, 0.01)
    observation_state = np.full(shape, REGION_UNKNOWN, dtype=np.uint8)
    observation_state[:, lips] = REGION_PROVISIONAL_OBSERVED
    provider = replace(
        provider,
        region_observation_state=observation_state,
        region_support_score=probability.astype(np.float32),
    )
    provider_path = write_visual_track_provider_result(provider_path, provider)
    bounds = {
        name: (f"{name}_normalized_error", 0.05, "normalized_face_scale")
        for name in REGION_NAMES
    }
    evidence = build_calibration_evidence(
        provider,
        provider_result_sha256=file_sha256(provider_path),
        fixture_sha256="c" * 64,
        labels_sha256="d" * 64,
        within_error_bound=target,
        label_available=available,
        occluded=occluded,
        error_bounds=bounds,
    )
    path = write_visual_track_calibration(tmp_path / "calibration.json", evidence)
    loaded = load_visual_track_calibration(
        path,
        expected_provider=provider,
        expected_provider_path=provider_path,
    )

    assert loaded.payload["schema_version"] == VISUAL_TRACK_CALIBRATION_SCHEMA_VERSION
    lips_metrics = loaded.payload["regions"]["lips_contact"]
    assert lips_metrics["overall"]["sample_count"] == 40
    assert lips_metrics["occluded"]["sample_count"] == 20
    assert lips_metrics["unoccluded"]["sample_count"] == 20
    assert lips_metrics["overall"]["ece"] == pytest.approx(0.01)
    assert lips_metrics["overall"]["brier"] == pytest.approx(0.0001)
    assert lips_metrics["gate_pass"] is True
    assert loaded.payload["summary"]["qualified_region_count"] == 1
    assert loaded.payload["claims"]["grants_motion_authority"] is False
    assert loaded.payload["claims"]["production_validated"] is False
    assert loaded.payload["regions"]["tongue"]["gate_pass"] is False

    overconfident_support = np.full(shape, np.nan, dtype=np.float32)
    overconfident_support[:, lips] = 0.99
    overconfident_provider = replace(
        provider, region_support_score=overconfident_support
    )
    overconfident_path = write_visual_track_provider_result(
        tmp_path / "overconfident-provider.npz", overconfident_provider
    )
    rejected = build_calibration_evidence(
        overconfident_provider,
        provider_result_sha256=file_sha256(overconfident_path),
        fixture_sha256="c" * 64,
        labels_sha256="d" * 64,
        within_error_bound=target,
        label_available=available,
        occluded=occluded,
        error_bounds=bounds,
    )
    rejected_lips = rejected.payload["regions"]["lips_contact"]
    assert rejected_lips["overall"]["ece"] > 0.45
    assert rejected_lips["overall"]["brier"] > 0.45
    assert rejected_lips["gate_pass"] is False
    assert "ece_above_limit" in rejected_lips["reasons"]


def test_unavailable_calibration_is_explicit_and_duplicate_json_fails_closed(
    tmp_path: Path,
) -> None:
    _, _, provider, provider_path, _, _ = _provider_bundle(tmp_path)
    evidence, path = _unavailable_calibration(tmp_path, provider, provider_path)
    assert evidence.payload["status"] == "unavailable"
    assert evidence.payload["summary"]["qualified_region_count"] == 0
    assert all(
        region["overall"]["ece"] is None
        and region["occluded"]["brier"] is None
        and region["unoccluded"]["sample_count"] == 0
        and region["gate_pass"] is False
        for region in evidence.payload["regions"].values()
    )

    duplicate = path.read_text(encoding="utf-8").replace(
        "{\n", '{\n  "schema_version": "duplicate",\n', 1
    )
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(AutoAnimError, match="Duplicate JSON member"):
        load_visual_track_calibration(path)


def test_bidirectional_candidate_is_motion_inert_and_respects_cut_boundaries(
    tmp_path: Path,
) -> None:
    _, _, provider, provider_path, _, _ = _provider_bundle(tmp_path)
    cut = np.zeros(provider.frame_count, dtype=np.bool_)
    cut[5] = True
    shot = np.zeros(provider.frame_count, dtype=np.int32)
    shot[5:] = 1
    tracking = np.zeros(provider.frame_count, dtype=np.int32)
    tracking[5:] = 1
    provider = replace(
        provider,
        cut_candidate=cut,
        shot_epoch_index=shot,
        tracking_epoch_index=tracking,
    )
    provider_path = write_visual_track_provider_result(provider_path, provider)
    calibration, calibration_path = _unavailable_calibration(
        tmp_path, provider, provider_path
    )
    expression = np.zeros((provider.frame_count, 383), dtype=np.float32)
    rotations = np.zeros((provider.frame_count, 4, 3), dtype=np.float32)
    translation = np.zeros((provider.frame_count, 3), dtype=np.float32)
    expression[4, 0] = 1.0
    anchors = np.zeros(provider.frame_count, dtype=np.bool_)
    anchors[4] = True
    source_hashes = (
        array_sha256(expression),
        array_sha256(rotations),
        array_sha256(translation),
    )
    candidate = build_shadow_video_sequence_candidate(
        provider,
        calibration,
        provider_result_sha256=file_sha256(provider_path),
        calibration_evidence_sha256=file_sha256(calibration_path),
        expression=expression,
        rotations=rotations,
        translation=translation,
        hard_anchor_mask=anchors,
    )

    assert candidate.metadata["schema_version"] == VIDEO_SEQUENCE_CANDIDATE_SCHEMA_VERSION
    assert candidate.metadata["motion_authority"] == MOTION_AUTHORITY
    assert candidate.metadata["claims"] == {
        "changes_final_gnm_motion": False,
        "candidate_is_shipped": False,
        "final_output_arrays_byte_identical": True,
        "grants_motion_authority": False,
        "production_validated": False,
    }
    assert candidate.expression[4, 0] == 1.0
    assert candidate.expression[3, 0] > 0.0
    assert np.all(candidate.expression[5:, 0] == 0.0)
    assert source_hashes == (
        array_sha256(expression),
        array_sha256(rotations),
        array_sha256(translation),
    )
    assert candidate.metadata["bindings"]["baseline_expression_sha256"] == source_hashes[0]
    assert candidate.metadata["bindings"]["shipped_expression_sha256"] == source_hashes[0]

    candidate_path = write_video_sequence_candidate(
        tmp_path / "candidate.npz", candidate
    )
    with np.load(candidate_path, allow_pickle=False) as source:
        tampered_arrays = {
            name: np.array(source[name], copy=True) for name in source.files
        }
    tampered_arrays["expression"][0, 0] = 0.25
    tampered_candidate_path = tmp_path / "tampered-candidate.npz"
    np.savez_compressed(tampered_candidate_path, **tampered_arrays)
    with pytest.raises(AutoAnimError, match="hashes do not reconstruct"):
        load_video_sequence_candidate(tampered_candidate_path)

    summary_path = write_video_sequence_summary(
        tmp_path / "candidate.json",
        candidate,
        candidate_sha256=file_sha256(candidate_path),
    )
    loaded = load_video_sequence_candidate(candidate_path)
    loaded.validate_inputs(
        provider,
        calibration,
        provider_result_sha256=file_sha256(provider_path),
        calibration_evidence_sha256=file_sha256(calibration_path),
        expression=expression,
        rotations=rotations,
        translation=translation,
    )
    summary = load_verified_video_sequence_summary(
        summary_path, candidate_path=candidate_path
    )
    assert summary["schemaVersion"] == VIDEO_SEQUENCE_SUMMARY_SCHEMA_VERSION
    assert summary["baselineHashes"] == summary["shippedHashes"]
    assert summary["candidateComparison"]["expressionDiffersFromBaseline"] is True


def test_bidirectional_solver_uses_exact_irregular_pts_intervals() -> None:
    values = np.zeros((5, 1), dtype=np.float64)
    values[3, 0] = 1.0
    uniform = _regularized_solve(
        values, 0.35, np.asarray((0, 1, 2, 3, 4), dtype=np.int64)
    )
    irregular = _regularized_solve(
        values, 0.35, np.asarray((0, 1, 2, 20, 21), dtype=np.int64)
    )
    assert uniform[2, 0] > 0.0
    assert irregular[2, 0] < uniform[2, 0] * 0.05
    assert irregular[4, 0] > 0.0


def test_sequence_rejects_stale_calibration_and_tampered_summary(tmp_path: Path) -> None:
    _, _, provider, provider_path, _, _ = _provider_bundle(tmp_path)
    calibration, calibration_path = _unavailable_calibration(
        tmp_path, provider, provider_path
    )
    expression = np.zeros((provider.frame_count, 383), dtype=np.float32)
    rotations = np.zeros((provider.frame_count, 4, 3), dtype=np.float32)
    translation = np.zeros((provider.frame_count, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="calibration does not bind"):
        build_shadow_video_sequence_candidate(
            provider,
            calibration,
            provider_result_sha256="e" * 64,
            calibration_evidence_sha256=file_sha256(calibration_path),
            expression=expression,
            rotations=rotations,
            translation=translation,
        )

    candidate = build_shadow_video_sequence_candidate(
        provider,
        calibration,
        provider_result_sha256=file_sha256(provider_path),
        calibration_evidence_sha256=file_sha256(calibration_path),
        expression=expression,
        rotations=rotations,
        translation=translation,
    )
    candidate_path = write_video_sequence_candidate(
        tmp_path / "candidate.npz", candidate
    )
    summary_path = write_video_sequence_summary(
        tmp_path / "candidate.json",
        candidate,
        candidate_sha256=file_sha256(candidate_path),
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["claims"]["production_validated"] = True
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AutoAnimError, match="does not reconstruct"):
        load_verified_video_sequence_summary(
            summary_path, candidate_path=candidate_path
        )
