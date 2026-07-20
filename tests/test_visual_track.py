from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
import warnings
import zipfile

import numpy as np
import pytest

import autoanim_gnm.visual_track as visual_track_module
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.video_capture import VideoCaptureRun
from autoanim_gnm.video_observation import analyze_rgb_frames
from autoanim_gnm.visual_track import (
    MOTION_AUTHORITY,
    REGION_NAMES,
    REGION_PROVISIONAL_OBSERVED,
    REGION_UNSUPPORTED_REASON_BIT,
    REGION_UNKNOWN,
    SUBJECT_OBSERVED_UNBOUND,
    SUBJECT_SELECTED_UNBOUND,
    TONGUE_UNOBSERVED_REASON_BIT,
    VISUAL_TRACK_POLICY,
    VISUAL_TRACK_SCHEMA_VERSION,
    VISUAL_TRACK_SUMMARY_SCHEMA_VERSION,
    build_visual_track,
    build_visual_track_summary,
    load_verified_visual_track_summary,
    load_visual_track,
    write_visual_track,
    write_visual_track_summary,
)

from test_video_observation import _capture, _same_frames, _texture


def _track(detected: tuple[bool, ...] = (True, True, False, True)):
    capture = _capture(detected)
    observations = analyze_rgb_frames(capture, _same_frames(capture.frame_count))
    return capture, observations, build_visual_track(capture, observations)


def _capture_run(capture, observations, *, thresholds=(0.41, 0.52, 0.63)):
    return VideoCaptureRun(
        track=capture,
        detector_ingress_rgb_sha256=observations.decoded_pixel_sha256,
        num_faces=1,
        confidence_thresholds=thresholds,
    )


def _assert_tracks_equal(left, right) -> None:
    for field in fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, np.ndarray):
            np.testing.assert_array_equal(left_value, right_value)
        else:
            assert left_value == right_value


def _rewrite_npz(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as values:
        arrays = {name: values[name] for name in values.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)


def _canonical(value: dict) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def test_visual_track_v1_is_exact_shadow_evidence_with_no_motion_authority() -> None:
    capture, observations, track = _track()
    metadata = track.metadata

    assert metadata["schema_version"] == VISUAL_TRACK_SCHEMA_VERSION
    assert metadata["policy"] == VISUAL_TRACK_POLICY
    assert metadata["motion_authority"] == MOTION_AUTHORITY == "none"
    assert metadata["consumed_by_retargeting"] is False
    assert metadata["claims"] == {
        "changes_final_gnm_motion": False,
        "confidence_calibrated": False,
        "covariance_available": False,
        "occlusion_validated": False,
        "identity_continuity_validated": False,
        "tongue_observed": False,
        "production_validated": False,
    }
    assert metadata["provider"]["detector_ingress_pixels_retained"] is False
    assert metadata["provider"]["evidence_pixels_relationship"] == (
        "redecoded_for_evidence"
    )
    assert metadata["provider"]["configuration"]["confidence_thresholds"] is None
    assert metadata["identity"]["subject_binding_state"] == "unbound"
    assert metadata["identity"]["identity_embedding_state"] == "not_computed"
    assert metadata["identity"]["subject_state_value_1_semantics"] == (
        "observed_unbound_not_identity_selected"
    )
    assert metadata["identity"]["character_revision_ref"] is None
    assert SUBJECT_SELECTED_UNBOUND == SUBJECT_OBSERVED_UNBOUND

    np.testing.assert_array_equal(track.source_pts, capture.source_pts)
    assert track.evidence_rgb_sha256 == observations.decoded_pixel_sha256
    assert track.shot_epoch_index.tolist() == [0, 0, 0, 0]
    assert track.tracking_epoch_index.tolist() == [0, 0, -1, 1]
    assert track.subject_epoch_index.tolist() == [-1, -1, -1, -1]
    assert track.subject_state.tolist() == [
        int(SUBJECT_SELECTED_UNBOUND),
        int(SUBJECT_SELECTED_UNBOUND),
        0,
        int(SUBJECT_SELECTED_UNBOUND),
    ]
    np.testing.assert_array_equal(track.point_xyz_normalized, capture.landmarks_xyz)
    expected_pixels = capture.landmarks_xyz[:, :, :2] * np.asarray(
        [capture.width, capture.height], dtype=np.float32
    )
    np.testing.assert_array_equal(track.point_xy_source_pixels, expected_pixels)


def test_unsupported_point_and_tongue_evidence_stays_nan_unknown() -> None:
    capture, _, track = _track()
    assert track.region_names == (
        "lips_contact",
        "jaw",
        "left_eyelid",
        "right_eyelid",
        "gaze",
        "brows",
        "cheeks_nose",
        "silhouette",
        "head",
        "tongue",
    )
    detected_points = np.broadcast_to(
        capture.detected[:, None], track.point_measurement_state.shape
    )
    assert np.isnan(track.point_covariance_xyz_packed).all()
    assert np.isnan(track.point_reprojection_residual_px).all()
    assert np.isnan(track.point_occlusion_probability).all()
    assert np.all(track.point_covariance_state == 0)
    assert np.all(track.point_occlusion_state[detected_points] == 1)
    assert np.all(track.point_occlusion_state[~detected_points] == 0)
    assert np.isnan(track.point_xyz_normalized[~detected_points]).all()

    tongue = REGION_NAMES.index("tongue")
    assert np.all(
        track.region_observation_state[capture.detected, tongue] == REGION_UNKNOWN
    )
    assert np.isnan(track.region_support_score[:, tongue]).all()
    assert np.all(
        track.region_reason_mask[capture.detected, tongue]
        == TONGUE_UNOBSERVED_REASON_BIT
    )
    assert np.isnan(track.region_calibrated_confidence).all()
    assert np.all(track.region_confidence_state == 0)
    finite_support = track.region_support_score[np.isfinite(track.region_support_score)]
    assert np.max(finite_support) < 0.75
    head = REGION_NAMES.index("head")
    assert track.region_observation_state[0, head] == REGION_PROVISIONAL_OBSERVED
    for name in (
        "lips_contact",
        "jaw",
        "left_eyelid",
        "right_eyelid",
        "gaze",
        "brows",
        "cheeks_nose",
        "silhouette",
    ):
        index = REGION_NAMES.index(name)
        assert np.all(track.region_observation_state[capture.detected, index] == REGION_UNKNOWN)
        assert np.isnan(track.region_support_score[:, index]).all()
        assert np.all(
            track.region_reason_mask[capture.detected, index]
            == REGION_UNSUPPORTED_REASON_BIT
        )


def test_shot_and_tracking_epochs_are_provisional_and_do_not_cross_a_cut() -> None:
    capture = _capture((True, True))
    base = _texture()
    dark = ((base.astype(np.uint16) // 4) + 8).astype(np.uint8)
    inverted = 255 - dark
    observations = analyze_rgb_frames(capture, [dark, inverted])
    assert observations.cut_candidate.tolist() == [False, True]

    track = build_visual_track(capture, observations)

    assert track.shot_epoch_index.tolist() == [0, 1]
    assert track.tracking_epoch_index.tolist() == [0, 1]
    summary = build_visual_track_summary(track)
    assert summary["epochs"]["shotAuthority"] == (
        "provisional_pixel_cut_candidate"
    )
    assert summary["epochs"]["trackingAuthority"].startswith(
        "autoanim_observation_continuity"
    )
    assert summary["epochs"]["shots"] == [
        {
            "epochIndex": 0,
            "startFrame": 0,
            "endFrameExclusive": 1,
            "startSourcePTS": 100,
            "endSourcePTSInclusive": 100,
        },
        {
            "epochIndex": 1,
            "startFrame": 1,
            "endFrameExclusive": 2,
            "startSourcePTS": 101,
            "endSourcePTSInclusive": 101,
        },
    ]
    assert summary["epochs"]["subjects"] == []
    assert summary["epochs"]["subjectAuthority"] == "none"


def test_dense_artifact_and_summary_are_deterministic_and_reconstructable(
    tmp_path: Path,
) -> None:
    capture, observations, track = _track()
    first = write_visual_track(tmp_path / "first.npz", track)
    second = write_visual_track(tmp_path / "second.npz", track)
    assert first.read_bytes() == second.read_bytes()
    loaded = load_visual_track(first)
    _assert_tracks_equal(loaded, track)
    loaded.validate_inputs(capture, observations)

    first_summary = write_visual_track_summary(tmp_path / "first.json", loaded)
    second_summary = write_visual_track_summary(tmp_path / "second.json", track)
    assert first_summary.read_bytes() == second_summary.read_bytes()
    payload = load_verified_visual_track_summary(
        first_summary,
        visual_track_path=first,
        expected_capture=capture,
        expected_observations=observations,
    )
    assert payload["schemaVersion"] == VISUAL_TRACK_SUMMARY_SCHEMA_VERSION
    assert payload["motionAuthority"] == "none"
    assert payload["consumedByRetargeting"] is False
    assert payload["identity"]["subject_binding_state"] == "unbound"
    assert "NaN" not in first_summary.read_text(encoding="utf-8")
    assert str(tmp_path) not in first_summary.read_text(encoding="utf-8")


def test_optional_capture_run_binds_exact_configuration_and_same_buffers(
    tmp_path: Path,
) -> None:
    capture, observations, legacy = _track()
    run = _capture_run(capture, observations)

    track = build_visual_track(capture, observations, capture_run=run)
    provider = track.metadata["provider"]
    assert provider["detector_ingress_pixels_retained"] is False
    assert provider["detector_ingress_hashes_retained"] is True
    assert provider["evidence_pixels_relationship"] == (
        "per_frame_sha256_equal_to_detector_ingress"
    )
    assert provider["configuration"] == run.detector_configuration()
    assert track.evidence_rgb_sha256 == run.detector_ingress_rgb_sha256
    assert track.metadata["motion_authority"] == "none"
    assert track.metadata["consumed_by_retargeting"] is False
    assert track.metadata["claims"]["changes_final_gnm_motion"] is False

    legacy_provider = legacy.metadata["provider"]
    assert legacy_provider["detector_ingress_hashes_retained"] is False
    assert legacy_provider["evidence_pixels_relationship"] == "redecoded_for_evidence"
    assert legacy_provider["configuration"]["confidence_thresholds"] is None

    dense = write_visual_track(tmp_path / "same-buffer.npz", track)
    summary = write_visual_track_summary(tmp_path / "same-buffer.json", track)
    loaded = load_visual_track(dense)
    loaded.validate_inputs(
        capture, observations, expected_capture_run=run
    )
    verified = load_verified_visual_track_summary(
        summary,
        visual_track_path=dense,
        expected_capture=capture,
        expected_observations=observations,
        expected_capture_run=run,
    )
    assert verified["provider"]["configuration"] == run.detector_configuration()
    with pytest.raises(ValueError, match="does not reconstruct"):
        loaded.validate_inputs(capture, observations)


def test_optional_capture_run_rejects_track_and_ingress_hash_mismatch() -> None:
    capture, observations, _ = _track()
    different_capture = _capture((True, False, False, True))
    wrong_track_run = VideoCaptureRun(
        track=different_capture,
        detector_ingress_rgb_sha256=observations.decoded_pixel_sha256,
        num_faces=1,
        confidence_thresholds=(0.5, 0.5, 0.5),
    )
    with pytest.raises(ValueError, match="does not exactly match"):
        build_visual_track(capture, observations, capture_run=wrong_track_run)

    hashes = list(observations.decoded_pixel_sha256)
    hashes[0] = "c" * 64
    wrong_hash_run = VideoCaptureRun(
        track=capture,
        detector_ingress_rgb_sha256=tuple(hashes),
        num_faces=1,
        confidence_thresholds=(0.5, 0.5, 0.5),
    )
    with pytest.raises(ValueError, match="Detector-ingress hashes"):
        build_visual_track(capture, observations, capture_run=wrong_hash_run)


@pytest.mark.parametrize(
    ("field", "value"),
    (("min_tracking_confidence", 1.5), ("num_faces", True)),
)
def test_same_buffer_configuration_tampering_fails_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    capture, observations, _ = _track()
    track = build_visual_track(
        capture, observations, capture_run=_capture_run(capture, observations)
    )
    path = write_visual_track(tmp_path / "same-buffer.npz", track)

    def change(arrays: dict[str, np.ndarray]) -> None:
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["provider"]["configuration"][field] = value
        arrays["metadata_json"] = np.asarray(_canonical(metadata))

    _rewrite_npz(path, change)
    with pytest.raises(AutoAnimError, match="same-buffer detector configuration"):
        load_visual_track(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata: metadata.update(motion_authority="visual"),
            "fail-closed",
        ),
        (
            lambda metadata: metadata.update(consumed_by_retargeting=True),
            "fail-closed",
        ),
        (
            lambda metadata: metadata["identity"].update(subject_id="person-1"),
            "fail-closed",
        ),
        (
            lambda metadata: metadata["identity"].update(
                identity_embedding_state="computed"
            ),
            "fail-closed",
        ),
        (
            lambda metadata: metadata["claims"].update(
                confidence_calibrated=True
            ),
            "fail-closed",
        ),
        (
            lambda metadata: metadata["source"].update(frame_count=True),
            "fail-closed",
        ),
        (
            lambda metadata: metadata["source"].update(source_start_pts=True),
            "fail-closed",
        ),
        (
            lambda metadata: metadata["provider"]["configuration"].update(
                num_faces=True
            ),
            "legacy detector configuration",
        ),
        (
            lambda metadata: metadata["source"].update(
                name="/private/tmp/source.mov"
            ),
            "basenames",
        ),
    ],
)
def test_metadata_claim_and_identity_tampering_fails_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    _, _, track = _track()
    path = write_visual_track(tmp_path / "track.npz", track)

    def change(arrays: dict[str, np.ndarray]) -> None:
        metadata = json.loads(str(arrays["metadata_json"].item()))
        mutation(metadata)
        arrays["metadata_json"] = np.asarray(_canonical(metadata))

    _rewrite_npz(path, change)
    with pytest.raises(AutoAnimError, match=message):
        load_visual_track(path)


@pytest.mark.parametrize(
    ("array_name", "mutation", "message"),
    [
        (
            "source_pts",
            lambda value: value.__setitem__(1, value[0]),
            "fail-closed|PTS|source_pts_sha256",
        ),
        (
            "tracking_epoch_index",
            lambda value: value.__setitem__(3, 0),
            "epochs",
        ),
        (
            "subject_epoch_index",
            lambda value: value.__setitem__(0, 0),
            "unbound",
        ),
        (
            "point_covariance_xyz_packed",
            lambda value: value.__setitem__((0, 0, 0), 0.0),
            "unsupported point evidence",
        ),
        (
            "point_occlusion_probability",
            lambda value: value.__setitem__((0, 0), 0.0),
            "unsupported point evidence",
        ),
        (
            "region_calibrated_confidence",
            lambda value: value.__setitem__((0, 0), 0.5),
            "calibrated regional confidence",
        ),
        (
            "region_support_score",
            lambda value: value.__setitem__((0, REGION_NAMES.index("tongue")), 0.0),
            "tongue evidence",
        ),
    ],
)
def test_dense_array_tampering_fails_closed(
    tmp_path: Path, array_name: str, mutation, message: str
) -> None:
    _, _, track = _track()
    path = write_visual_track(tmp_path / "track.npz", track)

    def change(arrays: dict[str, np.ndarray]) -> None:
        value = arrays[array_name].copy()
        mutation(value)
        arrays[array_name] = value

    _rewrite_npz(path, change)
    with pytest.raises(AutoAnimError, match=message):
        load_visual_track(path)


def test_summary_and_source_evidence_tampering_fails_reconstruction(
    tmp_path: Path,
) -> None:
    capture, observations, track = _track()
    dense = write_visual_track(tmp_path / "track.npz", track)
    summary = write_visual_track_summary(tmp_path / "track.json", track)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["epochs"]["subjects"] = [{"epochIndex": 0}]
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AutoAnimError, match="does not reconstruct"):
        load_verified_visual_track_summary(
            summary,
            visual_track_path=dense,
            expected_capture=capture,
            expected_observations=observations,
        )

    summary = write_visual_track_summary(summary, track)
    different_capture = _capture((True, False, False, True))
    different_observations = analyze_rgb_frames(
        different_capture, _same_frames(different_capture.frame_count)
    )
    with pytest.raises((AutoAnimError, ValueError), match="reconstruct|bound|differs"):
        load_verified_visual_track_summary(
            summary,
            visual_track_path=dense,
            expected_capture=different_capture,
            expected_observations=different_observations,
        )


def test_loader_rejects_duplicate_members_wrong_dtype_and_resource_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, track = _track()
    path = write_visual_track(tmp_path / "track.npz", track)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, mode="a") as archive:
            archive.writestr("source_pts.npy", b"ambiguous")
    with pytest.raises(AutoAnimError, match="duplicate members"):
        load_visual_track(path)

    path = write_visual_track(path, track)
    _rewrite_npz(
        path,
        lambda arrays: arrays.update(
            source_pts=arrays["source_pts"].astype(np.int32)
        ),
    )
    with pytest.raises(AutoAnimError, match="dtype"):
        load_visual_track(path)

    path = write_visual_track(path, track)
    monkeypatch.setattr(
        visual_track_module, "MAX_VISUAL_TRACK_UNCOMPRESSED_BYTES", 1
    )
    with pytest.raises(AutoAnimError, match="resource limit"):
        load_visual_track(path)


def test_noncanonical_or_duplicate_metadata_json_is_rejected(tmp_path: Path) -> None:
    _, _, track = _track()
    path = write_visual_track(tmp_path / "track.npz", track)
    _rewrite_npz(
        path,
        lambda arrays: arrays.update(
            metadata_json=np.asarray(
                '{"kind":"visual_track","kind":"forged"}'
            )
        ),
    )
    with pytest.raises(AutoAnimError, match="Duplicate JSON member"):
        load_visual_track(path)
