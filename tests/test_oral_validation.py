from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import autoanim_gnm.oral_validation as oral_validation_module
from autoanim_gnm.animated_gltf import export_animated_gnm_glb
from autoanim_gnm.animation import calibrate_lip_contact
from autoanim_gnm.gltf_export import export_gnm_glb
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.oral_validation import (
    OralValidationError,
    OralValidationThresholds,
    validate_controls_npz,
    validate_glb_oral_geometry,
    validate_oral_frames,
)
from autoanim_gnm.rig import ControlRig
from autoanim_gnm.semantic_decoder import ExpressionDecoder
from autoanim_gnm.serialization import write_npz


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adapter() -> GNMAdapter:
    return GNMAdapter()


@pytest.fixture(scope="module")
def rig(adapter: GNMAdapter) -> ControlRig:
    return ControlRig(
        adapter,
        ExpressionDecoder("gnm/shape/data/semantic_sampler/expression_decoder_model.h5"),
    )


def _controls(
    path: Path,
    expression: np.ndarray,
    *,
    target: np.ndarray | None = None,
    attained: np.ndarray | None = None,
    rotations: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> Path:
    frame_count = len(expression)
    payload: dict[str, np.ndarray] = {
        "expression": np.asarray(expression, dtype=np.float32),
        "rotations": (
            np.zeros((frame_count, 4, 3), dtype=np.float32)
            if rotations is None
            else np.asarray(rotations, dtype=np.float32)
        ),
        "translation": (
            np.zeros((frame_count, 3), dtype=np.float32)
            if translation is None
            else np.asarray(translation, dtype=np.float32)
        ),
        "timestamps": np.arange(frame_count, dtype=np.float32) / 30.0,
    }
    if target is not None:
        payload["lip_contact_target_gap"] = np.asarray(target, dtype=np.float32)
    if attained is not None:
        payload["lip_contact_attained"] = np.asarray(attained, dtype=bool)
    return write_npz(path, **payload)


def test_neutral_oral_inventory_is_complete_deterministic_and_truthful(
    adapter: GNMAdapter,
) -> None:
    frame = adapter.mesh()[None]
    first = validate_oral_frames(frame, adapter=adapter)
    second = validate_oral_frames(frame, adapter=adapter)

    assert first.as_dict() == second.as_dict()
    json.dumps(first.as_dict(), allow_nan=False)
    inventory = first.report["geometry_inventory"]
    assert inventory["vertex_groups"] == {
        "tongue": 933,
        "upper_teeth": 934,
        "lower_teeth": 934,
        "upper_lip": 145,
        "lower_lip": 145,
    }
    assert all(
        component["watertight"] is False
        for component in inventory["components"].values()
    )
    assert first.report["lip_contact"]["contact_frames"] == 0
    claims = first.report["claims"]
    assert claims["structural_geometry_measured"] is True
    assert claims["exact_surface_intersection_validated"] is False
    assert claims["penetration_free_validated"] is False
    assert claims["phoneme_correctness_validated"] is False
    assert claims["perceptual_correctness_validated"] is False
    assert claims["production_validated"] is False
    assert first.lip_gap_interocular.flags.writeable is False


def test_rigid_pose_and_translation_do_not_create_false_tongue_motion(
    adapter: GNMAdapter,
) -> None:
    rotations = np.zeros((2, 4, 3), dtype=np.float32)
    rotations[1, 0] = (0.10, 0.20, 0.05)
    translation = np.zeros((2, 3), dtype=np.float32)
    translation[1] = (0.10, -0.20, 0.30)
    frames = adapter.mesh(
        expression=np.zeros((2, 383), dtype=np.float32),
        rotations=rotations,
        translation=translation,
    )
    measured = validate_oral_frames(frames, adapter=adapter)

    assert measured.report["tongue_motion"]["frame_max_m"]["maximum"] < 1.0e-6
    assert measured.report["lip_contact"]["contact_frames"] == 0
    assert measured.report["tongue_teeth"]["collision_risk_frames"] == 0


def test_geometry_measures_tongue_motion_and_contact_without_claiming_phonemes(
    adapter: GNMAdapter, rig: ControlRig
) -> None:
    contact = calibrate_lip_contact(rig)
    expression = np.stack(
        (
            np.zeros(adapter.expression_dim, dtype=np.float32),
            rig.viseme("H"),
            np.float32(contact.maximum_alpha) * contact.direction,
        )
    )
    target = np.asarray((0.0, 0.0, contact.seal_gap_interocular), dtype=np.float32)
    declared = np.asarray((False, False, True), dtype=bool)
    result = validate_oral_frames(
        adapter.mesh(expression=expression),
        adapter=adapter,
        timestamps=np.asarray((0.0, 0.1, 0.2)),
        expected_lip_contact_target=target,
        declared_lip_contact_attained=declared,
    )

    np.testing.assert_array_equal(result.lip_contact_frames, (False, False, True))
    target_report = result.report["lip_contact"]["target_evidence"]
    assert target_report["candidate_frames"] == 1
    assert target_report["geometry_attained_frames"] == 1
    assert target_report["declared_geometry_disagreement_frames"] == 0
    assert target_report["phoneme_ground_truth"] is False
    assert result.report["tongue_motion"]["frame_max_m"]["maximum"] > 0.001
    assert result.report["claims"]["phoneme_correctness_validated"] is False


def test_missing_or_invalid_geometry_fails_closed(adapter: GNMAdapter, tmp_path: Path) -> None:
    with pytest.raises(OralValidationError) as absent:
        validate_oral_frames(np.zeros((1, 10, 3), dtype=np.float32), adapter=adapter)
    assert absent.value.code == "GEOMETRY_ABSENT"

    bad = adapter.mesh()[None]
    bad[0, 0, 0] = np.nan
    with pytest.raises(OralValidationError) as nonfinite:
        validate_oral_frames(bad, adapter=adapter)
    assert nonfinite.value.code == "INVALID_GEOMETRY"

    with pytest.raises(OralValidationError) as thresholds:
        OralValidationThresholds(
            tongue_teeth_near_contact_interocular=0.001,
            tongue_teeth_collision_risk_interocular=0.001,
        )
    assert thresholds.value.code == "INVALID_THRESHOLDS"

    with pytest.raises(OralValidationError) as controls:
        validate_controls_npz(tmp_path / "missing.npz", adapter=adapter)
    assert controls.value.code == "GEOMETRY_ABSENT"


def test_controls_validator_audits_geometry_transfer_not_speech_correctness(
    tmp_path: Path, adapter: GNMAdapter, rig: ControlRig
) -> None:
    contact = calibrate_lip_contact(rig)
    expression = np.stack(
        (
            np.zeros(383, dtype=np.float32),
            rig.viseme("H"),
            np.float32(contact.maximum_alpha) * contact.direction,
        )
    )
    controls = _controls(
        tmp_path / "controls.npz",
        expression,
        target=np.asarray((0.0, 0.0, contact.seal_gap_interocular)),
        attained=np.asarray((False, False, True)),
    )
    result = validate_controls_npz(controls, adapter=adapter, batch_size=2)

    assert result.report["source"]["artifact_sha256"] == hashlib.sha256(
        controls.read_bytes()
    ).hexdigest()
    evidence = result.report["control_evidence"]
    assert evidence["tongue_coefficient_peak"] > 0.0
    assert evidence["tongue_control_active_frames"] >= 1
    assert evidence["isolated_tongue_geometry_active_frames"] >= 1
    assert evidence["active_control_without_geometry_frames"] == 0
    assert result.report["claims"]["perceptual_correctness_validated"] is False


def test_controls_validator_streams_batches_and_can_reuse_complete_frames(
    tmp_path: Path,
    adapter: GNMAdapter,
    rig: ControlRig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expression = np.stack(
        [np.float32(value) * rig.viseme("H") for value in np.linspace(0.0, 0.3, 7)]
    )
    controls = _controls(tmp_path / "streamed-controls.npz", expression)
    original_mesh = adapter.mesh
    evaluated_frames = original_mesh(expression=expression)
    batch_sizes: list[int] = []

    def tracked_mesh(*args, **kwargs):
        values = np.asarray(kwargs["expression"])
        batch_sizes.append(len(values))
        return original_mesh(*args, **kwargs)

    monkeypatch.setattr(adapter, "mesh", tracked_mesh)
    streamed = validate_controls_npz(controls, adapter=adapter, batch_size=3)
    assert batch_sizes == [3, 3, 1]
    assert streamed.report["source"]["evaluation_mode"] == "streamed_controls"
    assert streamed.report["source"]["all_frames_evaluated"] is True
    assert len(streamed.lip_gap_interocular) == len(expression)

    batch_sizes.clear()
    reused = validate_controls_npz(
        controls,
        adapter=adapter,
        evaluated_frames=evaluated_frames,
        batch_size=2,
    )
    assert batch_sizes == []
    assert reused.report["source"]["evaluation_mode"] == "provided_complete_gnm_frames"
    np.testing.assert_allclose(
        reused.lip_gap_interocular,
        streamed.lip_gap_interocular,
        rtol=0.0,
        atol=1.0e-7,
    )


def test_long_control_track_never_exceeds_the_requested_geometry_batch(
    tmp_path: Path,
    adapter: GNMAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_count = 1_801
    batch_size = 37
    expression = np.zeros((frame_count, adapter.expression_dim), dtype=np.float32)
    controls = _controls(tmp_path / "long-controls.npz", expression)
    neutral = adapter.mesh()
    observed_batch_sizes: list[int] = []

    def bounded_mesh(*args, **kwargs):
        count = len(np.asarray(kwargs["expression"]))
        observed_batch_sizes.append(count)
        return np.broadcast_to(
            neutral,
            (count, adapter.model.num_vertices, 3),
        )

    def constant_nearest(frames, source_indices, target_indices, interocular):
        del source_indices, target_indices, interocular
        return (
            np.full(len(frames), 0.02, dtype=np.float64),
            np.full(len(frames), 0.025, dtype=np.float64),
        )

    monkeypatch.setattr(adapter, "mesh", bounded_mesh)
    monkeypatch.setattr(oral_validation_module, "_nearest_ratios", constant_nearest)
    result = validate_controls_npz(
        controls,
        adapter=adapter,
        batch_size=batch_size,
    )

    assert result.report["source"]["frame_count"] == frame_count
    assert result.report["source"]["all_frames_evaluated"] is True
    assert max(observed_batch_sizes) == batch_size
    assert sum(observed_batch_sizes) == frame_count
    assert len(result.lip_gap_interocular) == frame_count


def test_video_npz_aliases_preserve_contact_target_evidence(
    tmp_path: Path, adapter: GNMAdapter, rig: ControlRig
) -> None:
    contact = calibrate_lip_contact(rig)
    expression = np.stack(
        (
            np.zeros(adapter.expression_dim, dtype=np.float32),
            np.float32(contact.maximum_alpha) * contact.direction,
        )
    )
    controls = write_npz(
        tmp_path / "performance.npz",
        identity=np.zeros(adapter.identity_dim, dtype=np.float32),
        expression=expression,
        rotations=np.zeros((2, 4, 3), dtype=np.float32),
        translation=np.zeros((2, 3), dtype=np.float32),
        timestamps_seconds=np.asarray((0.0, 1.0 / 30.0), dtype=np.float32),
        lip_contact_target_gap_interocular=np.asarray(
            (0.0, contact.seal_gap_interocular), dtype=np.float32
        ),
        lip_contact_attained=np.asarray((False, True), dtype=bool),
    )

    result = validate_controls_npz(controls, adapter=adapter, batch_size=1)
    target = result.report["lip_contact"]["target_evidence"]
    assert target["candidate_frames"] == 1
    assert target["geometry_attained_frames"] == 1
    assert target["declared_geometry_disagreement_frames"] == 0


def test_animated_glb_oral_reconstruction_matches_source_controls(
    tmp_path: Path, adapter: GNMAdapter, rig: ControlRig
) -> None:
    expression = np.stack(
        (
            np.zeros(383, dtype=np.float32),
            np.float32(0.2) * rig.viseme("H"),
            np.zeros(383, dtype=np.float32),
        )
    )
    rotations = np.zeros((3, 4, 3), dtype=np.float32)
    rotations[1, 0] = (0.01, -0.02, 0.0)
    translation = np.zeros((3, 3), dtype=np.float32)
    timestamps = np.arange(3, dtype=np.float32) / 30.0
    frames = adapter.mesh(
        expression=expression, rotations=rotations, translation=translation
    )
    glb = tmp_path / "oral.glb"
    mapping = tmp_path / "oral-mapping.npz"
    export_animated_gnm_glb(
        glb, adapter, frames, timestamps, mapping_path=mapping
    )
    controls = _controls(
        tmp_path / "controls.npz",
        expression,
        rotations=rotations,
        translation=translation,
    )

    result = validate_glb_oral_geometry(
        glb,
        mapping,
        adapter=adapter,
        reference_controls_path=controls,
        batch_size=2,
    )
    reconstruction = result.report["structural_reconstruction"]
    assert reconstruction["status"] == "passed"
    assert reconstruction["validated"] is True
    assert reconstruction["oral_error_p95_mm"] <= 0.1
    assert reconstruction["oral_error_max_mm"] <= 0.5
    assert reconstruction["tongue_error_max_mm"] <= 0.5
    assert result.report["claims"]["structural_reconstruction_validated"] is True
    assert result.report["claims"]["phoneme_correctness_validated"] is False
    assert result.report["claims"]["production_validated"] is False

    wrong_controls = _controls(
        tmp_path / "wrong-controls.npz",
        np.zeros_like(expression),
        rotations=rotations,
        translation=translation,
    )
    failed = validate_glb_oral_geometry(
        glb,
        mapping,
        adapter=adapter,
        reference_controls_path=wrong_controls,
        batch_size=2,
    )
    assert failed.report["structural_reconstruction"]["status"] == "failed"
    assert failed.report["claims"]["structural_reconstruction_validated"] is False
    assert failed.report["claims"]["production_validated"] is False


def test_rank_zero_full_track_glb_can_be_structurally_validated(
    tmp_path: Path, adapter: GNMAdapter
) -> None:
    expression = np.zeros((4, adapter.expression_dim), dtype=np.float32)
    rotations = np.zeros((4, 4, 3), dtype=np.float32)
    translation = np.zeros((4, 3), dtype=np.float32)
    timestamps = np.arange(4, dtype=np.float32) / 30.0
    frames = adapter.mesh(
        expression=expression,
        rotations=rotations,
        translation=translation,
    )
    exported = export_animated_gnm_glb(
        tmp_path / "rank-zero.glb",
        adapter,
        frames,
        timestamps,
        mapping_path=tmp_path / "rank-zero-mapping.npz",
    )
    assert exported.rank == 0
    controls = _controls(
        tmp_path / "rank-zero-controls.npz",
        expression,
        rotations=rotations,
        translation=translation,
    )

    result = validate_glb_oral_geometry(
        exported.path,
        exported.mapping_path,
        adapter=adapter,
        reference_controls_path=controls,
        reference_frames=frames,
    )
    reconstruction = result.report["structural_reconstruction"]
    assert reconstruction["status"] == "passed"
    assert reconstruction["reference_evaluation_mode"] == (
        "provided_complete_gnm_frames"
    )
    assert result.report["claims"]["structural_reconstruction_validated"] is True


def test_glb_requires_complete_native_oral_mapping(
    tmp_path: Path, adapter: GNMAdapter
) -> None:
    exported = export_gnm_glb(
        tmp_path / "static.glb",
        adapter,
        adapter.mesh(),
        mapping_path=tmp_path / "mapping.npz",
    )
    static_result = validate_glb_oral_geometry(
        exported.path,
        exported.mapping_path,
        adapter=adapter,
    )
    assert static_result.report["structural_reconstruction"]["status"] == (
        "not_evaluated_no_reference"
    )
    assert (
        static_result.report["claims"]["structural_reconstruction_validated"] is False
    )
    with np.load(exported.mapping_path, allow_pickle=False) as values:
        payload = {name: np.asarray(values[name]).copy() for name in values.files}
    mapping = payload["glb_vertex_to_gnm_vertex"]
    missing_native = int(np.flatnonzero(adapter.vertex_group("tongue") > 0.5)[0])
    mapping[mapping == missing_native] = 0
    write_npz(tmp_path / "incomplete.npz", **payload)

    with pytest.raises(OralValidationError) as error:
        validate_glb_oral_geometry(
            exported.path,
            tmp_path / "incomplete.npz",
            adapter=adapter,
        )
    assert error.value.code == "GEOMETRY_ABSENT"


def _retained_real_job() -> Path:
    override = os.environ.get("AUTOANIM_RETAINED_ORAL_JOB")
    if override:
        path = Path(override).expanduser().resolve()
        assert path.is_dir(), f"AUTOANIM_RETAINED_ORAL_JOB is not a directory: {path}"
        return path

    candidates: list[tuple[int, Path]] = []
    for result_path in (ROOT / "artifacts/jobs").glob("*/result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        job = result_path.parent
        if (
            result.get("kind") != "audio_animation"
            or result.get("status") != "succeeded"
            or result.get("analysis", {}).get("motion_backend") != "learned_a2f"
            or not all(
                (job / name).is_file()
                for name in (
                    "controls.npz",
                    "animation.glb",
                    "animation-glb-mapping.npz",
                )
            )
        ):
            continue
        try:
            with np.load(job / "controls.npz", allow_pickle=False) as controls:
                frame_count = len(controls["expression"])
                targets = np.asarray(controls.get("lip_contact_target_gap", []))
        except (OSError, ValueError, KeyError):
            continue
        # Prefer a bounded retained take that exercises actual contact targets.
        rank = frame_count + (0 if np.count_nonzero(targets > 0.0) else 100_000)
        candidates.append((rank, job))
    if not candidates:
        pytest.skip(
            "retained learned-audio oral job unavailable; set AUTOANIM_RETAINED_ORAL_JOB"
        )
    return min(candidates, key=lambda value: value[0])[1]


def _verify_retained_artifacts(job: Path) -> None:
    result = json.loads((job / "result.json").read_text(encoding="utf-8"))
    artifacts = result.get("artifacts")
    assert isinstance(artifacts, dict)
    for logical, name in (
        ("controls", "controls.npz"),
        ("glb", "animation.glb"),
        ("glb_mapping", "animation-glb-mapping.npz"),
    ):
        entry = artifacts.get(logical)
        assert isinstance(entry, dict) and entry.get("name") == name
        path = job / name
        assert path.stat().st_size == entry.get("bytes")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry.get("sha256")


def test_retained_real_learned_audio_has_structural_oral_evidence_only(
    adapter: GNMAdapter,
) -> None:
    job = _retained_real_job()
    _verify_retained_artifacts(job)

    controls = validate_controls_npz(job / "controls.npz", adapter=adapter)
    assert controls.report["source"]["all_frames_evaluated"] is True
    assert controls.report["control_evidence"]["tongue_control_active_frames"] > 0
    assert controls.report["control_evidence"]["isolated_tongue_geometry_active_frames"] > 0
    assert controls.report["lip_contact"]["target_evidence"]["candidate_frames"] > 0
    assert controls.report["claims"]["phoneme_correctness_validated"] is False

    glb = validate_glb_oral_geometry(
        job / "animation.glb",
        job / "animation-glb-mapping.npz",
        adapter=adapter,
        reference_controls_path=job / "controls.npz",
    )
    assert glb.report["structural_reconstruction"]["status"] == "passed"
    assert glb.report["structural_reconstruction"]["validated"] is True
    assert glb.report["claims"]["exact_surface_intersection_validated"] is False
    assert glb.report["claims"]["perceptual_correctness_validated"] is False
    assert glb.report["claims"]["production_validated"] is False
