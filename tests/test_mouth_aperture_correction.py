from __future__ import annotations

import numpy as np
import pytest

from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.animation import _mouth_step_quality_ratio
from autoanim_gnm.mouth_aperture_correction import (
    MouthApertureConfig,
    MouthContactEvidence,
    correct_mouth_aperture,
    mouth_aperture_target_attainment,
)
from autoanim_gnm.rig import ControlRig


def _track(rig: ControlRig, expressions: list[np.ndarray] | np.ndarray) -> dict[str, object]:
    expression = np.asarray(expressions, dtype=np.float32)
    frames = len(expression)
    return {
        "rig": rig,
        "identity": rig.identity.copy(),
        "expression": expression,
        "rotations": np.zeros((frames, rig.adapter.model.num_joints, 3), dtype=np.float32),
        "translation": np.zeros((frames, 3), dtype=np.float32),
        "timestamps_seconds": np.arange(frames, dtype=np.float64) * 0.1,
        "eligible_frames": np.ones(frames, dtype=bool),
        "contact_evidence": MouthContactEvidence(
            anchor=np.zeros(frames, dtype=bool),
            confidence=np.zeros(frames, dtype=np.float32),
            label=tuple("none" for _ in range(frames)),
        ),
    }


def _mild_open(rig: ControlRig, scale: float = 0.20) -> np.ndarray:
    return np.float32(scale) * rig.viseme("B")


def _call(track: dict[str, object], **changes: object):
    arguments = dict(track)
    arguments.update(changes)
    return correct_mouth_aperture(**arguments)


def test_gain_one_is_an_exact_noop_for_the_entire_track(rig: ControlRig) -> None:
    expression = np.stack(
        (
            np.zeros(rig.adapter.expression_dim, dtype=np.float32),
            _mild_open(rig),
            np.float32(0.45) * rig.viseme("H") + np.float32(0.2) * rig.emotion("joy"),
        )
    )
    track = _track(rig, expression)
    rotations = np.asarray(track["rotations"]).copy()
    rotations[1, 1] = np.asarray((0.01, -0.02, 0.005), dtype=np.float32)
    track["rotations"] = rotations
    result = _call(track, config=MouthApertureConfig(gain=1.0))

    assert result.expression.tobytes() == expression.tobytes()
    assert result.rotations.tobytes() == rotations.tobytes()
    assert result.translation.tobytes() == np.asarray(track["translation"]).tobytes()
    assert not result.correction_applied.any()
    assert mouth_aperture_target_attainment(result) == 1.0
    assert result.mesh_validation_passed
    assert not result.expression.flags.writeable


def test_closed_contact_and_pbm_frames_are_byte_identical(rig: ControlRig) -> None:
    closed = np.zeros(rig.adapter.expression_dim, dtype=np.float32)
    opened = _mild_open(rig)
    expression = np.stack((closed, opened, opened, opened, opened))
    track = _track(rig, expression)
    track["contact_evidence"] = MouthContactEvidence(
        anchor=np.asarray((False, True, False, False, False), dtype=bool),
        confidence=np.asarray((0.0, 0.0, 0.8, 0.0, 0.0), dtype=np.float32),
        label=("none", "none", "none", "P", "m"),
    )
    result = _call(track, config=MouthApertureConfig(gain=1.8))

    for frame in (0, 1, 2, 3, 4):
        assert result.expression[frame].tobytes() == expression[frame].tobytes()
    assert result.reports[0].status == "closed_or_neutral"
    assert all(result.reports[frame].status == "protected_contact" for frame in (1, 2, 3, 4))
    np.testing.assert_array_equal(result.protected_contact, (False, True, True, True, True))


def test_slight_opening_reaches_its_declared_geometry_target(rig: ControlRig) -> None:
    expression = np.stack((_mild_open(rig), _mild_open(rig)))
    track = _track(rig, expression)
    result = _call(track, config=MouthApertureConfig(gain=1.5))

    for report in result.reports:
        assert report.status == "corrected"
        assert report.target_attained
        assert report.final_gap_interocular == pytest.approx(
            report.bounded_target_gap_interocular,
            abs=2.1e-4,
        )
        assert report.final_gap_interocular > report.original_gap_interocular
        assert report.maximum_coefficient_delta > 0.0


def test_authored_opening_is_locally_reduced_at_the_final_quality_step_limit(
    rig: ControlRig,
) -> None:
    expression = np.stack(
        (
            np.zeros(rig.adapter.expression_dim, dtype=np.float32),
            _mild_open(rig, 0.40),
            np.zeros(rig.adapter.expression_dim, dtype=np.float32),
        )
    )
    track = _track(rig, expression)
    track["timestamps_seconds"] = np.arange(3, dtype=np.float64) * 0.1
    maximum_speed = 0.35
    result = _call(
        track,
        config=MouthApertureConfig(
            gain=2.0,
            maximum_final_mouth_speed_interocular_per_second=maximum_speed,
        ),
    )

    assert 0.0 < result.final_continuity_scale[1] < 1.0
    assert "final_mouth_step_continuity" in result.reports[1].bounds
    assert not result.reports[1].target_attained
    assert result.expression[0].tobytes() == expression[0].tobytes()
    assert result.expression[2].tobytes() == expression[2].tobytes()
    assert max(
        _mouth_step_quality_ratio(rig, result.expression[index - 1], result.expression[index])
        for index in range(1, len(expression))
    ) <= maximum_speed * 0.1 + 1.0e-7
    assert result.final_continuity_limit_interocular == pytest.approx(
        maximum_speed * 0.1
    )
    assert result.final_continuity_speed_interocular_per_second == maximum_speed


def test_authored_mouth_aperture_speed_has_30_60_time_parity(
    rig: ControlRig,
) -> None:
    duration = 0.5
    maximum_speed = 0.06
    config = MouthApertureConfig(
        gain=2.0,
        maximum_correction_velocity_interocular_per_second=10.0,
        maximum_final_mouth_speed_interocular_per_second=maximum_speed,
    )
    observed_speeds: dict[int, float] = {}

    for fps in (30, 60):
        timestamps = np.arange(int(duration * fps), dtype=np.float64) / fps
        expression = np.stack(
            [
                np.float32(0.10 + 0.50 * timestamp) * rig.viseme("B")
                for timestamp in timestamps
            ]
        ).astype(np.float32)
        track = _track(rig, expression)
        track["timestamps_seconds"] = timestamps
        result = _call(track, config=config)
        speeds = np.asarray(
            [
                _mouth_step_quality_ratio(rig, left, right) * fps
                for left, right in zip(
                    result.expression[:-1],
                    result.expression[1:],
                    strict=True,
                )
            ]
        )
        observed_speeds[fps] = float(np.max(speeds))

        assert np.count_nonzero(result.final_continuity_scale < 1.0 - 1.0e-6) > 0
        assert maximum_speed - 1.0e-4 <= observed_speeds[fps] <= maximum_speed + 1.0e-5
        assert result.final_continuity_limit_interocular == pytest.approx(
            maximum_speed / fps,
            abs=1.0e-12,
        )
        assert result.final_continuity_speed_interocular_per_second == maximum_speed

    assert observed_speeds[30] == pytest.approx(observed_speeds[60], abs=1.0e-4)


@pytest.mark.parametrize("fps", (12, 24))
def test_authored_aperture_retains_absolute_step_safety_below_30fps(
    rig: ControlRig,
    fps: int,
) -> None:
    expression = np.stack(
        (
            np.zeros(rig.adapter.expression_dim, dtype=np.float32),
            _mild_open(rig, 0.40),
            np.zeros(rig.adapter.expression_dim, dtype=np.float32),
        )
    )
    track = _track(rig, expression)
    track["timestamps_seconds"] = np.arange(3, dtype=np.float64) / fps
    config = MouthApertureConfig(gain=2.0)

    result = _call(track, config=config)

    maximum_step = max(
        _mouth_step_quality_ratio(
            rig,
            result.expression[index - 1],
            result.expression[index],
        )
        for index in range(1, len(expression))
    )
    assert maximum_step <= config.maximum_final_mouth_step_interocular + 1.0e-7
    assert result.final_continuity_limit_interocular == pytest.approx(
        config.maximum_final_mouth_step_interocular
    )


def test_hard_limits_are_enforced_and_every_active_limit_is_reported(rig: ControlRig) -> None:
    expression = np.stack((_mild_open(rig, 0.35),))
    track = _track(rig, expression)
    config = MouthApertureConfig(
        gain=8.0,
        bias_interocular=0.1,
        maximum_target_gap_interocular=0.12,
        maximum_added_aperture_interocular=0.04,
        maximum_coefficient_delta=0.01,
    )
    result = _call(track, config=config)
    report = result.reports[0]

    assert report.status == "limited"
    assert "maximum_target_gap" in report.bounds
    assert "maximum_added_aperture" in report.bounds
    assert "coefficient_delta" in report.bounds
    assert "target_unattained" in report.bounds
    assert report.maximum_coefficient_delta <= config.maximum_coefficient_delta + 2.0e-6
    assert report.nonmouth_displacement_interocular <= (
        config.maximum_nonmouth_displacement_interocular + 1.0e-7
    )
    assert report.upper_face_displacement_interocular <= (
        config.maximum_upper_face_displacement_interocular + 1.0e-7
    )
    assert report.tongue_displacement_interocular <= (
        config.maximum_tongue_displacement_interocular + 1.0e-7
    )
    assert np.max(np.abs(result.expression), initial=0.0) <= 3.0
    assert mouth_aperture_target_attainment(result) == 0.0


def test_target_attainment_is_not_vacuously_perfect_without_eligible_frames(
    rig: ControlRig,
) -> None:
    closed = np.zeros((2, rig.adapter.expression_dim), dtype=np.float32)
    result = _call(
        _track(rig, closed),
        config=MouthApertureConfig(gain=1.8),
    )
    assert mouth_aperture_target_attainment(result) is None


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("tampered_identity", "identity does not exactly match"),
        ("nonfinite_expression", "nonfinite"),
        ("bad_timestamps", "strictly increasing"),
        ("bad_confidence", "between zero and one"),
        ("bad_labels", "one string per frame"),
        ("bad_dtype", "must use float32"),
    ),
)
def test_invalid_or_tampered_inputs_are_rejected(
    rig: ControlRig,
    mutation: str,
    match: str,
) -> None:
    track = _track(rig, np.stack((_mild_open(rig), _mild_open(rig))))
    if mutation == "tampered_identity":
        identity = np.asarray(track["identity"]).copy()
        identity[0] += np.float32(1.0e-4)
        track["identity"] = identity
    elif mutation == "nonfinite_expression":
        expression = np.asarray(track["expression"]).copy()
        expression[0, 200] = np.nan
        track["expression"] = expression
    elif mutation == "bad_timestamps":
        track["timestamps_seconds"] = np.asarray((0.1, 0.1), dtype=np.float64)
    elif mutation == "bad_confidence":
        track["contact_evidence"] = MouthContactEvidence(
            anchor=np.zeros(2, dtype=bool),
            confidence=np.asarray((0.0, 1.01), dtype=np.float32),
            label=("none", "none"),
        )
    elif mutation == "bad_labels":
        track["contact_evidence"] = MouthContactEvidence(
            anchor=np.zeros(2, dtype=bool),
            confidence=np.zeros(2, dtype=np.float32),
            label=("none",),
        )
    elif mutation == "bad_dtype":
        track["expression"] = np.asarray(track["expression"], dtype=np.float64)
    with pytest.raises(AutoAnimError, match=match):
        _call(track, config=MouthApertureConfig(gain=1.5))


@pytest.mark.parametrize(
    "config",
    (
        MouthApertureConfig(gain=0.99),
        MouthApertureConfig(bias_interocular=-0.001),
        MouthApertureConfig(maximum_coefficient_delta=0.0),
        MouthApertureConfig(contact_confidence_threshold=1.1),
        MouthApertureConfig(gain=float("nan")),
    ),
)
def test_invalid_artist_or_safety_configuration_is_rejected(
    rig: ControlRig,
    config: MouthApertureConfig,
) -> None:
    track = _track(rig, np.stack((_mild_open(rig),)))
    with pytest.raises(AutoAnimError, match="configuration|cannot|limits|threshold"):
        _call(track, config=config)


def test_identity_variation_uses_each_characters_neutral_geometry(adapter, decoder) -> None:
    neutral_rig = ControlRig(adapter, decoder)
    identity = np.zeros(adapter.identity_dim, dtype=np.float32)
    identity[:60] = np.random.default_rng(17).normal(0.0, 0.35, 60).astype(np.float32)
    character_rig = ControlRig(adapter, decoder, identity=identity)

    outputs = []
    for current_rig in (neutral_rig, character_rig):
        track = _track(current_rig, np.stack((_mild_open(current_rig),)))
        outputs.append(_call(track, config=MouthApertureConfig(gain=1.4)))

    assert outputs[0].identity_sha256 != outputs[1].identity_sha256
    assert outputs[0].neutral_gap_interocular != pytest.approx(
        outputs[1].neutral_gap_interocular,
        abs=1.0e-6,
    )
    assert outputs[0].reports[0].target_attained
    assert outputs[1].reports[0].target_attained
    assert outputs[0].mesh_validation_passed and outputs[1].mesh_validation_passed


def test_mutated_rig_identity_cannot_reuse_stale_neutral_geometry(adapter, decoder) -> None:
    current_rig = ControlRig(adapter, decoder)
    current_rig.identity.setflags(write=True)
    current_rig.identity[0] += np.float32(0.01)
    current_rig.identity.setflags(write=False)
    track = _track(current_rig, np.stack((_mild_open(current_rig),)))

    with pytest.raises(AutoAnimError, match="cached neutral geometry"):
        _call(track, config=MouthApertureConfig(gain=1.4))


def test_audit_hash_covers_labels_and_artist_configuration(rig: ControlRig) -> None:
    track = _track(rig, np.stack((_mild_open(rig),)))
    baseline = _call(track, config=MouthApertureConfig(gain=1.4))
    relabeled = _call(
        track,
        contact_evidence=MouthContactEvidence(
            anchor=np.zeros(1, dtype=bool),
            confidence=np.zeros(1, dtype=np.float32),
            label=("unprotected-observation",),
        ),
        config=MouthApertureConfig(gain=1.4),
    )
    retuned = _call(track, config=MouthApertureConfig(gain=1.5))

    assert baseline.input_sha256 != relabeled.input_sha256
    assert baseline.input_sha256 != retuned.input_sha256


def test_correction_envelope_is_continuous_around_unchanged_frames(rig: ControlRig) -> None:
    closed = np.zeros(rig.adapter.expression_dim, dtype=np.float32)
    opened = _mild_open(rig, 0.35)
    expression = np.stack((closed, opened, opened, opened, closed))
    track = _track(rig, expression)
    timestamps = np.arange(5, dtype=np.float64) / 30.0
    track["timestamps_seconds"] = timestamps
    config = MouthApertureConfig(
        gain=3.0,
        maximum_correction_velocity_interocular_per_second=0.12,
    )
    result = _call(track, config=config)
    lifts = np.asarray(
        [report.final_gap_interocular - report.original_gap_interocular for report in result.reports]
    )
    per_step = config.maximum_correction_velocity_interocular_per_second / 30.0

    assert np.max(np.abs(np.diff(lifts)), initial=0.0) <= per_step + 3.0e-4
    assert any("continuity_velocity" in report.bounds for report in result.reports[1:4])
    assert result.expression[0].tobytes() == expression[0].tobytes()
    assert result.expression[-1].tobytes() == expression[-1].tobytes()


def test_no_collateral_controls_and_mesh_residuals_stay_inside_audited_bounds(
    rig: ControlRig,
) -> None:
    expression = _mild_open(rig, 0.28) + np.float32(0.35) * rig.emotion("joy")
    expression[350:382] = np.float32(0.65) * rig.viseme("H")[350:382]
    track = _track(rig, np.stack((expression,)))
    rotations = np.asarray(track["rotations"]).copy()
    rotations[0, 0] = np.asarray((-0.01, 0.02, 0.005), dtype=np.float32)
    rotations[0, 1] = np.asarray((0.015, -0.01, 0.0), dtype=np.float32)
    translation = np.asarray(((0.01, -0.02, 0.03),), dtype=np.float32)
    track["rotations"] = rotations
    track["translation"] = translation
    config = MouthApertureConfig(gain=1.6)
    result = _call(track, config=config)
    report = result.reports[0]

    np.testing.assert_array_equal(result.expression[:, :200], np.stack((expression,))[:, :200])
    np.testing.assert_array_equal(result.expression[:, 350:], np.stack((expression,))[:, 350:])
    assert result.rotations.tobytes() == rotations.tobytes()
    assert result.translation.tobytes() == translation.tobytes()
    assert report.nonmouth_displacement_interocular <= config.maximum_nonmouth_displacement_interocular
    assert report.upper_face_displacement_interocular <= config.maximum_upper_face_displacement_interocular
    assert report.tongue_displacement_interocular <= config.maximum_tongue_displacement_interocular

    mesh = rig.adapter.mesh(
        identity=rig.identity,
        expression=result.expression[0],
        rotations=result.rotations[0],
        translation=result.translation[0],
    )
    assert np.isfinite(mesh).all()
