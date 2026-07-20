from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import json
from typing import Any

import numpy as np
import pytest

from autoanim_gnm.animation import (
    _mouth_gap_interocular,
    _mouth_lip_order_minimum_interocular,
    _mouth_step_quality_ratio,
    apply_lip_contact_correction,
    calibrate_lip_contact,
    compose_animation,
)
from autoanim_gnm.articulation_projection import (
    ARTICULATION_PROJECTION_SCHEMA_VERSION,
    LOWER_FACE_SLICE,
    PUPIL_SLICE,
    TONGUE_SLICE,
    articulation_array_sha256,
    articulation_arrays_byte_exact,
    project_articulation_trajectory,
)
from autoanim_gnm.audio import MouthCue
from autoanim_gnm.audio_pipeline import (
    _validated_articulation_projection_evidence,
)
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.rig import ControlRig


def _mouth_step(rig: ControlRig):
    return lambda left, right: _mouth_step_quality_ratio(rig, left, right)


def _mouth_gap(rig: ControlRig):
    return lambda expression: _mouth_gap_interocular(rig, expression)


def _lip_order(rig: ControlRig):
    return lambda expression: _mouth_lip_order_minimum_interocular(rig, expression)


def _assert_protected_exact(desired: np.ndarray, projected: np.ndarray) -> None:
    assert projected[:, : LOWER_FACE_SLICE.start].tobytes() == (
        desired[:, : LOWER_FACE_SLICE.start].tobytes()
    )
    assert projected[:, LOWER_FACE_SLICE.stop :].tobytes() == (
        desired[:, LOWER_FACE_SLICE.stop :].tobytes()
    )


def _assert_edge_limits(
    rig: ControlRig,
    expression: np.ndarray,
    timestamps: np.ndarray,
    *,
    maximum_step: float = 0.039,
    maximum_speed: float = 1.17,
) -> None:
    for index, delta_seconds in enumerate(np.diff(timestamps), start=1):
        observed = _mouth_step_quality_ratio(
            rig,
            expression[index - 1],
            expression[index],
        )
        expected = min(maximum_step, maximum_speed * float(delta_seconds))
        assert observed <= expected + 2.0e-6


def _projection(
    rig: ControlRig,
    desired: np.ndarray,
    timestamps: np.ndarray,
    **changes: Any,
):
    arguments: dict[str, Any] = {
        "mouth_step_metric": _mouth_step(rig),
        "lip_order_metric": _lip_order(rig),
        "mouth_gap_metric": _mouth_gap(rig),
    }
    arguments.update(changes)
    return project_articulation_trajectory(
        desired,
        timestamps,
        **arguments,
    )


def _canonical_report(value: Any) -> bytes:
    if is_dataclass(value):
        value = asdict(value)

    def normalize(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return {
                "dtype": item.dtype.str,
                "shape": list(item.shape),
                "value": item.tolist(),
            }
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(key): normalize(member) for key, member in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(member) for member in item]
        return item

    return json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def test_projection_contract_and_regions_are_frozen() -> None:
    assert ARTICULATION_PROJECTION_SCHEMA_VERSION == (
        "autoanim.articulation-projection/1.0"
    )
    assert LOWER_FACE_SLICE == slice(200, 350)
    assert TONGUE_SLICE == slice(350, 382)
    assert PUPIL_SLICE == slice(382, 383)


def test_authority_array_equality_includes_floating_sign_bits() -> None:
    positive = np.asarray((0.0,), dtype=np.float32)
    negative = np.asarray((-0.0,), dtype=np.float32)

    assert np.array_equal(positive, negative)
    assert not articulation_arrays_byte_exact(positive, negative)


def test_lower_face_jump_is_limited_without_touching_independent_tongue(
    rig: ControlRig,
) -> None:
    desired = np.zeros((4, rig.adapter.expression_dim), dtype=np.float32)
    desired[1:3, LOWER_FACE_SLICE] = rig.viseme("D")[LOWER_FACE_SLICE]
    desired[1, TONGUE_SLICE] = (
        np.float32(0.71) * rig.viseme("H")[TONGUE_SLICE]
    )
    desired[:, 17] = np.asarray((0.11, -0.07, 0.19, -0.13), dtype=np.float32)
    desired[:, PUPIL_SLICE] = np.asarray((0.03, 0.09, -0.02, 0.04), dtype=np.float32)[
        :, None
    ]
    timestamps = np.arange(len(desired), dtype=np.float64) / 30.0

    result = _projection(rig, desired, timestamps)

    assert np.any(result.limited_frames)
    assert result.expression[:, LOWER_FACE_SLICE].tobytes() != (
        desired[:, LOWER_FACE_SLICE].tobytes()
    )
    _assert_protected_exact(desired, result.expression)
    assert result.expression[:, TONGUE_SLICE].tobytes() == (
        desired[:, TONGUE_SLICE].tobytes()
    )
    _assert_edge_limits(rig, result.expression, timestamps)
    assert result.report["schema_version"] == ARTICULATION_PROJECTION_SCHEMA_VERSION
    assert result.report["algorithm"]
    assert len(result.report["desired_expression_sha256"]) == 64
    assert len(result.report["projected_expression_sha256"]) == 64
    assert result.report["desired_expression_sha256"] != (
        result.report["projected_expression_sha256"]
    )
    assert len(result.report["timestamps_sha256"]) == 64
    assert result.report["protected_channels_exact"] is True
    assert result.report["projection_induced_tongue_max_abs_control_delta"] == 0.0
    assert result.report["regions"]["tongue"]["exactly_preserved"] is True
    assert result.report["regions"]["lower_face"]["changed_frames"] > 0
    assert result.report["production_validated"] is False


@pytest.mark.parametrize(
    "timestamps",
    (
        np.arange(7, dtype=np.float64) / 30.0,
        np.arange(7, dtype=np.float64) / 60.0,
        np.asarray((0.0, 0.011, 0.044, 0.058, 0.103, 0.121, 0.190)),
    ),
    ids=("30hz", "60hz", "irregular"),
)
def test_projection_uses_exact_per_edge_time_bound(
    rig: ControlRig,
    timestamps: np.ndarray,
) -> None:
    desired = np.zeros((len(timestamps), rig.adapter.expression_dim), dtype=np.float32)
    desired[2:5, LOWER_FACE_SLICE] = (
        np.float32(0.90) * rig.viseme("D")[LOWER_FACE_SLICE]
    )

    result = _projection(rig, desired, timestamps)

    assert np.any(result.limited_frames)
    _assert_edge_limits(rig, result.expression, timestamps)
    _assert_protected_exact(desired, result.expression)


def test_protected_channel_mouth_leakage_fails_closed() -> None:
    desired = np.zeros((2, 383), dtype=np.float32)
    desired[1, 0] = np.float32(1.0)
    timestamps = np.asarray((0.0, 1.0 / 30.0), dtype=np.float64)

    def protected_step(left: np.ndarray, right: np.ndarray) -> float:
        return abs(float(right[0] - left[0]))

    with pytest.raises(AutoAnimError, match="[Pp]rotected|infeasible") as caught:
        project_articulation_trajectory(
            desired,
            timestamps,
            mouth_step_metric=protected_step,
        )
    assert caught.value.code == "ARTICULATION_PROJECTION_INFEASIBLE"


@pytest.mark.parametrize(
    "mutation",
    (
        "nonfinite_expression",
        "nonfinite_timestamps",
        "duplicate_timestamps",
        "descending_timestamps",
        "wrong_expression_rank",
        "wrong_expression_width",
        "wrong_timestamp_shape",
        "frame_count_mismatch",
    ),
)
def test_invalid_trajectory_inputs_fail_closed(
    rig: ControlRig,
    mutation: str,
) -> None:
    desired = np.zeros((3, rig.adapter.expression_dim), dtype=np.float32)
    timestamps = np.asarray((0.0, 1.0 / 30.0, 2.0 / 30.0), dtype=np.float64)
    if mutation == "nonfinite_expression":
        desired[1, 200] = np.nan
    elif mutation == "nonfinite_timestamps":
        timestamps[1] = np.nan
    elif mutation == "duplicate_timestamps":
        timestamps[2] = timestamps[1]
    elif mutation == "descending_timestamps":
        timestamps = timestamps[::-1].copy()
    elif mutation == "wrong_expression_rank":
        desired = desired[0]
    elif mutation == "wrong_expression_width":
        desired = desired[:, :-1]
    elif mutation == "wrong_timestamp_shape":
        timestamps = timestamps[:, None]
    elif mutation == "frame_count_mismatch":
        timestamps = timestamps[:-1]

    with pytest.raises(AutoAnimError):
        project_articulation_trajectory(
            desired,
            timestamps,
            mouth_step_metric=_mouth_step(rig),
        )


def test_projection_is_byte_deterministic_including_report(rig: ControlRig) -> None:
    desired = np.zeros((8, rig.adapter.expression_dim), dtype=np.float32)
    desired[2:6] = np.float32(0.55) * rig.viseme("D")
    desired[3:5, TONGUE_SLICE] = (
        np.float32(0.63) * rig.viseme("H")[TONGUE_SLICE]
    )
    timestamps = np.asarray(
        (0.0, 0.021, 0.052, 0.083, 0.117, 0.139, 0.178, 0.221),
        dtype=np.float64,
    )

    first = _projection(rig, desired, timestamps)
    second = _projection(rig, desired.copy(), timestamps.copy())

    assert first.expression.tobytes() == second.expression.tobytes()
    assert first.limited_frames.tobytes() == second.limited_frames.tobytes()
    assert first.contact_attained.tobytes() == second.contact_attained.tobytes()
    assert first.contact_continuity_restored.tobytes() == (
        second.contact_continuity_restored.tobytes()
    )
    assert _canonical_report(first.report) == _canonical_report(second.report)


def test_contact_anchor_is_restored_without_moving_tongue(rig: ControlRig) -> None:
    frame_count = 60
    timestamps = np.arange(frame_count, dtype=np.float64) / 30.0
    desired = np.zeros((frame_count, rig.adapter.expression_dim), dtype=np.float32)
    open_pose = np.float32(0.20) * rig.viseme("D")
    desired[6:54] = open_pose
    contact_pose, changed, target_gap = apply_lip_contact_correction(
        rig,
        open_pose,
        calibrate_lip_contact(rig),
        1.0,
    )
    assert changed
    desired[30] = contact_pose
    desired[31:36] = 0.0
    desired[30, TONGUE_SLICE] = (
        np.float32(0.75) * rig.viseme("H")[TONGUE_SLICE]
    )
    contact_targets = np.zeros(frame_count, dtype=np.float32)
    contact_targets[30] = np.float32(target_gap)

    result = _projection(
        rig,
        desired,
        timestamps,
        contact_target_gap=contact_targets,
    )

    assert result.contact_attained[30]
    assert result.contact_continuity_restored[30]
    assert result.expression[30, LOWER_FACE_SLICE].tobytes() == (
        desired[30, LOWER_FACE_SLICE].tobytes()
    )
    assert result.expression[:, TONGUE_SLICE].tobytes() == (
        desired[:, TONGUE_SLICE].tobytes()
    )
    assert _mouth_gap_interocular(rig, result.expression[30]) <= target_gap + 0.001
    _assert_edge_limits(rig, result.expression, timestamps)
    _assert_protected_exact(desired, result.expression)


def test_projection_does_not_introduce_inner_lip_inversion(rig: ControlRig) -> None:
    timestamps = np.arange(9, dtype=np.float64) / 60.0
    desired = np.zeros((len(timestamps), rig.adapter.expression_dim), dtype=np.float32)
    desired[2:4] = np.float32(0.65) * rig.viseme("B")
    desired[4:7] = np.float32(0.45) * rig.viseme("D")
    original_order = np.asarray(
        [_mouth_lip_order_minimum_interocular(rig, frame) for frame in desired]
    )
    assert np.all(original_order >= -0.0005)

    result = _projection(rig, desired, timestamps)
    projected_order = np.asarray(
        [_mouth_lip_order_minimum_interocular(rig, frame) for frame in result.expression]
    )

    assert np.all(projected_order >= -0.0005 - 1.0e-7)
    _assert_edge_limits(rig, result.expression, timestamps)


def test_pipeline_evidence_binding_recomputes_every_projection_hash(
    rig: ControlRig,
) -> None:
    track = compose_animation(
        [MouthCue(0.0, 1.0, "D")],
        1.0,
        30,
        rig,
        "neutral",
        head_motion=False,
    )

    validated = _validated_articulation_projection_evidence(
        track,
        rig,
        external_face_controls=False,
    )

    assert validated is not None
    report, desired, projected = validated
    assert report["desired_expression_sha256"] == articulation_array_sha256(desired)
    assert report["projected_expression_sha256"] == articulation_array_sha256(
        projected
    )
    assert report["timestamps_sha256"] == articulation_array_sha256(track.timestamps)
    assert report["contact_tolerance_interocular"] == 0.001
    assert report["lip_order_floor_interocular"] == -0.0005
    assert report["contact_horizon_seconds"] == pytest.approx(8.0 / 30.0)
    assert report["metric_contract_bound"] is True
    assert set(report["metric_contract"]) == {"mouth_step", "mouth_gap", "lip_order"}
    assert set(report["rig_binding"]) == {
        "gnm_head_sha256",
        "identity_array_sha256",
        "landmark_regressor_sha256",
    }


def test_pipeline_projection_evidence_rejects_partial_and_tampered_state(
    rig: ControlRig,
) -> None:
    track = compose_animation(
        [MouthCue(0.0, 0.5, "D")],
        0.5,
        30,
        rig,
        "neutral",
        head_motion=False,
    )

    with pytest.raises(AutoAnimError) as partial:
        _validated_articulation_projection_evidence(
            replace(track, articulation_projection_output=None),
            rig,
            external_face_controls=False,
        )
    assert partial.value.code == "ARTICULATION_PROJECTION_EVIDENCE_INCOMPLETE"

    tampered_report = dict(track.articulation_projection_report or {})
    tampered_report["projected_expression_sha256"] = "0" * 64
    with pytest.raises(AutoAnimError) as tampered:
        _validated_articulation_projection_evidence(
            replace(track, articulation_projection_report=tampered_report),
            rig,
            external_face_controls=False,
        )
    assert tampered.value.code == "ARTICULATION_PROJECTION_EVIDENCE_HASH_MISMATCH"


def test_pipeline_projection_evidence_is_mandatory(rig: ControlRig) -> None:
    track = compose_animation(
        [MouthCue(0.0, 0.5, "D")],
        0.5,
        30,
        rig,
        "neutral",
        head_motion=False,
    )

    with pytest.raises(AutoAnimError) as caught:
        _validated_articulation_projection_evidence(
            replace(
                track,
                articulation_projection_report=None,
                articulation_projection_desired=None,
                articulation_projection_output=None,
            ),
            rig,
            external_face_controls=False,
        )

    assert caught.value.code == "ARTICULATION_PROJECTION_EVIDENCE_INCOMPLETE"


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("limited_frames", 999),
        ("maximum_observed_step_interocular", 999.0),
        ("contact_anchor_loss_frames", 999),
        (
            "regions",
            {
                "tongue": {
                    "changed_frames": 999,
                    "exactly_preserved": True,
                    "max_abs_control_delta": 0.0,
                    "rms_control_delta": 0.0,
                    "start": 350,
                    "stop": 382,
                }
            },
        ),
        ("boundary_rest_frames", []),
        ("desired_snapshot_stage", "forged"),
    ),
)
def test_pipeline_projection_evidence_replays_derived_report(
    rig: ControlRig,
    field: str,
    forged_value: Any,
) -> None:
    track = compose_animation(
        [MouthCue(0.0, 0.5, "D")],
        0.5,
        30,
        rig,
        "neutral",
        head_motion=False,
    )
    forged_report = dict(track.articulation_projection_report or {})
    forged_report[field] = forged_value

    with pytest.raises(AutoAnimError) as caught:
        _validated_articulation_projection_evidence(
            replace(track, articulation_projection_report=forged_report),
            rig,
            external_face_controls=False,
        )

    assert caught.value.code == "ARTICULATION_PROJECTION_EVIDENCE_DECISION_MISMATCH"


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("frame_count", 16.0),
        ("expression_dim", 383.0),
        ("projectable_range", [200.0, 350.0]),
        ("boundary_rest_included_before_desired_hash", 1),
    ),
)
def test_pipeline_projection_evidence_rejects_json_type_substitution(
    rig: ControlRig,
    field: str,
    forged_value: Any,
) -> None:
    track = compose_animation(
        [MouthCue(0.0, 0.5, "D")],
        0.5,
        30,
        rig,
        "neutral",
        head_motion=False,
    )
    forged_report = dict(track.articulation_projection_report or {})
    forged_report[field] = forged_value

    with pytest.raises(AutoAnimError):
        _validated_articulation_projection_evidence(
            replace(track, articulation_projection_report=forged_report),
            rig,
            external_face_controls=False,
        )


@pytest.mark.parametrize(
    "field",
    (
        "mouth_speed_limited",
        "lip_contact_attained",
        "contact_continuity_restored",
    ),
)
def test_pipeline_projection_evidence_replays_decision_arrays(
    rig: ControlRig,
    field: str,
) -> None:
    track = compose_animation(
        [MouthCue(0.0, 0.5, "D")],
        0.5,
        30,
        rig,
        "neutral",
        head_motion=False,
    )
    forged = ~np.asarray(getattr(track, field), dtype=bool)

    with pytest.raises(AutoAnimError) as caught:
        _validated_articulation_projection_evidence(
            replace(track, **{field: forged}),
            rig,
            external_face_controls=False,
        )

    assert caught.value.code == "ARTICULATION_PROJECTION_EVIDENCE_DECISION_MISMATCH"


def test_pipeline_projection_evidence_rejects_invalid_clock_and_delivery(
    rig: ControlRig,
) -> None:
    track = compose_animation(
        [MouthCue(0.0, 0.5, "D")],
        0.5,
        30,
        rig,
        "neutral",
        head_motion=False,
    )
    duplicate_clock = track.timestamps.copy()
    duplicate_clock[1] = duplicate_clock[0]
    refreshed = dict(track.articulation_projection_report or {})
    refreshed["timestamps_sha256"] = articulation_array_sha256(duplicate_clock)

    with pytest.raises(AutoAnimError) as clock_error:
        _validated_articulation_projection_evidence(
            replace(
                track,
                timestamps=duplicate_clock,
                articulation_projection_report=refreshed,
            ),
            rig,
            external_face_controls=False,
        )
    assert clock_error.value.code == "ARTICULATION_PROJECTION_EVIDENCE_INCOMPLETE"

    nonfinite_delivery = track.expression.copy()
    nonfinite_delivery[0, 0] = np.nan
    with pytest.raises(AutoAnimError) as delivery_error:
        _validated_articulation_projection_evidence(
            replace(track, expression=nonfinite_delivery),
            rig,
            external_face_controls=False,
        )
    assert delivery_error.value.code == "ARTICULATION_PROJECTION_EVIDENCE_INCOMPLETE"


def test_pipeline_projection_protected_authority_is_byte_exact(
    rig: ControlRig,
) -> None:
    track = compose_animation(
        [MouthCue(0.0, 0.5, "D")],
        0.5,
        30,
        rig,
        "neutral",
        head_motion=False,
    )
    forged_output = np.asarray(track.articulation_projection_output).copy()
    assert forged_output[0, 0] == 0.0
    forged_output[0, 0] = np.float32(-0.0)
    assert forged_output[0, 0].tobytes() != np.float32(0.0).tobytes()
    refreshed = dict(track.articulation_projection_report or {})
    refreshed["projected_expression_sha256"] = articulation_array_sha256(
        forged_output
    )

    with pytest.raises(AutoAnimError) as caught:
        _validated_articulation_projection_evidence(
            replace(
                track,
                articulation_projection_report=refreshed,
                articulation_projection_output=forged_output,
            ),
            rig,
            external_face_controls=False,
        )

    assert caught.value.code == "ARTICULATION_PROJECTION_EVIDENCE_INCOMPLETE"


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("schema_version", "forged"),
        ("algorithm", "forged"),
        ("projectable_range", [0, 383]),
        ("protected_ranges", []),
        ("maximum_step_interocular", 999.0),
        ("maximum_speed_interocular_per_second", 999.0),
        ("contact_tolerance_interocular", 999.0),
        ("lip_order_floor_interocular", -999.0),
        ("contact_horizon_seconds", 999.0),
    ),
)
def test_pipeline_projection_evidence_rejects_forged_compiler_contract(
    rig: ControlRig,
    field: str,
    forged_value: Any,
) -> None:
    track = compose_animation(
        [MouthCue(0.0, 0.5, "D")],
        0.5,
        30,
        rig,
        "neutral",
        head_motion=False,
    )
    forged_report = dict(track.articulation_projection_report or {})
    forged_report[field] = forged_value

    with pytest.raises(AutoAnimError) as caught:
        _validated_articulation_projection_evidence(
            replace(track, articulation_projection_report=forged_report),
            rig,
            external_face_controls=False,
        )

    assert caught.value.code == "ARTICULATION_PROJECTION_EVIDENCE_CONFIG_MISMATCH"


def test_pipeline_projection_evidence_rejects_wrong_compiler_family(
    rig: ControlRig,
) -> None:
    fallback_track = compose_animation(
        [MouthCue(0.0, 0.5, "D")],
        0.5,
        30,
        rig,
        "neutral",
        head_motion=False,
    )

    with pytest.raises(AutoAnimError) as caught:
        _validated_articulation_projection_evidence(
            fallback_track,
            rig,
            external_face_controls=True,
        )

    assert caught.value.code == "ARTICULATION_PROJECTION_EVIDENCE_CONFIG_MISMATCH"
