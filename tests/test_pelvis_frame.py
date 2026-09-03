"""D7: `Hips` gets its own frame from the pelvis's own landmarks.

Every test here asserts it imported THIS worktree's package, and every synthetic body is
posed through our own forward kinematics so the answer is exact by construction rather
than by agreement with anything.

docs/reviews/pelvis-frame-2026-09-04.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import autoanim_gnm
from autoanim_gnm import commercial_multiview as cm
from autoanim_gnm.body import DETAILED_HUMANOID


def test_the_package_under_test_is_this_worktree() -> None:
    """The venv's editable install points at the MAIN tree; a green test there is a lie."""

    here = Path(__file__).resolve().parents[1]
    assert str(Path(autoanim_gnm.__file__).resolve()).startswith(str(here))


# --------------------------------------------------------------- the synthetic posed body
CAPTURE_FROM_RIG = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
"""Rig Y-up -> capture Z-up. `positions_to_body_track` applies the inverse to its input."""


def _to_capture(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float64) @ CAPTURE_FROM_RIG.T


def posed_body(
    frames: int = 24,
    pelvis_deg: float = 32.0,
    thorax_deg: float = -14.0,
    wobble_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A body whose PELVIS and THORAX deliberately disagree, in capture Z-up metres.

    Returns ``(positions[frame, 19, 3], spine1[frame, 3], pelvis_rotations[frame, 3, 3])``
    with the pelvis rotations expressed in the RIG's frame -- which is what the converter
    writes on `Hips`.

    The three pelvis landmarks are placed by rotating the SHIPPED rest template
    (`SOMA77_REST_PELVIS_TEMPLATE_M`), so recovering the rotation from them is an IDENTITY
    and not a fit. That is what makes the closure band discriminating: it can only fail if
    the converter's construction is wrong, never because the fixture was generous.

    `wobble_m` adds a deterministic zig-zag to the spine point so a temporal window has
    something to remove; the truth rotation is unaffected by it.
    """

    template = np.asarray(cm.SOMA77_REST_PELVIS_TEMPLATE_M, dtype=np.float64)
    to_spine1, to_left, to_right = template
    positions = np.zeros((frames, len(cm.JOINT_NAMES), 3), dtype=np.float64)
    spine = np.zeros((frames, 3), dtype=np.float64)
    pelvis_rotations = np.zeros((frames, 3, 3), dtype=np.float64)
    for frame in range(frames):
        phase = frame / max(frames - 1, 1)
        pelvis = Rotation.from_euler(
            "xyz", (pelvis_deg * (0.5 + 0.5 * phase), 11.0 * phase, 4.0 * phase), degrees=True
        )
        thorax = Rotation.from_euler(
            "xyz", (thorax_deg * (0.5 + 0.5 * phase), -7.0 * phase, 0.0), degrees=True
        )
        pelvis_rotations[frame] = pelvis.as_matrix()
        root_rig = np.asarray((0.0, 0.98, 0.4 * phase))
        left_rig = root_rig + pelvis.apply(to_left)
        right_rig = root_rig + pelvis.apply(to_right)
        spine_rig = root_rig + pelvis.apply(to_spine1)
        if wobble_m:
            spine_rig = spine_rig + np.asarray(
                (wobble_m * (-1.0) ** frame, 0.0, wobble_m * (-1.0) ** (frame // 2)))
        hip_mid_rig = 0.5 * (left_rig + right_rig)
        neck_rig = hip_mid_rig + thorax.apply((0.0, 1.0, 0.0)) * 0.58
        shoulder_half = thorax.apply((1.0, 0.0, 0.0)) * 0.27
        at = {
            "root": root_rig, "left_hip": left_rig, "right_hip": right_rig,
            "neck": neck_rig,
            "left_shoulder": neck_rig + shoulder_half,
            "right_shoulder": neck_rig - shoulder_half,
            "left_elbow": neck_rig + shoulder_half * 2.0 - (0.0, 0.28, 0.0),
            "right_elbow": neck_rig - shoulder_half * 2.0 - (0.0, 0.28, 0.0),
            "left_wrist": neck_rig + shoulder_half * 2.0 - (0.0, 0.53, 0.0),
            "right_wrist": neck_rig - shoulder_half * 2.0 - (0.0, 0.53, 0.0),
            "left_knee": left_rig - (0.0, 0.42, 0.0),
            "right_knee": right_rig - (0.0, 0.42, 0.0),
            "left_ankle": left_rig - (0.0, 0.82, 0.0),
            "right_ankle": right_rig - (0.0, 0.82, 0.0),
            "nose": neck_rig + (0.0, 0.18, 0.05),
            "left_eye": neck_rig + (0.03, 0.20, 0.06),
            "right_eye": neck_rig + (-0.03, 0.20, 0.06),
            "left_ear": neck_rig + (0.07, 0.19, 0.0),
            "right_ear": neck_rig + (-0.07, 0.19, 0.0),
        }
        for name, value in at.items():
            positions[frame, cm.JOINT_INDEX[name]] = _to_capture(np.asarray(value, dtype=np.float64))
        spine[frame] = _to_capture(spine_rig)
    return positions, spine, pelvis_rotations


def hips_world_rotations(track) -> np.ndarray:
    """`Hips`' world rotation from the track. Its parent `Root` is the identity here."""

    index = track.joint_names.index("Hips")
    return Rotation.from_quat(
        np.asarray(track.local_rotations_xyzw[:, index], dtype=np.float64)
    ).as_matrix()


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.einsum("nij,nkj->nik", a, b)
    trace = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def convert(positions: np.ndarray, spine: np.ndarray | None, **kwargs):
    return cm.positions_to_body_track(
        positions,
        sample_rate_hz=30,
        provenance_sha256="a" * 64,
        spine_world_z_up_m=spine,
        skeleton=DETAILED_HUMANOID,
        **kwargs,
    )


# --------------------------------------------------------------------- the closure band
def test_a_clean_synthetic_body_round_trips_the_pelvis_frame() -> None:
    """EXACT-RECOVERY ORACLE. The posed pelvis comes back out of the converter."""

    positions, spine, truth = posed_body()
    track = convert(positions, spine)
    error = geodesic_deg(hips_world_rotations(track), truth)
    assert error.max() < 0.01, f"worst {error.max():.6f} deg"


def test_without_a_spine_landmark_the_pelvis_is_the_trunk_and_that_is_the_defect() -> None:
    """MUST-FAIL arm of the same oracle: the pre-D7 behaviour reads the posed difference."""

    positions, spine, truth = posed_body()
    track = convert(positions, None)
    error = geodesic_deg(hips_world_rotations(track), truth)
    assert np.median(error) > 10.0, f"median {np.median(error):.3f} deg"


def test_a_world_vertical_pelvis_is_a_degenerate_and_is_not_what_ships() -> None:
    """The plausible shortcut, run through the identical code path, must not be exact."""

    positions, spine, truth = posed_body()
    hip_mid = 0.5 * (positions[:, cm.JOINT_INDEX["left_hip"]]
                     + positions[:, cm.JOINT_INDEX["right_hip"]])
    vertical = hip_mid + np.asarray((0.0, 0.0, 0.108))
    track = convert(positions, vertical)
    error = geodesic_deg(hips_world_rotations(track), truth)
    assert np.median(error) > 5.0, f"median {np.median(error):.3f} deg"


# ------------------------------------------------------------- legacy behaviour is exact
def test_no_spine_input_never_reaches_the_pelvis_code(monkeypatch) -> None:
    """The branch is one line: with no spine landmark nothing D7 added executes."""

    def explode(*args, **kwargs):  # pragma: no cover -- it must not run
        raise AssertionError("_pelvis_world_frames ran on a legacy call")

    positions, _, _ = posed_body()
    reference = convert(positions, None)
    monkeypatch.setattr(cm, "_pelvis_world_frames", explode)
    track = convert(positions, None)
    assert np.array_equal(track.local_rotations_xyzw, reference.local_rotations_xyzw)
    assert np.array_equal(track.root_translation_m, reference.root_translation_m)
    assert np.array_equal(track.foot_contacts, reference.foot_contacts)


def test_the_legacy_hips_channel_is_still_the_trunk_line_bit_for_bit() -> None:
    """What `Hips` was before D7, asserted against the construction itself."""

    positions, _, _ = posed_body()
    track = convert(positions, None)
    points = positions[..., (0, 2, 1)].copy()
    points[..., 2] *= -1.0
    index = track.joint_names.index("Hips")
    for frame in range(len(points)):
        p = points[frame]
        pelvis = 0.5 * (p[cm.JOINT_INDEX["left_hip"]] + p[cm.JOINT_INDEX["right_hip"]])
        expected = cm._frame_alignment(
            (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
            p[cm.JOINT_INDEX["neck"]] - pelvis,
            p[cm.JOINT_INDEX["left_hip"]] - p[cm.JOINT_INDEX["right_hip"]],
        )
        got = np.asarray(track.local_rotations_xyzw[frame, index], dtype=np.float64)
        assert min(np.abs(got - expected).max(), np.abs(got + expected).max()) < 1e-6


# --------------------------------------------------- gaps: whole subject, never per frame
def test_a_gap_is_interpolated_and_the_definition_never_switches_per_frame() -> None:
    positions, spine, truth = posed_body()
    holed = spine.copy()
    holed[5:9] = np.nan
    quaternions, report = cm._pelvis_world_frames(
        _rig_points(positions), _rig_spine(holed)
    )
    assert report["status"] == "solved"
    assert report["interpolated_frames"] == 4
    assert quaternions is not None and np.isfinite(quaternions).all()
    # no frame keeps a trunk-derived pelvis: the largest single-frame step is bounded,
    # where a per-frame flip between the two definitions would show tens of degrees.
    steps = geodesic_deg(
        Rotation.from_quat(quaternions[1:]).as_matrix(),
        Rotation.from_quat(quaternions[:-1]).as_matrix(),
    )
    assert steps.max() < 10.0, f"largest step {steps.max():.3f} deg"


def test_too_few_resolved_frames_falls_the_WHOLE_subject_back_with_a_reason() -> None:
    positions, spine, _ = posed_body()
    holed = spine.copy()
    holed[: int(0.8 * len(holed))] = np.nan
    quaternions, report = cm._pelvis_world_frames(
        _rig_points(positions), _rig_spine(holed)
    )
    assert quaternions is None
    assert report["status"] == "fell_back_to_torso_frame"
    assert "PELVIS_MINIMUM_RESOLVED_FRACTION" in report["reason"]
    # and the converter honours it: the track is the legacy one, exactly.
    reference = convert(positions, None)
    track = convert(positions, holed)
    assert np.array_equal(track.local_rotations_xyzw, reference.local_rotations_xyzw)
    assert np.array_equal(track.root_translation_m, reference.root_translation_m)


def _rig_points(positions: np.ndarray) -> np.ndarray:
    points = positions[..., (0, 2, 1)].copy()
    points[..., 2] *= -1.0
    return points


def _rig_spine(spine: np.ndarray) -> np.ndarray:
    out = spine[..., (0, 2, 1)].copy()
    out[..., 2] *= -1.0
    return out


# ------------------------------------------------------------------------- the interface
def test_a_wrong_shaped_spine_input_is_rejected() -> None:
    positions, spine, _ = posed_body()
    with pytest.raises(cm.CommercialMultiviewError):
        convert(positions, spine[:, :2])
    with pytest.raises(cm.CommercialMultiviewError):
        convert(positions, spine[:-1])


def test_an_unknown_pelvis_source_is_rejected_rather_than_silently_defaulted() -> None:
    positions, spine, _ = posed_body()
    with pytest.raises(cm.CommercialMultiviewError):
        cm._pelvis_world_frames(_rig_points(positions), _rig_spine(spine), mode="nonsense")


def test_the_report_is_carried_out_for_the_diagnostics() -> None:
    positions, spine, _ = posed_body()
    report: dict = {}
    convert(positions, spine, pelvis_report_out=report)
    assert report["status"] == "solved"
    assert report["mode"] == cm.PELVIS_FRAME_SOURCE
    report.clear()
    convert(positions, None, pelvis_report_out=report)
    assert report["status"] == "not_attempted"


def test_diagnostics_serialise_the_new_fields_and_legacy_runs_carry_empty_ones() -> None:
    diagnostics = cm.ReconstructionDiagnostics(
        frame_count=1, subject_count=1, camera_count=4, valid_joint_fraction=1.0,
        constraint_recovered_joint_fraction=0.0,
        frames_deferred_to_exhaustive_association=0, frames_without_association=0,
        median_reprojection_error_px=1.0, p95_reprojection_error_px=2.0,
        maximum_reprojection_error_px=3.0, association_objective_median=0.1,
        interpolated_joint_fraction=0.0, temporally_rejected_subject_frames=0,
        contact_frames=((0, 0),),
    )
    payload = diagnostics.as_dict()
    assert payload["spine_triangulation"] == []
    assert payload["pelvis_frame"] == []


def test_every_candidate_and_control_runs_through_the_one_code_path() -> None:
    """`_pelvis_world_frames` is module level and called by bare name, like
    `_leg_root_offset`, so an instrument substitutes it and runs its controls through the
    identical construction rather than a re-implementation of it."""

    positions, spine, truth = posed_body()
    for mode in ("A_root_to_spine1", "B_hipmid_to_spine1", "C_kabsch_pelvis"):
        quaternions, report = cm._pelvis_world_frames(
            _rig_points(positions), _rig_spine(spine), mode=mode
        )
        assert report["mode"] == mode and quaternions is not None
        error = geodesic_deg(Rotation.from_quat(quaternions).as_matrix(), truth)
        # B is exact by construction here (the fixture is built from B's rest vectors);
        # A and C read the convention difference between the rest vectors, which is what
        # `artifacts/compare/d7-pelvis-frame/synthetic.json`'s clean arm measures.
        if mode == cm.PELVIS_FRAME_SOURCE:
            assert error.max() < 0.01, f"{mode}: worst {error.max():.6f} deg"
        else:
            # the other two read the difference between their own shipped rest vector and
            # the template the fixture was built from -- the convention residual the clean
            # arm of `artifacts/compare/d7-pelvis-frame/synthetic.json` measures.
            assert error.max() < 30.0


def test_a_smoothing_window_is_a_knob_and_it_moves_the_answer() -> None:
    """The window is REPORTED, never banded -- and it must actually do something."""

    positions, spine, _ = posed_body(wobble_m=0.01)
    base, _ = cm._pelvis_world_frames(_rig_points(positions), _rig_spine(spine),
                                      smoothing_frames=0)
    wide, _ = cm._pelvis_world_frames(_rig_points(positions), _rig_spine(spine),
                                      smoothing_frames=9)
    moved = geodesic_deg(Rotation.from_quat(wide).as_matrix(),
                         Rotation.from_quat(base).as_matrix())
    assert moved.max() > 0.05
