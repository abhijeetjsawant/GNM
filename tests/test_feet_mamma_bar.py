"""Unit tests for the I4 feet instrument's geometry, on constructed poses only.

No artifact is read here. Everything is a hand-built pose whose answer is known, which is
what makes these tests able to catch a frame or sign convention that has silently
inverted -- the class of defect that produced this lane's crossed subject map.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "feet"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "compare" / "extractors"))


@pytest.fixture(scope="module")
def feet():
    import mamma_feet_bar

    return mamma_feet_bar


def standing(n: int = 1):
    """Z-up world. Subject faces +x, left is +y. Knee above ankle, hips level."""
    hip_l = np.tile([0.0, 0.1, 0.9], (n, 1))
    hip_r = np.tile([0.0, -0.1, 0.9], (n, 1))
    knee = np.tile([0.0, 0.1, 0.5], (n, 1))
    ankle = np.tile([0.0, 0.1, 0.1], (n, 1))
    return hip_l, hip_r, knee, ankle


def test_shin_frame_is_right_handed_and_anterior(feet):
    hip_l, hip_r, knee, ankle = standing()
    F, conditioning = feet.shin_frames(hip_l, hip_r, knee, ankle)
    m, l, d = F[0, :, 0], F[0, :, 1], F[0, :, 2]
    assert np.allclose(d, [0.0, 0.0, -1.0]), "d must point knee -> ankle, i.e. down"
    assert np.allclose(m, [1.0, 0.0, 0.0]), f"m must be anterior (+x), got {m}"
    assert np.allclose(l, [0.0, -1.0, 0.0]), f"l must be the subject's right (-y), got {l}"
    assert np.isclose(np.linalg.det(F[0]), 1.0), "the frame must be right-handed"
    assert np.allclose(F[0].T @ F[0], np.eye(3), atol=1e-12), "and orthonormal"
    assert np.isclose(conditioning[0], 1.0), "shin perpendicular to the pelvic axis"


def test_degenerate_shin_parallel_to_pelvic_axis_is_flagged(feet):
    """A leg abducted straight out along the pelvic axis: the frame is undefined."""
    hip_l = np.array([[0.0, 0.1, 0.9]])
    hip_r = np.array([[0.0, -0.1, 0.9]])
    knee = np.array([[0.0, 0.3, 0.9]])
    ankle = np.array([[0.0, 0.7, 0.9]])          # shin along +y, parallel to r_hip
    _, conditioning = feet.shin_frames(hip_l, hip_r, knee, ankle)
    assert conditioning[0] < feet.CONDITION_MIN_SIN


def test_foot_square_to_the_shin_reads_ninety_degrees(feet):
    """Neutral standing foot: anterior, level. phi = 90, psi = 0."""
    hip_l, hip_r, knee, ankle = standing()
    F, _ = feet.shin_frames(hip_l, hip_r, knee, ankle)
    foot = ankle + np.array([[0.14, 0.0, 0.0]])          # straight anterior
    f = feet.in_frame(F, ankle, foot)
    rom = feet.range_of_motion(np.repeat(f, 3, axis=0), "L")
    assert rom["dorsi_plantarflexion_phi"]["median_deg"] == pytest.approx(90.0, abs=1e-6)
    assert rom["ab_adduction_psi_medial_positive"]["median_deg"] == pytest.approx(0.0, abs=1e-6)
    assert rom["ankle_angle_shin_to_foot_deg"]["median_deg"] == pytest.approx(90.0, abs=1e-6)


def test_plantarflexion_is_below_ninety_and_dorsiflexion_above(feet):
    hip_l, hip_r, knee, ankle = standing()
    F, _ = feet.shin_frames(hip_l, hip_r, knee, ankle)
    pointed = feet.in_frame(F, ankle, ankle + np.array([[0.10, 0.0, -0.10]]))   # toes down
    raised = feet.in_frame(F, ankle, ankle + np.array([[0.10, 0.0, 0.10]]))     # toes up
    phi = lambda f: np.degrees(np.arctan2(f[0, 0], f[0, 2]))
    assert phi(pointed) == pytest.approx(45.0, abs=1e-6), "plantarflexion < 90"
    assert phi(raised) == pytest.approx(135.0, abs=1e-6), "dorsiflexion > 90"


def test_ab_adduction_sign_is_medial_positive_on_both_sides(feet):
    """A foot swung toward the midline must read POSITIVE psi on the left AND the right."""
    hip_l, hip_r, knee, ankle = standing()
    F, _ = feet.shin_frames(hip_l, hip_r, knee, ankle)
    # subject's left leg: the midline is to its RIGHT, which is -y in world = +l
    medial_left = feet.in_frame(F, ankle, ankle + np.array([[0.14, -0.05, 0.0]]))
    # subject's right leg: the midline is to its LEFT, +y in world = -l
    medial_right = feet.in_frame(F, ankle, ankle + np.array([[0.14, 0.05, 0.0]]))
    left = feet.range_of_motion(np.repeat(medial_left, 3, axis=0), "L")
    right = feet.range_of_motion(np.repeat(medial_right, 3, axis=0), "R")
    assert left["ab_adduction_psi_medial_positive"]["median_deg"] > 5.0
    assert right["ab_adduction_psi_medial_positive"]["median_deg"] > 5.0


def test_circular_statistics_wrap_across_the_branch_cut(feet):
    values = np.array([179.0, -179.0, 180.0, -178.0])
    stats = feet.circular_stats(values)
    assert abs(abs(stats["circular_mean_deg"]) - 180.0) < 2.0
    assert stats["range_deg"] < 5.0, "a 4-degree cluster must not read as a 358-degree range"


def test_kabsch_recovers_a_known_rotation_and_reports_conditioning(feet):
    rng = np.random.default_rng(0)
    target = feet.unit(rng.normal(size=(200, 3)))          # isotropic: well conditioned
    angle = np.radians(37.0)
    known = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                      [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    source = target @ known.T
    R, conditioning = feet.kabsch_directions(source, target)
    assert feet.rotation_magnitude_deg(R) == pytest.approx(37.0, abs=1e-6)
    assert conditioning > 0.5, "an isotropic cloud identifies all three axes"


def test_kabsch_is_UNIDENTIFIED_for_a_constant_source_which_is_why_G1_uses_the_mean(feet):
    """The defect this instrument was corrected for: a constant direction pins no rotation."""
    constant = np.tile([1.0, 0.0, 0.0], (100, 1))
    rng = np.random.default_rng(1)
    target = feet.unit(np.tile([1.0, 0.0, 0.0], (100, 1)) + 0.05 * rng.normal(size=(100, 3)))
    _, conditioning = feet.kabsch_directions(constant, target)
    assert conditioning < 1e-9, "rank-1 cross-covariance: the rotation is arbitrary"
    block, _ = feet.compare(constant, target)
    # the gated offset is the mean-direction angle, which IS identified and near zero
    assert block["mean_direction_offset_deg"] < 2.0
    assert block["G1_offset_within_90deg"] is True


def test_G1_rejects_a_mirrored_anterior_axis_that_kabsch_would_absorb(feet):
    hip_l, hip_r, knee, ankle = standing(120)
    F, _ = feet.shin_frames(hip_l, hip_r, knee, ankle)
    swing = np.linspace(-0.05, 0.05, 120)[:, None] * np.array([0.0, 0.0, 1.0])
    reference = feet.in_frame(F, ankle, ankle + np.array([[0.14, 0.0, 0.0]]) + swing)
    mirrored = reference * np.array([-1.0, 1.0, 1.0])
    block, residual = feet.compare(mirrored, reference)
    assert block["G1_offset_within_90deg"] is False, "a posterior foot must fail G1"
    assert block["mean_direction_offset_deg"] > 150.0
    # and the point of the band: the offset-removed SPREAD does not catch it
    assert np.median(residual) < 1.0


def test_adjudicate_requires_the_worst_block_and_counts_a_G1_failure_as_rejected(feet):
    blocks = {
        "ours_delivered": {"G1_offset_within_90deg": True, "mean_direction_offset_deg": 12.0},
        "ours_triangulated_toebase": {"G1_offset_within_90deg": True,
                                      "mean_direction_offset_deg": 11.0},
        "CONTROL_welded_to_shin_zero_articulation": {"G1_offset_within_90deg": True,
                                                     "mean_direction_offset_deg": 0.0},
        "CONTROL_mirrored_anterior_axis": {"G1_offset_within_90deg": False,
                                           "mean_direction_offset_deg": 170.0},
    }
    boot = {
        "10": {c: {"P_beats": {"CONTROL_welded_to_shin_zero_articulation": 0.99,
                               "CONTROL_mirrored_anterior_axis": 0.10}}
               for c in ("ours_delivered", "ours_triangulated_toebase")},
        "30": {c: {"P_beats": {"CONTROL_welded_to_shin_zero_articulation": 0.80,
                               "CONTROL_mirrored_anterior_axis": 0.10}}
               for c in ("ours_delivered", "ours_triangulated_toebase")},
    }
    out = feet.adjudicate(blocks, boot)["ours_delivered"]
    assert out["G2_beats_welded_P_min_over_blocks"] == 0.80, "the WORST block, not the best"
    assert out["G2_pass"] is False
    assert out["G3_controls"]["CONTROL_mirrored_anterior_axis"]["control_rejected"] is True, \
        "a control that fails G1 is rejected even though it wins on spread"
    assert "NOT SHOWN" in out["verdict"]


def test_extractor_is_empty_without_a_report(tmp_path, monkeypatch):
    import i4_feet

    monkeypatch.setattr(i4_feet, "REPORT", tmp_path / "absent.json")
    assert i4_feet.x_feet_bar({}) == ([], [])


def test_extractor_figures_carry_the_mamma_reference_not_our_own(tmp_path, monkeypatch):
    """The retired instrument's reference is ours on both sides. These must not mix."""
    import i4_feet

    if not i4_feet.REPORT.exists():
        pytest.skip("report not on disk in this checkout")
    figures, controls = i4_feet.x_feet_bar({})
    assert figures and controls
    assert all(f["reference"] is i4_feet.REF for f in figures)
    assert any("ORACLE" in c["label"] for c in controls), "the gate needs its oracle arm"
    assert any("must fail" in c["label"] for c in controls)
