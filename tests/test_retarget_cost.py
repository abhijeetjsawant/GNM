"""Step I1's instrument, tested where it needs no artifacts on disk.

Every test here builds its own body, so they run without `artifacts/` and without
MAMMA. The two claims that need real files -- the oracle's 36-47 mm and arm B --
are asserted by the instrument itself when it runs and are skipped here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))


def _load_instrument():
    spec = importlib.util.spec_from_file_location(
        "retarget_cost", ROOT / "tools/swap-harness/retarget_cost.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rc = _load_instrument()
cm = rc.cm


@pytest.fixture(scope="module")
def synthetic_observations():
    """A plausible 19-joint observation set: the canonical rest body, jittered."""
    from autoanim_gnm.body import DETAILED_HUMANOID, forward_kinematics_positions

    frames = 12
    quats = np.zeros((frames, len(DETAILED_HUMANOID.joints), 4))
    quats[..., 3] = 1.0
    fk = forward_kinematics_positions(np.zeros((frames, 3)), quats, skeleton=DETAILED_HUMANOID)
    landmarks_y_up = rc.landmarks_from_fk(np.asarray(fk, dtype=np.float64))
    rng = np.random.default_rng(7)
    landmarks_y_up = landmarks_y_up + 0.02 * rng.standard_normal(landmarks_y_up.shape)
    return rc.Z_UP_FROM_Y_UP(landmarks_y_up)


@pytest.fixture(scope="module")
def solved(synthetic_observations):
    track = rc.solve(synthetic_observations)
    return track, rc.fk_of(track)


def test_converter_rejects_nan_in_the_unmapped_joints(synthetic_observations):
    """The execution plan claimed it tolerates them. It does not, and arm B's
    adapter exists because of that."""
    tolerated, message = rc.nan_tolerance_probe(synthetic_observations)
    assert tolerated is False
    assert "CommercialMultiviewError" in message


def test_adapter_filler_is_inert(synthetic_observations):
    """The four joints PAIRS does not cover must not reach the solve."""
    pred = np.zeros((len(synthetic_observations), 127, 3))
    for name, mi in rc.PAIRS.items():
        pred[:, mi] = synthetic_observations[:, cm.JOINT_INDEX[name]]
    assert rc.filler_is_inert(pred) is True


def test_round_trip_oracle_is_exact_on_the_legs(solved, synthetic_observations):
    """A body with canonical proportions BY CONSTRUCTION must come back unchanged
    below the waist. This is the arm the whole instrument rests on."""
    _, fk = solved
    synth = rc.landmarks_from_fk(fk)
    fk_again = rc.retarget_then_fk(rc.Z_UP_FROM_Y_UP(synth))
    err = rc.score(fk_again, synth)
    for name in rc.GROUPS["legs"] + rc.GROUPS["torso"]:
        # exact to floating point: the real take reports 0.00 mm to two decimals.
        assert float(np.median(err[name])) < 1e-3, name


def test_converter_rotations_depend_on_bone_lengths_only_through_the_clavicle(synthetic_observations):
    """Until D2 (2026-09-02) this asserted that a re-solve on a sized skeleton returned
    bit-identical rotations, because every rotation aimed a rest DIRECTION and sizing
    does not turn one. D2 measures the clavicle from the sized rig's OWN shoulder origin,
    which sizing does move, so the clavicle chain -- and nothing else -- now depends on
    the skeleton it is solved on. The scoreboard's replayed sized arm (rung 11) is
    therefore no longer equivalent to a re-solve; both are reported there."""
    from sized_skeleton import sized_skeleton
    from autoanim_gnm.body import DETAILED_HUMANOID

    skel, _ = sized_skeleton(DETAILED_HUMANOID, synthetic_observations)
    canonical = rc.solve(synthetic_observations)
    sized = rc.solve(synthetic_observations, skel)
    a = np.asarray(canonical.local_rotations_xyzw)
    b = np.asarray(sized.local_rotations_xyzw)
    chain = {DETAILED_HUMANOID.index(f"{side}{part}")
             for side in ("Left", "Right") for part in ("Shoulder", "UpperArm", "LowerArm", "Hand")}
    others = [i for i in range(a.shape[1]) if i not in chain]
    assert np.allclose(a[:, others], b[:, others], atol=1e-12)
    assert not np.allclose(a[:, sorted(chain)], b[:, sorted(chain)], atol=1e-6), (
        "the clavicle chain must move with the skeleton it is solved on since D2")
    # Until D3 (2026-09-03) this asserted the root translation was independent of the
    # skeleton. D3 put the per-performer rest into the root formula
    # (`pelvis - rest["Hips"] - R_hips . mid(leg rests)`), so the ROOT moves with the
    # rest by construction and the invariant is one level down: D2b's claim that the
    # LEG-ROOT MIDPOINT sits on the captured hip midpoint, on whichever skeleton the
    # track is solved. (Rewritten 2026-09-06; it had been red since D3.)
    from autoanim_gnm.body import forward_kinematics_positions
    ref_y = rc.Y_UP_FROM_Z_UP(synthetic_observations)
    hip_mid = 0.5 * (ref_y[:, rc.cm.JOINT_INDEX["left_hip"]] + ref_y[:, rc.cm.JOINT_INDEX["right_hip"]])
    for track, skeleton in ((canonical, DETAILED_HUMANOID), (sized, skel)):
        fk = forward_kinematics_positions(
            np.asarray(track.root_translation_m, np.float64),
            np.asarray(track.local_rotations_xyzw, np.float64), skeleton=skeleton)
        roots_mid = 0.5 * (fk[:, skeleton.index("LeftUpperLeg")] + fk[:, skeleton.index("RightUpperLeg")])
        # the foot-contact projection may hoist the root vertically after the solve;
        # horizontally the leg roots must sit on the captured hips on both skeletons
        assert np.allclose(roots_mid[:, [0, 2]], hip_mid[:, [0, 2]], atol=1e-6), (
            "the leg-root midpoint left the captured hip midpoint on this skeleton")


def test_copy_through_passes_the_position_score_and_fails_integrity(synthetic_observations, solved):
    """The degenerate the positional figure cannot reject, and what does reject it."""
    track, fk = solved
    ref_y = rc.Y_UP_FROM_Z_UP(synthetic_observations)
    copied = rc.copy_through_fk(synthetic_observations)
    err = rc.score(copied, ref_y)
    assert max(float(np.median(v)) for v in err.values()) < 1e-9      # scores perfectly
    degenerate = rc.integrity(copied, None)
    honest = rc.integrity(fk, track)
    assert degenerate["rotations_present"] is False
    assert degenerate["bone_length_std_mm_max"] > 1.0                 # bones wander
    assert honest["bone_length_std_mm_max"] == pytest.approx(0.0, abs=1e-6)
    assert honest["bone_length_dev_from_rest_mm_max"] == pytest.approx(0.0, abs=1e-6)


def test_translation_only_alignment_sees_a_facing_error(solved, synthetic_observations):
    ref_y = rc.Y_UP_FROM_Z_UP(synthetic_observations)
    _, fk = solved
    base = rc.score(fk, ref_y)
    yawed = rc.score(rc.yaw180(fk), ref_y)
    for name in rc.GROUPS["legs"]:
        assert float(np.median(yawed[name])) > float(np.median(base[name])) + 100.0


def test_a_rotational_alignment_would_hide_the_same_facing_error(solved, synthetic_observations):
    """Why the instrument removes translation only: Kabsch absorbs the yaw exactly."""
    ref_y = rc.Y_UP_FROM_Z_UP(synthetic_observations)
    _, fk = solved
    base = rc.score_procrustes(fk, ref_y)
    yawed = rc.score_procrustes(rc.yaw180(fk), ref_y)
    for name in base:
        assert float(np.median(yawed[name])) == pytest.approx(float(np.median(base[name])), abs=1e-6)


@pytest.mark.parametrize("mangle", ["permute_labels", "swap_left_right"])
def test_mislabelled_input_must_fail(synthetic_observations, solved, mangle):
    ref_y = rc.Y_UP_FROM_Z_UP(synthetic_observations)
    _, fk = solved
    base = rc.score(fk, ref_y)
    bad = rc.score(rc.retarget_then_fk(getattr(rc, mangle)(synthetic_observations)), ref_y)
    pooled_base = float(np.median(np.concatenate(list(base.values()))))
    pooled_bad = float(np.median(np.concatenate(list(bad.values()))))
    assert pooled_bad > pooled_base + 100.0


def test_block_bootstrap_pairs_both_arms_on_the_same_draws(solved, synthetic_observations):
    """A margin quoted on 150 correlated frames is only meaningful if the two arms
    are resampled together; identical arms must return a margin of exactly zero."""
    ref_y = rc.Y_UP_FROM_Z_UP(synthetic_observations)
    _, fk = solved
    err = rc.score(fk, ref_y)
    rng = np.random.default_rng(1)
    margin = rc.block_bootstrap_margin(err, err, "legs", rng)
    assert margin["median_mm"] == 0.0
    assert margin["ci95_mm"] == [0.0, 0.0]
