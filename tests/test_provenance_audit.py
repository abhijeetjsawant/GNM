"""Step I8: the provenance audit and the thorax window sweep, tested for their claims.

These do not assert `provenance.py` exits zero. It exits ONE today, deliberately --
`THORAX_SMOOTHING_FRAMES` WAS the known leak (re-selected on synthetic truth 2026-09-02) and the script is meant to gate a delivery
build on exactly that. What is tested is the classification: that the rule finds the leak
it is supposed to find, that a declared item is declared rather than condemned, that
`unknown` stays honest, and that the sweep's own scoring functions behave.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    for extra in ("src", "scripts", "workers/commercial_multiview"):
        path = str(ROOT / extra)
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def provenance():
    return _load("i8_provenance_tool", "tools/compare/provenance.py")


@pytest.fixture(scope="module")
def sweep():
    return _load("i8_thorax_sweep", "tools/head/thorax_window_sweep.py")


# ------------------------------------------------------------------------ the audit

def test_the_former_leak_is_now_selected_on_synthetic_truth(provenance):
    # I8 found THORAX_SMOOTHING_FRAMES chosen on the MAMMA oracle arm; on 2026-09-02 it was
    # re-selected on synthetic truth and the constant changed (15 -> 9). The entry must say
    # where it comes from NOW and keep the history of where it came from.
    entry = provenance.CURATED["THORAX_SMOOTHING_FRAMES"]
    assert entry["provenance"] == provenance.SYNTHETIC
    assert "thorax_window_sweep" in entry["evidence"]
    assert "oracle" in entry["evidence"]  # the history is kept, not erased


def test_the_scan_reaches_the_delivery_entry_points(provenance):
    _, live = provenance.scan()
    for name in ("reconstruct_multiview", "positions_to_body_track",
                 "_solve_head_for_subject", "_thorax_frames", "triangulate_point"):
        assert name in live


def test_the_manifest_is_clean_and_finds_the_declared_rig(provenance):
    scanned, _ = provenance.scan()
    manifest = provenance.classify(scanned)
    leaks = [e for e in manifest if e["provenance"] == provenance.MAMMA_DERIVED]
    assert leaks == [], [e["name"] for e in leaks]  # the manifest is clean since 2026-09-02
    thorax = next(e for e in manifest if e["name"] == "THORAX_SMOOTHING_FRAMES")
    assert thorax["provenance"] == provenance.SYNTHETIC
    declared = [e for e in provenance.DATA_INPUTS
                if e["provenance"] == provenance.DECLARED_MAMMA]
    paths = " ".join(e["path"] for e in declared)
    assert "camera-rig.json" in paths


def test_the_declared_rig_is_declared_and_not_a_leak(provenance):
    rig = next(e for e in provenance.DATA_INPUTS if e["name"] == "camera rig calibration")
    assert rig["provenance"] == provenance.DECLARED_MAMMA
    assert rig["provenance"] != provenance.MAMMA_DERIVED
    assert rig["declared"] is True
    assert rig["retired_by"]


def test_an_uncurated_constant_becomes_unknown_never_a_guess(provenance):
    invented = [{"name": "SOMETHING_NOBODY_WROTE_DOWN", "kind": "module constant",
                 "file": "nowhere:1", "value": "42"}]
    out = provenance.classify(invented)
    entry = next(e for e in out if e["name"] == "SOMETHING_NOBODY_WROTE_DOWN")
    assert entry["provenance"] == provenance.UNKNOWN
    assert "not curated" in entry["evidence"]


def test_every_curated_entry_carries_evidence(provenance):
    for name, entry in provenance.CURATED.items():
        assert entry.get("evidence"), name
        assert entry["provenance"] in (
            provenance.CLEAN | {provenance.UNKNOWN, provenance.DECLARED_MAMMA,
                                provenance.MAMMA_DERIVED}), name


def test_unknown_entries_carry_their_evidence_and_claim_nothing(provenance):
    """An `unknown` still has to say WHAT was searched, and must not carry a selection."""
    for name, entry in provenance.CURATED.items():
        if entry["provenance"] != provenance.UNKNOWN:
            continue
        assert len(entry["evidence"]) > 60, name          # not a bare shrug
        assert "selected_against" not in entry, name      # only a leak has one


def test_the_unknown_list_is_pinned_so_a_new_unaudited_constant_fails(provenance):
    """The audited-as-unknown set, snapshotted.

    A constant added to the delivery path with no curated provenance lands in `unknown`
    and breaks this test, which is the point: the audit has to notice growth, not just
    describe the past. Removing one because it was traced is also a deliberate edit.
    """
    scanned, _ = provenance.scan()
    manifest = provenance.classify(scanned)
    unknown = sorted({e["name"] for e in manifest
                      if e["provenance"] == provenance.UNKNOWN})
    assert unknown == [
        "0.72 * torso_up (clavicle origin)",
        "BOX_PADDING",
        "DEFAULT_WEIGHTS grid extent",
        "ambiguity_margin_px",
        "ambiguity_ratio",
        "foot contact envelope (capture path)",
        "ground_band_m",
        "inlier_threshold_px",
        "length_weight",
        "maximum_epipolar_px",
        "maximum_root_correction_m",
        "minimum_contact_frames",
        "minimum_shared_joints",
        "neck_sigma_m",
        "smooth_weight",
        "template_prior",
        "velocity_threshold_m_per_s",
    ]


# ------------------------------------------------------------------------ the sweep

def test_geodesic_normalises_before_comparing(sweep):
    """CLAUDE.md: an unnormalised scaled matrix reads 0.00 deg apart. It must not here."""
    identity = np.eye(3)[None]
    half = np.pi / 4.0
    turned = np.asarray([[[np.cos(half), -np.sin(half), 0.0],
                          [np.sin(half), np.cos(half), 0.0],
                          [0.0, 0.0, 1.0]]])
    assert sweep.geodesic_deg(identity, turned)[0] == pytest.approx(45.0, abs=1e-6)
    # The same rotation scaled 100x -- the failure mode the gotcha describes.
    assert sweep.geodesic_deg(identity * 100.0, turned * 100.0)[0] == pytest.approx(45.0, abs=1e-6)


def test_the_sigma_conversions_from_our_own_instruments_agree(sweep):
    """Two independent own-detector figures must land on the same per-axis sigma."""
    a = sweep.sigma_from_reprojection(sweep.REPROJECTION_MEDIAN_PX, 4)
    b = sweep.sigma_from_epipolar(sweep.EPIPOLAR_ONE_SIDED_PX[2])
    assert 3.0 < a < 3.3
    assert 3.1 < b < 3.4
    assert abs(a - b) < 0.25
    assert abs(sweep.NOISE_SIGMA_PX - 0.5 * (a + b)) < 0.15


def test_the_heavy_tail_arm_is_median_matched_and_heavier(sweep):
    rng = np.random.default_rng(0)
    heavy = np.asarray([sweep.heavy_tail_magnitude(rng, sweep.NOISE_SIGMA_PX)
                        for _ in range(40000)])
    rayleigh = np.linalg.norm(rng.normal(0.0, sweep.NOISE_SIGMA_PX, size=(40000, 2)), axis=1)
    assert np.median(heavy) == pytest.approx(np.median(rayleigh), rel=0.05)
    assert np.percentile(heavy, 95) > 2.0 * np.percentile(rayleigh, 95)


def test_lag_recovers_a_known_shift(sweep):
    t = np.linspace(0.0, 8.0 * np.pi, 300)
    truth = 40.0 * np.sin(t)
    assert sweep.lag_frames(truth, np.roll(truth, 3)) == pytest.approx(3.0, abs=0.35)
    assert sweep.lag_frames(truth, truth) == pytest.approx(0.0, abs=0.2)


def test_window_3_at_polyorder_2_is_the_identity(sweep):
    """A parabola through three points is exact, so window 3 must equal no smoothing.

    This is the harness's own self-check: if these ever differ, the sweep is not calling
    the shipped `_thorax_frames` and every other cell is suspect.
    """
    from autoanim_gnm.commercial_multiview import JOINT_INDEX, _thorax_frames
    rng = np.random.default_rng(7)
    positions = np.zeros((40, len(JOINT_INDEX), 3))
    positions[:, JOINT_INDEX["root"]] = (0.0, 0.0, 1.0)
    positions[:, JOINT_INDEX["neck"]] = (0.0, 0.0, 1.5)
    angle = np.linspace(0.0, 1.0, 40) + rng.normal(0.0, 0.02, 40)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        for joint, height in ((f"{side}_shoulder", 1.45), (f"{side}_hip", 1.0)):
            positions[:, JOINT_INDEX[joint], 0] = sign * 0.2 * np.cos(angle)
            positions[:, JOINT_INDEX[joint], 1] = sign * 0.2 * np.sin(angle)
            positions[:, JOINT_INDEX[joint], 2] = height
    none = _thorax_frames(positions, smoothing_frames=0)
    three = _thorax_frames(positions, smoothing_frames=3)
    assert float(np.max(sweep.geodesic_deg(none, three))) < 1e-6


def test_a_window_wider_than_the_take_is_not_a_freeze(sweep):
    """`_thorax_frames` returns the UNSMOOTHED frame once the window exceeds the take.

    This is why the freeze control is built as the take's mean frame and not as a very
    wide window -- the brief assumed the opposite.
    """
    from autoanim_gnm.commercial_multiview import JOINT_INDEX, _thorax_frames
    positions = np.zeros((20, len(JOINT_INDEX), 3))
    positions[:, JOINT_INDEX["root"]] = (0.0, 0.0, 1.0)
    positions[:, JOINT_INDEX["neck"]] = (0.0, 0.0, 1.5)
    angle = np.linspace(0.0, 1.0, 20)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        for joint, height in ((f"{side}_shoulder", 1.45), (f"{side}_hip", 1.0)):
            positions[:, JOINT_INDEX[joint], 0] = sign * 0.2 * np.cos(angle)
            positions[:, JOINT_INDEX[joint], 1] = sign * 0.2 * np.sin(angle)
            positions[:, JOINT_INDEX[joint], 2] = height
    wide = _thorax_frames(positions, smoothing_frames=45)
    none = _thorax_frames(positions, smoothing_frames=0)
    assert float(np.max(sweep.geodesic_deg(wide, none))) < 1e-9
    spread = sweep.yaw_deg(wide)
    assert spread.max() - spread.min() > 1.0     # it still moves; it is not frozen


def test_the_joint_map_is_soma_order_not_joint_index_order(sweep):
    """The plan brief had this backwards; `to_19` exists because of it."""
    from autoanim_gnm.commercial_multiview import JOINT_INDEX
    from soma77_pose import SOMA77_TO_AUTOANIM
    take = np.arange(77 * 3, dtype=float).reshape(1, 77, 3)
    out = sweep.to_19(take)
    assert out.shape == (1, len(JOINT_INDEX), 3)
    assert np.allclose(out[0, JOINT_INDEX["neck"]], take[0, SOMA77_TO_AUTOANIM["neck"]])
    for name in ("left_ear", "right_ear"):
        assert not np.isfinite(out[0, JOINT_INDEX[name]]).any()


def test_the_mamma_arm_transcription_reads_both_performers(sweep):
    """The docstring's own table, both columns -- they do not say the same thing.

    Performer 1's p95 has an interior minimum at 15; performer 0's falls monotonically to
    31, the widest window it tested. Asserting only one column is how the earlier reading
    of this table went wrong.
    """
    arm = sweep.MAMMA_ORACLE_SWEEP_REPORTS_NEVER_SELECTS
    windows = [row["window"] for row in arm["rows"]]
    for performer, expected in ((0, 31), (1, 15)):
        p95 = [row["p95_deg"][performer] for row in arm["rows"]]
        assert windows[p95.index(min(p95))] == expected, performer
        median = [row["median_deg"][performer] for row in arm["rows"]]
        # The shape that DOES agree with ours: median prefers narrower than p95.
        assert windows[median.index(min(median))] <= expected, performer
    p95_zero = [row["p95_deg"][0] for row in arm["rows"]]
    assert p95_zero == sorted(p95_zero, reverse=True)      # monotone: no interior optimum
    p95_one = [row["p95_deg"][1] for row in arm["rows"]]
    assert p95_one != sorted(p95_one, reverse=True)        # interior optimum, at 15
    assert arm["its_own_p95_argmin_by_performer"] == {"0": 31, "1": 15}
    assert arm["its_stated_choice"] == 15


# ------------------------------------------------------------------------ the extractor

def test_the_extractor_stub_matches_the_ladder_figure_shape():
    """`x_*` returns (figures, controls), each a list of `fig(...)`-shaped dicts."""
    path = str(ROOT / "tools/compare")
    if path not in sys.path:
        sys.path.insert(0, path)
    from extractors.i8_provenance import fig, x_provenance

    keys = set(fig("label", 1.0, "unit", "reference"))
    assert keys == {"key", "label", "value", "unit", "reference", "better", "note"}
    figures, controls = x_provenance({})
    if not figures and not controls:
        pytest.skip("reports not on disk; run provenance.py and thorax_window_sweep.py")
    for entry in figures + controls:
        assert set(entry) == keys
        assert entry["reference"]
    named = {e["key"] for e in figures}
    assert {"leaks", "unknown"} <= named


def test_the_extractor_never_puts_the_mamma_arm_on_a_scored_axis():
    """The MAMMA arm is carried as a COUNT with its own reference, never as a score."""
    path = str(ROOT / "tools/compare")
    if path not in sys.path:
        sys.path.insert(0, path)
    from extractors.i8_provenance import COUNT, x_provenance

    _, controls = x_provenance({})
    if not controls:
        pytest.skip("reports not on disk")
    arm = next((c for c in controls if c["key"] == "thorax_mamma_arm"), None)
    if arm is None:
        pytest.skip("sweep report not on disk")
    assert arm["better"] == COUNT
    assert "DIFFERENT" in arm["reference"]
