"""Unit tests for the I6 surface instrument (`tools/compare/silhouette.py`).

These test the *instrument*, not the fixture: the rasteriser against geometry whose
projected area is known in closed form, the score function against masks whose overlap
is known by construction, the degenerate behaviours the controls rely on, and the
bootstrap's null. Nothing here needs `artifacts/`; the two tests that do are skipped
when the retained fixture is absent, because `artifacts/` is gitignored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "compare"))

from autoanim_gnm.commercial_multiview import CalibratedCamera  # noqa: E402
from silhouette import (  # noqa: E402
    frame_alignment_check, moving_block_bootstrap, rasterise, score, summary,
)

WIDTH, HEIGHT = 200, 200
FX = 100.0


def camera() -> CalibratedCamera:
    """Looking down +Z from the origin in a computer-vision basis: identity rotation."""
    return CalibratedCamera(
        name="unit", width=WIDTH, height=HEIGHT,
        intrinsics=np.asarray(((FX, 0.0, WIDTH / 2), (0.0, FX, HEIGHT / 2), (0.0, 0.0, 1.0))),
        camera_center_world_m=np.zeros(3),
        camera_to_world_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
    )


def square(half: float, depth: float) -> tuple[np.ndarray, np.ndarray]:
    verts = np.array([[-half, -half, depth], [half, -half, depth],
                      [half, half, depth], [-half, half, depth]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, faces


# --------------------------------------------------------------------------- rasteriser

def test_rasterises_a_square_to_its_closed_form_area():
    # A 1 m square at 2 m through f = 100 px spans 100 * 1 / 2 = 50 px on a side,
    # from x = 75 to x = 125 inclusive: cv2 paints the boundary, so 51 x 51 pixels.
    image = rasterise(*square(0.5, 2.0), camera(), (WIDTH, HEIGHT))
    assert abs(int(image.sum()) - 51 * 51) <= 51
    ys, xs = np.nonzero(image)
    assert abs(xs.min() - 75) <= 1 and abs(xs.max() - 125) <= 1
    assert abs(ys.min() - 75) <= 1 and abs(ys.max() - 125) <= 1


def test_area_scales_with_the_inverse_square_of_depth():
    near = rasterise(*square(0.5, 2.0), camera(), (WIDTH, HEIGHT)).sum()
    far = rasterise(*square(0.5, 4.0), camera(), (WIDTH, HEIGHT)).sum()
    assert abs(near / far - 4.0) < 0.2


def test_geometry_behind_the_camera_is_dropped_not_mirrored():
    """The defect this guards: a negative depth flips the projection and paints a ghost."""
    assert not rasterise(*square(0.5, -2.0), camera(), (WIDTH, HEIGHT)).any()


def test_a_silhouette_is_a_union_not_an_even_odd_fill():
    """The bug this guards, and it shipped in the first version of the instrument.

    `cv2.fillPoly` handed N polygons in one call scanline-fills the region bounded by
    all N contours under an EVEN-ODD rule, so overlapping polygons cancel. A closed body
    self-overlaps in every view -- the front of the torso covers the back -- so batching
    silently loses pixels. A hidden square behind a larger one must change nothing.
    """
    front, faces = square(0.5, 2.0)
    behind, _ = square(0.5, 3.0)          # smaller on screen, entirely inside the front
    single = rasterise(front, faces, camera(), (WIDTH, HEIGHT))
    both = rasterise(np.vstack([front, behind]),
                     np.vstack([faces, faces + 4]), camera(), (WIDTH, HEIGHT))
    assert np.array_equal(single, both)

    import cv2
    naive = np.zeros((HEIGHT, WIDTH), np.uint8)
    uv, _ = camera().project(np.vstack([front, behind]))
    cv2.fillPoly(naive, uv[np.vstack([faces, faces + 4])].astype(np.int32), 1)
    assert naive.astype(bool).sum() < single.sum()   # the even-odd fill loses the middle


# -------------------------------------------------------------------------------- score

def test_score_on_a_perfect_match_is_one():
    mask = np.zeros((10, 10), bool)
    mask[2:8, 2:8] = True
    assert score(mask.copy(), mask) == (1.0, 1.0, 1.0)


def test_score_on_disjoint_regions_is_zero():
    a = np.zeros((10, 10), bool); a[0:3, 0:3] = True
    b = np.zeros((10, 10), bool); b[7:10, 7:10] = True
    assert score(a, b) == (0.0, 0.0, 0.0)


def test_an_empty_render_scores_zero_precision_not_nan():
    mask = np.zeros((10, 10), bool)
    mask[2:8, 2:8] = True
    precision, recall, iou = score(np.zeros((10, 10), bool), mask)
    assert (precision, recall, iou) == (0.0, 0.0, 0.0)


def test_dilation_buys_recall_with_precision():
    """The whole reason precision and recall are reported separately, never IoU alone."""
    import cv2
    mask = np.zeros((60, 60), bool)
    mask[10:50, 20:40] = True
    render = np.zeros((60, 60), bool)
    render[15:45, 24:36] = True                       # inside the mask, too small
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    grown = cv2.dilate(render.astype(np.uint8), kernel).astype(bool)
    p0, r0, _ = score(render, mask)
    p1, r1, _ = score(grown, mask)
    assert r1 > r0
    assert p1 < p0


def test_a_bounding_box_billboard_reaches_high_recall_and_poor_precision():
    mask = np.zeros((60, 60), bool)
    mask[10:50, 25:35] = True                          # a tall thin person
    box = np.zeros((60, 60), bool)
    box[10:50, 10:50] = True                           # the camera-facing rectangle
    precision, recall, _ = score(box, mask)
    assert recall == pytest.approx(1.0)
    assert precision < 0.3


# ------------------------------------------------------------------------ the yaw probe

def test_turning_a_mesh_180_degrees_twice_is_the_identity():
    """The transform the front/back probe and the facing control are both built on."""
    rng = np.random.default_rng(0)
    verts = rng.normal(size=(50, 3))
    pelvis = np.array([0.3, -0.2, 0.9])

    def yaw180(v):
        d = v - pelvis
        return np.stack([-d[:, 0], -d[:, 1], d[:, 2]], axis=-1) + pelvis

    assert np.allclose(yaw180(yaw180(verts)), verts)
    assert np.allclose(yaw180(verts)[:, 2], verts[:, 2])       # height is preserved
    assert not np.allclose(yaw180(verts), verts)


# ---------------------------------------------------------------------------- bootstrap

def test_the_bootstrap_null_contains_zero():
    rng = np.random.default_rng(1)
    a = np.cumsum(rng.normal(size=400)) * 0.01 + 0.6      # autocorrelated, like the take
    out = moving_block_bootstrap(a, a.copy(), np.random.default_rng(2))
    assert out["median_difference"] == pytest.approx(0.0)
    lo, hi = out["ci95_of_the_median_difference"]
    assert lo <= 0.0 <= hi


def test_the_bootstrap_sees_a_real_offset():
    rng = np.random.default_rng(3)
    a = np.cumsum(rng.normal(size=400)) * 0.01 + 0.6
    out = moving_block_bootstrap(a, a + 0.2, np.random.default_rng(4))
    assert out["median_difference"] == pytest.approx(-0.2)
    lo, hi = out["ci95_of_the_median_difference"]
    assert hi < 0.0


def test_summary_reports_all_three_metrics():
    rows = np.array([[0.9, 0.5, 0.4], [0.8, 0.6, 0.5], [0.7, 0.7, 0.6]])
    out = summary(rows)
    assert set(out) == {"precision", "recall", "iou"}
    assert out["precision"]["median"] == pytest.approx(0.8)
    assert out["iou"]["median"] == pytest.approx(0.5)


# --------------------------------------------------------------- alignment, synthetic

def _offset(f: int) -> float:
    """A synthetic subject that stays in frame: +-0.5 m is +-25 px at this camera."""
    return 0.5 * float(np.sin(2.0 * np.pi * f / 25.0))


class _FakeMasks:
    """A mask store whose foreground oscillates, so a one-frame lag is detectable."""

    def __init__(self, camera_, shape, verts, faces):
        self.stack = np.stack([
            rasterise(verts + np.array([_offset(f), 0.0, 0.0]), faces, camera_, shape)
            for f in range(150)])

    def get(self, camera_name, tracklet):
        del camera_name, tracklet
        return self.stack


def test_frame_alignment_finds_lag_zero_when_aligned():
    cam, shape = camera(), (WIDTH, HEIGHT)
    verts, faces = square(0.3, 2.0)
    moving = np.stack([verts + np.array([_offset(f), 0.0, 0.0]) for f in range(150)])
    masks = _FakeMasks(cam, shape, verts, faces)
    out = frame_alignment_check({0: (moving, faces), 1: (moving, faces)}, masks,
                                {"unit": {0: 1, 1: 2}}, {"unit": cam}, shape, ("unit",))
    assert out["best_lag"] == 0
    assert out["verdict"] == "aligned"
    assert out["lag0_relative_shortfall"] == 0.0


def test_frame_alignment_detects_a_one_frame_offset():
    cam, shape = camera(), (WIDTH, HEIGHT)
    verts, faces = square(0.3, 2.0)
    shifted = np.stack([verts + np.array([_offset(f + 1), 0.0, 0.0]) for f in range(150)])
    masks = _FakeMasks(cam, shape, verts, faces)
    out = frame_alignment_check({0: (shifted, faces), 1: (shifted, faces)}, masks,
                                {"unit": {0: 1, 1: 2}}, {"unit": cam}, shape, ("unit",))
    assert out["best_lag"] == 1
    assert out["verdict"] == "MISALIGNED"
    # a REAL one-frame offset is decisive, not a fraction of a percent
    assert out["lag0_relative_shortfall"] > 0.05


# ------------------------------------------------------- the report, when it is on disk

REPORT = ROOT / "artifacts/compare/silhouette.json"


@pytest.mark.skipif(not REPORT.exists(), reason="artifacts/ is gitignored; run the instrument")
def test_the_oracle_beats_every_control_and_every_control_fails():
    report = json.loads(REPORT.read_text())
    arms = report["arms"]

    def ious(name):
        return [arms[name][cam][f"subject_{s:02d}"]["iou"]["median"]
                for cam in report["cameras"] for s in (0, 1)]

    oracle = np.array(ious("ORACLE_mamma_mesh"))
    assert (oracle > np.array(ious("ours_delivered"))).all()
    for control in [k for k in arms if k.startswith("control_")]:
        assert (oracle > np.array(ious(control))).all(), control
    # the degenerates a solver actually produces must fall well short, not marginally
    for control in ("control_frozen_pose_tracked", "control_frozen_pose_static",
                    "control_shuffled_subject", "control_billboard"):
        assert np.median(ious(control)) < 0.6 * np.median(oracle), control
    # dilation must trade precision for recall monotonically
    radii = sorted(int(k.split("_")[-1].removesuffix("px"))
                   for k in arms if k.startswith("control_dilated"))
    previous = (1.0, 0.0)
    for radius in radii:
        entry = arms[f"control_dilated_{radius}px"]
        precision = np.median([entry[c][f"subject_{s:02d}"]["precision"]["median"]
                               for c in report["cameras"] for s in (0, 1)])
        recall = np.median([entry[c][f"subject_{s:02d}"]["recall"]["median"]
                            for c in report["cameras"] for s in (0, 1)])
        assert precision < previous[0] and recall > previous[1], radius
        previous = (precision, recall)


@pytest.mark.skipif(not REPORT.exists(), reason="artifacts/ is gitignored; run the instrument")
def test_the_report_declares_its_identity_evidence_alignment_and_blindness():
    report = json.loads(REPORT.read_text())
    assert report["render"]["frame_alignment"]["verdict"] == "aligned"
    assert "depth" in report["blind_to"].lower()
    assert report["standing_rules"]["mamma_ships"] is False
    identity = report["identity"]
    assert "IoU" in identity["method"]["never"]
    for cam, evidence in identity["evidence"].items():
        assert evidence["straight_total"] > 5 * evidence["crossed_total"], cam
        assert len(evidence["per_frame"]) == report["frames"], cam
