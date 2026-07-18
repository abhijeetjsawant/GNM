from pathlib import Path

import cv2
import numpy as np
import pytest

from autoanim_gnm.gnm_adapter import EXPECTED, GNMAdapter
from autoanim_gnm.render import MeshRenderer
from autoanim_gnm.rig import ControlRig
from autoanim_gnm.semantic_decoder import EXPRESSION_NAMES, ExpressionDecoder
from autoanim_gnm.serialization import write_json, write_npz


def test_asset_runtime_truth(adapter: GNMAdapter) -> None:
    assert adapter.model.num_vertices == EXPECTED["vertices"]
    assert adapter.triangles.shape == (EXPECTED["triangles"], 3)
    assert adapter.identity_dim == EXPECTED["identity"]
    assert adapter.expression_dim == EXPECTED["expression"]
    assert adapter.model.version.value == "3.0"


def test_decoder_graph_shape_determinism_and_golden(decoder: ExpressionDecoder) -> None:
    actual = decoder.prototype("surprise")
    expected_prefix = np.array(
        [
            0.69523895,
            -0.43739477,
            0.27803870,
            0.29269320,
            -0.15550110,
            -0.04692641,
            -0.30424988,
            -0.17245385,
        ],
        dtype=np.float32,
    )
    assert actual.shape == (383,)
    np.testing.assert_allclose(actual[:8], expected_prefix, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(actual, decoder.prototype("surprise"))
    assert len(EXPRESSION_NAMES) == 20


def test_decoder_rejects_bad_shapes(decoder: ExpressionDecoder) -> None:
    with pytest.raises(ValueError):
        decoder.decode(np.zeros(63), np.zeros(20))
    with pytest.raises(KeyError):
        decoder.prototype("not-real")


def test_compact_landmarks_match_official_gnm(adapter: GNMAdapter) -> None:
    rng = np.random.default_rng(42)
    identity = np.zeros(adapter.identity_dim, dtype=np.float32)
    expression = np.zeros(adapter.expression_dim, dtype=np.float32)
    identity[:20] = rng.normal(0, 0.3, 20)
    expression[200:240] = rng.normal(0, 0.1, 40)
    compact = (
        adapter.compact_template
        + np.einsum("i,ilc->lc", identity, adapter.compact_identity_basis)
        + np.einsum("i,ilc->lc", expression, adapter.compact_expression_basis)
    )
    official = adapter.landmarks(identity=identity, expression=expression)
    np.testing.assert_allclose(compact, official, atol=1e-6, rtol=1e-6)


def test_region_masks_and_bounds(rig: ControlRig) -> None:
    for cue in "XABCDEFGH":
        control = rig.viseme(cue)
        assert control.shape == (383,)
        assert np.max(np.abs(control)) <= 3.0
        assert not np.any(control[:200])
        assert not np.any(control[382:])
        if cue != "H":
            assert not np.any(control[350:382])
    assert np.any(rig.viseme("H")[350:382])

    for name in ("neutral", "joy", "sad", "anger", "fear", "disgust", "surprise", "contempt"):
        control = rig.emotion(name)
        assert not np.any(control[350:])
        assert np.max(np.abs(control)) <= 3.0

    _, clipped = rig.compose(np.full(383, 3.0), np.full(383, 3.0))
    assert clipped


def test_manual_emotion_preserves_viseme_motion(rig: ControlRig) -> None:
    viseme = rig.viseme("D")
    emotion = rig.emotion("anger")
    composed, clipped = rig.compose(viseme, emotion)
    assert not clipped
    assert np.any(composed[:200])
    np.testing.assert_allclose(
        composed[200:350] - emotion[200:350],
        viseme[200:350],
        atol=1e-6,
        rtol=0,
    )
    assert not np.any(composed[350:])


def test_real_gnm_viseme_geometry(rig: ControlRig) -> None:
    metrics = {cue: rig.geometry_metrics(rig.viseme(cue)) for cue in "XABCDEFGH"}
    aperture = {cue: value["mouth_aperture"] for cue, value in metrics.items()}
    assert aperture["D"] > 1.20 * aperture["C"]
    assert aperture["C"] > 1.20 * aperture["B"]
    assert aperture["B"] > 1.05 * aperture["X"]
    assert aperture["A"] <= aperture["X"]
    assert aperture["G"] <= aperture["X"]
    assert metrics["F"]["mouth_width"] < 0.97 * metrics["C"]["mouth_width"]
    assert metrics["H"]["tongue_motion"] > 0.0005


def test_mesh_topology_finiteness_and_obj_export(adapter: GNMAdapter, rig: ControlRig, tmp_path: Path) -> None:
    expression = rig.viseme("D")
    vertices = adapter.mesh(expression=expression)
    assert vertices.shape == (17_821, 3)
    assert np.isfinite(vertices).all()
    assert adapter.triangles.min() >= 0
    assert adapter.triangles.max() < len(vertices)
    output = adapter.export_obj(tmp_path / "head.obj", vertices)
    assert output.stat().st_size > 1_000_000
    with output.open(encoding="utf-8") as handle:
        assert handle.readline().startswith("# AutoAnim GNM Head")


def test_renderer_produces_visible_real_mesh(adapter: GNMAdapter, rig: ControlRig, tmp_path: Path) -> None:
    expression = rig.viseme("D")
    vertices = adapter.mesh(expression=expression)
    landmarks = adapter.landmarks(expression=expression)
    renderer = MeshRenderer(adapter)
    image = renderer.render(vertices, landmarks)
    assert image.shape == (640, 640, 3)
    assert image.dtype == np.uint8
    assert np.count_nonzero(image) > 20_000
    path = renderer.save_png(tmp_path / "preview.png", vertices, landmarks)
    decoded = cv2.imread(str(path))
    np.testing.assert_array_equal(decoded, image)


def test_atomic_json_and_npz_exports(tmp_path: Path, rig: ControlRig) -> None:
    expression = rig.viseme("F")
    json_path = write_json(tmp_path / "metrics.json", {"finite": True, "cue": "F"})
    npz_path = write_npz(
        tmp_path / "controls.npz",
        expression=expression[None, :],
        fps=np.asarray(30, dtype=np.int32),
    )
    assert json_path.read_text(encoding="utf-8").endswith("\n")
    with np.load(npz_path, allow_pickle=False) as values:
        np.testing.assert_array_equal(values["expression"], expression[None, :])
        assert int(values["fps"]) == 30
    assert not list(tmp_path.glob("*.tmp"))
