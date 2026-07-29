from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from workers.gem_x.prepare_cpu_vitpose import (
    CpuVitPoseRunner,
    flip_heatmaps_soma77,
    infer_vitpose,
    keypoints_from_heatmaps,
    vitpose_preprocess,
)


def test_vitpose_preprocess_matches_gem_x_color_and_shape() -> None:
    frames = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    frames[0, ...] = [10, 20, 30]
    boxes = torch.tensor([[15.5, 15.5, 31.0]])

    result = vitpose_preprocess(frames, boxes)

    expected_rgb = np.array([30, 20, 10], dtype=np.float32) / 255.0
    expected = (expected_rgb - np.array([0.485, 0.456, 0.406])) / np.array(
        [0.229, 0.224, 0.225]
    )
    assert result.shape == (1, 3, 256, 256)
    assert result.dtype == np.float32
    assert result[0, :, 128, 128] == pytest.approx(expected, abs=1e-5)


def test_vitpose_preprocess_rejects_frame_box_mismatch() -> None:
    with pytest.raises(ValueError, match="one .* box per frame"):
        vitpose_preprocess(
            np.zeros((2, 16, 16, 3), dtype=np.uint8),
            torch.tensor([[8.0, 8.0, 16.0]]),
        )


def test_flip_heatmaps_reverses_width_and_swaps_soma_pair() -> None:
    heatmaps = np.zeros((1, 77, 2, 3), dtype=np.float32)
    heatmaps[0, 9, 0, 0] = 3.0
    heatmaps[0, 10, 1, 2] = 7.0

    flipped = flip_heatmaps_soma77(heatmaps)

    assert flipped[0, 10, 0, 2] == 3.0
    assert flipped[0, 9, 1, 0] == 7.0


def test_keypoint_decode_uses_gem_x_affine_coordinates() -> None:
    heatmaps = np.zeros((1, 77, 4, 4), dtype=np.float32)
    heatmaps[:, :, 1, 2] = 9.0

    points, confidence = keypoints_from_heatmaps(
        heatmaps,
        centers=np.array([[50.0, 60.0]], dtype=np.float32),
        scales=np.array([[0.5, 1.0]], dtype=np.float32),
    )

    assert points.shape == (1, 77, 2)
    assert confidence.shape == (1, 77, 1)
    assert points[0, 0] == pytest.approx([50.0, 10.0])
    assert confidence[0, 0, 0] == 9.0


def test_runner_requests_only_cpu_execution_provider(tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeSession:
        def __init__(self, path, *, sess_options, providers):
            captured.update(path=path, options=sess_options, providers=providers)

        def get_inputs(self):
            return [SimpleNamespace(name="imgs")]

        def get_outputs(self):
            return [SimpleNamespace(name="heatmaps")]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, output_names, feed):
            batch = feed["imgs"].shape[0]
            return [np.zeros((batch, 77, 64, 48), dtype=np.float32)]

    fake_ort = SimpleNamespace(
        SessionOptions=type("SessionOptions", (), {}),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        InferenceSession=FakeSession,
    )
    model = tmp_path / "vitpose.onnx"
    model.write_bytes(b"fixture")

    runner = CpuVitPoseRunner(model, ort_module=fake_ort)
    result = runner.run(np.zeros((2, 3, 256, 192), dtype=np.float32))

    assert captured["providers"] == ["CPUExecutionProvider"]
    assert runner.providers == ["CPUExecutionProvider"]
    assert result.shape == (2, 77, 64, 48)


def test_model_free_pipeline_batches_and_emits_keypoints() -> None:
    class FakeRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, images: np.ndarray) -> np.ndarray:
            self.calls += 1
            result = np.zeros((len(images), 77, 64, 48), dtype=np.float32)
            result[:, :, 32, 24] = 1.0
            return result

    frames = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    boxes = torch.tensor([[32.0, 32.0, 64.0]] * 3)
    runner = FakeRunner()

    result = infer_vitpose(frames, boxes, runner, batch_size=2, flip_test=True)

    assert runner.calls == 4
    assert result.shape == (3, 77, 3)
    assert torch.isfinite(result).all()
    # A peak on an even-width heatmap lands on adjacent pixels after the
    # synthetic flip, so averaging the two passes produces two 0.5 peaks.
    assert torch.all(result[..., 2] == 0.5)
