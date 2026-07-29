"""Prepare GEM-X VitPose keypoints with ONNX Runtime's CPU provider.

The crop, flip-test, SOMA-77 joint mapping, and heatmap decoding are adapted
from NVIDIA's Apache-2.0 GEM-X implementation at commit
32992550dba114c62243fb55e361311972dce8f9. The app-owned runner deliberately
selects only ``CPUExecutionProvider`` because the macOS CoreML provider fails
this dynamically batched graph.

Run this script with the pinned GEM-X virtual environment. The generated
``vitpose.pt`` is provider-internal pickle data and must never cross AutoAnim's
safe JSON/NPZ boundary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_VITPOSE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_VITPOSE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SOMA77_FLIP_PAIRS = (
    (9, 10),
    (11, 39),
    (12, 40),
    (13, 41),
    (14, 42),
    (15, 43),
    (16, 44),
    (17, 45),
    (18, 46),
    (19, 47),
    (20, 48),
    (21, 49),
    (22, 50),
    (23, 51),
    (24, 52),
    (25, 53),
    (26, 54),
    (27, 55),
    (28, 56),
    (29, 57),
    (30, 58),
    (31, 59),
    (32, 60),
    (33, 61),
    (34, 62),
    (35, 63),
    (36, 64),
    (37, 65),
    (38, 66),
    (67, 72),
    (68, 73),
    (69, 74),
    (70, 75),
    (71, 76),
)


class CpuVitPoseRunner:
    """Small ONNX adapter that cannot silently activate CoreML."""

    def __init__(
        self,
        model_path: Path,
        *,
        intra_op_threads: int = 0,
        ort_module: Any | None = None,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"VitPose ONNX model does not exist: {model_path}")
        if ort_module is None:
            import onnxruntime as ort_module

        options = ort_module.SessionOptions()
        options.graph_optimization_level = ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads > 0:
            options.intra_op_num_threads = intra_op_threads
        self.session = ort_module.InferenceSession(
            str(model_path.resolve()),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [value.name for value in self.session.get_inputs()]
        self.output_names = [value.name for value in self.session.get_outputs()]
        self.providers = list(self.session.get_providers())
        if self.input_names != ["imgs"]:
            raise RuntimeError(f"Unexpected VitPose inputs: {self.input_names}")
        if self.output_names != ["heatmaps"]:
            raise RuntimeError(f"Unexpected VitPose outputs: {self.output_names}")
        if self.providers != ["CPUExecutionProvider"]:
            raise RuntimeError(f"CPU-only ONNX session was not honored: {self.providers}")

    def run(self, images: np.ndarray) -> np.ndarray:
        images = np.ascontiguousarray(images, dtype=np.float32)
        if images.ndim != 4 or images.shape[1:] != (3, 256, 192):
            raise ValueError(f"Expected VitPose input (N,3,256,192), got {images.shape}")
        result = self.session.run(self.output_names, {"imgs": images})[0]
        heatmaps = np.asarray(result, dtype=np.float32)
        if heatmaps.ndim != 4 or heatmaps.shape[0] != images.shape[0] or heatmaps.shape[1] != 77:
            raise RuntimeError(
                f"Unexpected heatmap shape {heatmaps.shape} for input batch {images.shape}"
            )
        if not np.isfinite(heatmaps).all():
            raise RuntimeError("VitPose produced non-finite heatmaps")
        return heatmaps


def read_video_frames(video_path: Path, timing_path: Path) -> np.ndarray:
    """Re-decode and prove byte-identical RGB frames against detection."""

    try:
        from .video_timing import (
            decode_video_with_timing,
            load_timing,
            verify_decoded_frames,
        )
    except ImportError:  # Direct provider-worker execution.
        from video_timing import (
            decode_video_with_timing,
            load_timing,
            verify_decoded_frames,
        )
    timing = load_timing(timing_path, video_path)
    frames, _ = decode_video_with_timing(video_path)
    verify_decoded_frames(frames, timing)
    return frames


def vitpose_preprocess(frames: np.ndarray, bbx_xys: torch.Tensor) -> np.ndarray:
    """Apply GEM-X's affine square crop and ImageNet normalization."""

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected BGR video frames (T,H,W,3), got {frames.shape}")
    if tuple(bbx_xys.shape) != (len(frames), 3):
        raise ValueError(
            f"Expected one (cx,cy,scale) box per frame, got frames={len(frames)}, "
            f"boxes={tuple(bbx_xys.shape)}"
        )
    boxes = bbx_xys.detach().cpu().float()
    if not torch.isfinite(boxes).all() or torch.any(boxes[:, 2] <= 0):
        raise ValueError("Bounding-box centers/scales must be finite with positive scale")

    output = np.zeros((len(frames), 3, 256, 256), dtype=np.float32)
    for index, (frame, box) in enumerate(zip(frames, boxes, strict=True)):
        center_x, center_y, scale = (float(value) for value in box)
        half_scale = scale / 2
        source = np.array(
            [
                [center_x - half_scale, center_y - half_scale],
                [center_x + half_scale, center_y - half_scale],
                [center_x, center_y],
            ],
            dtype=np.float32,
        )
        destination = np.array([[0, 0], [255, 0], [127.5, 127.5]], dtype=np.float32)
        affine = cv2.getAffineTransform(source, destination)
        crop = cv2.warpAffine(frame, affine, (256, 256), flags=cv2.INTER_LINEAR)
        crop = crop[..., ::-1].astype(np.float32) / 255.0
        crop = (crop - _VITPOSE_MEAN) / _VITPOSE_STD
        output[index] = crop.transpose(2, 0, 1)
    return output


def flip_heatmaps_soma77(heatmaps: np.ndarray) -> np.ndarray:
    """Undo a horizontal input flip and swap GEM-X's left/right SOMA joints."""

    if heatmaps.ndim != 4 or heatmaps.shape[1] != 77:
        raise ValueError(f"Expected heatmaps (N,77,H,W), got {heatmaps.shape}")
    remapped = heatmaps.copy()
    for left, right in _SOMA77_FLIP_PAIRS:
        remapped[:, left] = heatmaps[:, right]
        remapped[:, right] = heatmaps[:, left]
    return np.ascontiguousarray(remapped[..., ::-1])


def keypoints_from_heatmaps(
    heatmaps: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode heatmaps with GEM-X's quarter-pixel UDP adjustment."""

    batch_size, joint_count, height, width = heatmaps.shape
    if centers.shape != (batch_size, 2) or scales.shape != (batch_size, 2):
        raise ValueError("Heatmap centers/scales do not match the batch")
    flattened = heatmaps.reshape(batch_size, joint_count, -1)
    indices = flattened.argmax(-1)
    point_x = (indices % width).astype(np.float32)
    point_y = (indices // width).astype(np.float32)
    for batch_index in range(batch_size):
        for joint_index in range(joint_count):
            heatmap = heatmaps[batch_index, joint_index]
            x_value = int(point_x[batch_index, joint_index])
            y_value = int(point_y[batch_index, joint_index])
            if 1 < x_value < width - 1:
                point_x[batch_index, joint_index] += (
                    np.sign(
                        heatmap[y_value, x_value + 1] - heatmap[y_value, x_value - 1]
                    )
                    * 0.25
                )
            if 1 < y_value < height - 1:
                point_y[batch_index, joint_index] += (
                    np.sign(
                        heatmap[y_value + 1, x_value] - heatmap[y_value - 1, x_value]
                    )
                    * 0.25
                )
    predictions = np.stack([point_x, point_y], axis=-1)
    predictions[..., 0] = (
        predictions[..., 0] / width * (scales[:, [0]] * 200)
        + centers[:, [0]]
        - scales[:, [0]] * 100
    )
    predictions[..., 1] = (
        predictions[..., 1] / height * (scales[:, [1]] * 200)
        + centers[:, [1]]
        - scales[:, [1]] * 100
    )
    confidence = flattened.max(-1, keepdims=True)
    return predictions, confidence


def postprocess_heatmaps(heatmaps: np.ndarray, bbx_xys: torch.Tensor) -> torch.Tensor:
    boxes = bbx_xys.detach().cpu().float()
    centers = boxes[:, :2].numpy()
    scales = (
        torch.cat((boxes[:, [2]] * 24 / 32, boxes[:, [2]]), dim=1) / 200
    ).numpy()
    predictions, confidence = keypoints_from_heatmaps(heatmaps, centers, scales)
    return torch.from_numpy(np.concatenate((predictions, confidence), axis=-1))


def infer_vitpose(
    frames: np.ndarray,
    bbx_xys: torch.Tensor,
    runner: CpuVitPoseRunner,
    *,
    batch_size: int = 1,
    flip_test: bool = True,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    square_crops = vitpose_preprocess(frames, bbx_xys)
    model_crops = square_crops[:, :, :, 32:224]
    results: list[torch.Tensor] = []
    for start in range(0, len(model_crops), batch_size):
        batch = model_crops[start : start + batch_size]
        heatmaps = runner.run(batch)
        if flip_test:
            flipped_input = np.ascontiguousarray(batch[..., ::-1])
            flipped_heatmaps = flip_heatmaps_soma77(runner.run(flipped_input))
            heatmaps = (heatmaps + flipped_heatmaps) * 0.5
        results.append(postprocess_heatmaps(heatmaps, bbx_xys[start : start + len(batch)]))
    keypoints = torch.cat(results, dim=0).clone()
    if tuple(keypoints.shape) != (len(frames), 77, 3):
        raise RuntimeError(f"Unexpected VitPose result shape: {tuple(keypoints.shape)}")
    if not torch.isfinite(keypoints).all():
        raise RuntimeError("VitPose produced non-finite keypoints")
    return keypoints


def prepare_vitpose(
    video_path: Path,
    boxes_path: Path,
    timing_path: Path,
    output_path: Path,
    model_path: Path,
    *,
    batch_size: int = 1,
    intra_op_threads: int = 0,
    flip_test: bool = True,
) -> dict[str, object]:
    for label, path in (
        ("input video", video_path),
        ("bounding boxes", boxes_path),
        ("frame timing", timing_path),
        ("VitPose model", model_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    started = time.perf_counter()
    frames = read_video_frames(video_path, timing_path)
    boxes_data = torch.load(boxes_path, map_location="cpu", weights_only=True)
    if not isinstance(boxes_data, dict) or "bbx_xys" not in boxes_data:
        raise ValueError("Bounding-box file must contain a bbx_xys tensor")
    bbx_xys = boxes_data["bbx_xys"]
    if not isinstance(bbx_xys, torch.Tensor):
        raise ValueError("bbx_xys must be a torch tensor")
    if len(frames) != len(bbx_xys):
        raise RuntimeError(
            f"Video/box frame count mismatch: video={len(frames)}, boxes={len(bbx_xys)}"
        )

    session_started = time.perf_counter()
    runner = CpuVitPoseRunner(
        model_path,
        intra_op_threads=intra_op_threads,
    )
    session_seconds = time.perf_counter() - session_started
    inference_started = time.perf_counter()
    keypoints = infer_vitpose(
        frames,
        bbx_xys,
        runner,
        batch_size=batch_size,
        flip_test=flip_test,
    )
    inference_seconds = time.perf_counter() - inference_started

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.partial")
    torch.save(keypoints, temporary)
    temporary.replace(output_path)
    confidence = keypoints[..., 2]
    return {
        "batch_size": batch_size,
        "confidence_max": float(confidence.max()),
        "confidence_mean": float(confidence.mean()),
        "confidence_min": float(confidence.min()),
        "execution_providers": runner.providers,
        "flip_test": flip_test,
        "frame_count": len(frames),
        "inference_seconds": inference_seconds,
        "input_path": str(video_path.resolve()),
        "model_path": str(model_path.resolve()),
        "output_path": str(output_path.resolve()),
        "output_shape": list(keypoints.shape),
        "session_seconds": session_seconds,
        "total_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--no-flip-test", action="store_true")
    arguments = parser.parse_args()
    result = prepare_vitpose(
        arguments.video.resolve(),
        arguments.boxes.resolve(),
        arguments.timing.resolve(),
        arguments.output.resolve(),
        arguments.model.resolve(),
        batch_size=arguments.batch_size,
        intra_op_threads=arguments.threads,
        flip_test=not arguments.no_flip_test,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
