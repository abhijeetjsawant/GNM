"""Prepare GEM-X bounding boxes with ONNX Runtime's CPU provider.

GEM-X's macOS demo currently always enables the CoreML execution provider for
YOLOX. ONNX Runtime 1.28 fails that graph at runtime because a CoreML partition
reports a different output rank. This worker-local adapter preserves GEM-X's
detector and ByteTrack logic while selecting CPUExecutionProvider explicitly.

Run this script with the pinned GEM-X virtual environment from the GEM-X root.
The generated ``bbx.pt`` is provider-internal pickle data and must never cross
AutoAnim's safe JSON/NPZ boundary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from gem.utils.geo_transform import get_bbx_xys_from_xyxy
from gem.utils.kp2d_utils import smooth_bbx_xyxy
from gem.utils.yolox_detector import (
    YOLOXDetector,
    _download_yolox_onnx,
    detect_and_track,
)
try:
    from .video_timing import decode_video_with_timing, write_timing
except ImportError:  # Direct provider-worker execution.
    from video_timing import decode_video_with_timing, write_timing


def _cpu_detector(model_path: Path | None) -> YOLOXDetector:
    resolved_model = Path(_download_yolox_onnx() if model_path is None else model_path)
    if not resolved_model.is_file():
        raise FileNotFoundError(f"YOLOX model does not exist: {resolved_model}")
    detector = object.__new__(YOLOXDetector)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    detector.sess = ort.InferenceSession(
        str(resolved_model.resolve()),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    detector.input_size = (640, 640)
    detector.score_thr = 0.5
    detector.nms_thr = 0.45
    return detector


def prepare_detection(
    video_path: Path,
    output_path: Path,
    timing_output_path: Path,
    *,
    model_path: Path | None = None,
) -> dict[str, object]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    frames, timing = decode_video_with_timing(video_path)
    frame_count, height, width = frames.shape[:3]
    boxes_numpy, _ = detect_and_track(frames, _cpu_detector(model_path))
    boxes = smooth_bbx_xyxy(torch.from_numpy(boxes_numpy).float(), window=3)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height - 1)
    if not torch.isfinite(boxes).all() or torch.any(boxes[:, 2:] <= boxes[:, :2]):
        raise RuntimeError("YOLOX/ByteTrack produced invalid bounding boxes")
    box_centers_scales = get_bbx_xys_from_xyxy(boxes, base_enlarge=1.2).float()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.partial")
    torch.save({"bbx_xyxy": boxes, "bbx_xys": box_centers_scales}, temporary)
    temporary.replace(output_path)
    write_timing(timing_output_path, timing)
    return {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "execution_provider": "CPUExecutionProvider",
        "output_path": str(output_path.resolve()),
        "timing_output_path": str(timing_output_path.resolve()),
        "box_min": np.min(boxes_numpy, axis=0).tolist(),
        "box_max": np.max(boxes_numpy, axis=0).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    arguments = parser.parse_args()
    result = prepare_detection(
        arguments.video.resolve(),
        arguments.output.resolve(),
        arguments.timing_output.resolve(),
        model_path=arguments.model.resolve() if arguments.model else None,
    )
    import json

    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
