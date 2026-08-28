#!/usr/bin/env python3
"""Third 2D body detector: NVIDIA GEM-X's SOMA-77 whole-body model.

Emits the same versioned observation contract as the Apple Vision and MediaPipe
workers. Reimplemented against the ONNX graph and the published preprocessing
recipe rather than importing GEM-X, so this worker depends only on numpy, cv2
and onnxruntime.

Why it exists. Battle 0 concluded Apple Vision is model-limited at ~26 mm of 2D
error regardless of resolution; increment 2 measured its joint definitions
drifting ~10% of limb length with pose; increment 3 found MediaPipe worse still.
All three findings point at the detector. This model differs from those two in
the way that should matter most: it predicts **skeletal joint centres** on a
77-joint whole-body skeleton, where Apple Vision and MediaPipe predict *surface*
landmarks. Surface landmarks move relative to the underlying bone as a subject
turns, which is the mechanism behind the limb-length instability. Interior joint
centres should not.

Two structural differences from the other workers:

* It is **top-down**: it needs a person box per subject. Boxes are read from an
  existing observation file rather than from GEM-X's bundled YOLOX, whose
  checkpoint is Human-Art-trained and has not cleared the asset licence gate.
  Reusing boxes we already have keeps that asset out of the path entirely.
* SOMA-77 carries **15 articulated finger joints per hand**, which the AutoAnim-19
  contract cannot express. They are decoded and discarded here. That is the
  N5.1 hand blocker's missing input, and it is worth its own increment.

    python workers/commercial_multiview/soma77_pose.py MODEL.onnx BOXES.jsonl FRAME...

`BOXES.jsonl` is any file in the observation contract; each frame's people supply
the boxes for the same frame here.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCHEMA_VERSION = "autoanim.body-observations/1.1"
DETECTOR = "nvidia_gemx_soma77"

# Input is a 256x256 square crop, of which the model consumes the centre 192
# columns. Both numbers come from the published preprocessing.
CROP = 256
MODEL_WIDTH = 192
_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
# The box is squared up and enlarged before cropping; a tight joint hull cuts off
# the head and feet, which are exactly the joints already worst served.
BOX_PADDING = 1.25

# SOMA-77 indices, from src/autoanim_gnm/data/somaskel77-v1.json.
SOMA77_TO_AUTOANIM: dict[str, int] = {
    "neck": 5,            # Neck2
    "left_shoulder": 12,  # LeftArm -- the glenohumeral joint, not the clavicle
    "left_elbow": 13,     # LeftForeArm
    "left_wrist": 14,     # LeftHand
    "right_shoulder": 40,  # RightArm
    "right_elbow": 41,    # RightForeArm
    "right_wrist": 42,    # RightHand
    "root": 0,            # Hips
    "left_hip": 67,       # LeftLeg
    "left_knee": 68,      # LeftShin
    "left_ankle": 69,     # LeftFoot
    "right_hip": 72,      # RightLeg
    "right_knee": 73,     # RightShin
    "right_ankle": 74,    # RightFoot
    "left_eye": 9,
    "right_eye": 10,
    # `nose` has no SOMA counterpart; Head is the nearest and is not the same
    # point. Emitted, but it is an approximation and the head is GNM's anyway.
    "nose": 6,
}
# SOMA-77 has no ears. They are left absent rather than faked; the reconstruction
# side already falls back to the neck for missing head landmarks.


def _square_box(joints: dict[str, Any]) -> tuple[float, float, float] | None:
    points = [
        (float(j["x"]), float(j["y"]))
        for j in joints.values()
        if np.isfinite(j.get("x", np.nan)) and np.isfinite(j.get("y", np.nan))
        and float(j.get("confidence", 0.0)) >= 0.2
    ]
    if len(points) < 4:
        return None
    array = np.asarray(points, dtype=np.float64)
    low, high = array.min(axis=0), array.max(axis=0)
    centre = 0.5 * (low + high)
    side = float(np.max(high - low)) * BOX_PADDING
    return float(centre[0]), float(centre[1]), max(side, 16.0)


def _crop(frame: np.ndarray, box: tuple[float, float, float]) -> np.ndarray:
    import cv2

    cx, cy, side = box
    half = side / 2.0
    source = np.asarray(
        [[cx - half, cy - half], [cx + half, cy - half], [cx, cy]], dtype=np.float32
    )
    target = np.asarray([[0, 0], [CROP - 1, 0], [CROP / 2 - 0.5, CROP / 2 - 0.5]], dtype=np.float32)
    warped = cv2.warpAffine(
        frame, cv2.getAffineTransform(source, target), (CROP, CROP), flags=cv2.INTER_LINEAR
    )
    rgb = warped[..., ::-1].astype(np.float32) / 255.0
    normalised = (rgb - _MEAN) / _STD
    centred = normalised.transpose(2, 0, 1)[:, :, (CROP - MODEL_WIDTH) // 2 : (CROP + MODEL_WIDTH) // 2]
    return centred


def _decode(heatmaps: np.ndarray, box: tuple[float, float, float]) -> np.ndarray:
    """argmax with the published quarter-pixel UDP nudge, back into image space."""

    keypoints, height, width = heatmaps.shape
    flat = heatmaps.reshape(keypoints, -1)
    index = flat.argmax(axis=1)
    x = (index % width).astype(np.float64)
    y = (index // width).astype(np.float64)
    for k in range(keypoints):
        ix, iy = int(x[k]), int(y[k])
        if 1 < ix < width - 1:
            x[k] += np.sign(heatmaps[k, iy, ix + 1] - heatmaps[k, iy, ix - 1]) * 0.25
        if 1 < iy < height - 1:
            y[k] += np.sign(heatmaps[k, iy + 1, ix] - heatmaps[k, iy - 1, ix]) * 0.25
    cx, cy, side = box
    # Scale is anisotropic: the model saw 192 of the 256 crop columns.
    scale_x = side * MODEL_WIDTH / CROP
    out = np.empty((keypoints, 3), dtype=np.float64)
    out[:, 0] = x / width * scale_x + cx - scale_x / 2.0
    out[:, 1] = y / height * side + cy - side / 2.0
    out[:, 2] = flat.max(axis=1)
    return out


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    import cv2
    import onnxruntime

    model = Path(sys.argv[1]).resolve(strict=True)
    boxes_by_frame: dict[int, list[dict[str, Any]]] = {}
    for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            boxes_by_frame[int(record["frame_index"])] = record.get("people", [])
    frames = [Path(value) for value in sys.argv[3:]]

    session = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for path in frames:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        index = int(path.stem)
        people: list[dict[str, Any]] = []
        for person in boxes_by_frame.get(index, []):
            box = _square_box(person.get("joints", {}))
            if box is None:
                continue
            batch = _crop(image, box)[None].astype(np.float32)
            heatmaps = session.run(None, {input_name: batch})[0][0]
            decoded = _decode(heatmaps, box)
            # Measured, the peaks already lie in [0.895, 0.992], so they are
            # usable as-is. An earlier draft pushed them through a sigmoid,
            # which flattened every joint to ~0.72 and destroyed what little
            # discrimination the confidence carried.
            confidences = np.clip(decoded[:, 2], 0.0, 1.0)
            joints = {
                name: {
                    "x": float(decoded[soma, 0]),
                    "y": float(decoded[soma, 1]),
                    "confidence": float(confidences[soma]),
                }
                for name, soma in SOMA77_TO_AUTOANIM.items()
            }
            people.append({"index": len(people), "joints": joints})
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "detector": DETECTOR,
                    "frame_index": index,
                    "width": width,
                    "height": height,
                    "image_path": str(path),
                    "people": people,
                },
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
