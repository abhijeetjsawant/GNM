#!/usr/bin/env python3
"""Second 2D body detector for the clean-room multiview lane.

Emits the same versioned observation contract as
`workers/commercial_multiview/apple_vision_pose.swift`, so the reconstruction
side cannot tell which detector produced a frame. Existing purely so the two can
be measured against each other: Battle 0 concluded Apple Vision is model-limited
at roughly 26 mm of 2D error at the subject regardless of input resolution, and
increment 2 found its joint definitions drift by ~10% of limb length with pose.
Neither is fixable downstream, so the question is whether a different detector
carries different error.

Uses the MediaPipe Tasks API. The legacy `mediapipe.python.solutions` graph is
documented in docs/MAMMA_MULTIVIEW_EXECUTION.md as aborting natively on this
host; the Tasks runtime was probed on the same machine and runs on CPU through
XNNPACK.

    python workers/commercial_multiview/mediapipe_pose.py MODEL.task FRAME...

Set AUTOANIM_MP_CONFIDENCE to override the detection/presence/tracking floor.

Writes one JSON object per frame to stdout, in argument order.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCHEMA_VERSION = "autoanim.body-observations/1.1"
DETECTOR = "mediapipe_pose_landmarker"

# MediaPipe's pose landmark indices, from the Pose Landmarker model card.
_NOSE = 0
_LEFT_EYE, _RIGHT_EYE = 2, 5
_LEFT_EAR, _RIGHT_EAR = 7, 8
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_LEFT_ELBOW, _RIGHT_ELBOW = 13, 14
_LEFT_WRIST, _RIGHT_WRIST = 15, 16
_LEFT_HIP, _RIGHT_HIP = 23, 24
_LEFT_KNEE, _RIGHT_KNEE = 25, 26
_LEFT_ANKLE, _RIGHT_ANKLE = 27, 28

# AutoAnim's 19 body points, mapped from MediaPipe's 33. Two are *derived*
# midpoints rather than predicted landmarks, because MediaPipe has no neck and
# no pelvis. That is a definitional difference from Apple Vision, which predicts
# both directly, and it is exactly the class of discrepancy Battle 0 identified
# as the dominant error term -- so it must be read as a caveat on any comparison
# between the two detectors, not as noise.
DIRECT_JOINTS: dict[str, int] = {
    "nose": _NOSE,
    "right_shoulder": _RIGHT_SHOULDER,
    "right_elbow": _RIGHT_ELBOW,
    "right_wrist": _RIGHT_WRIST,
    "left_shoulder": _LEFT_SHOULDER,
    "left_elbow": _LEFT_ELBOW,
    "left_wrist": _LEFT_WRIST,
    "right_hip": _RIGHT_HIP,
    "right_knee": _RIGHT_KNEE,
    "right_ankle": _RIGHT_ANKLE,
    "left_hip": _LEFT_HIP,
    "left_knee": _LEFT_KNEE,
    "left_ankle": _LEFT_ANKLE,
    "right_eye": _RIGHT_EYE,
    "left_eye": _LEFT_EYE,
    "right_ear": _RIGHT_EAR,
    "left_ear": _LEFT_EAR,
}
DERIVED_JOINTS: dict[str, tuple[int, int]] = {
    "neck": (_LEFT_SHOULDER, _RIGHT_SHOULDER),
    "root": (_LEFT_HIP, _RIGHT_HIP),
}

MAXIMUM_PEOPLE = 2
# MediaPipe's 0.5 defaults are tuned for a close-up single subject. On a wide
# four-camera stage they find ~0.55 people per frame -- i.e. they miss both
# performers most of the time. Measured over 40 frames of the reference fixture:
# 0.5 -> 0.57 people/frame and 1 frame in 40 with two; 0.1 -> 1.82 and 33 in 40.
# The reconstruction side already gates on confidence and rejects outliers, so
# it is better to let weak detections through and let geometry judge them.
DEFAULT_CONFIDENCE = 0.1


def _confidence(landmark: Any) -> float:
    """MediaPipe reports visibility and presence separately; trust the weaker."""

    values = [
        float(getattr(landmark, name))
        for name in ("visibility", "presence")
        if getattr(landmark, name, None) is not None
    ]
    return min(values) if values else 1.0


def _person(landmarks: Any, width: int, height: int) -> dict[str, Any]:
    joints: dict[str, dict[str, float]] = {}
    for name, index in DIRECT_JOINTS.items():
        landmark = landmarks[index]
        joints[name] = {
            "x": float(landmark.x) * width,
            "y": float(landmark.y) * height,
            "confidence": _confidence(landmark),
        }
    for name, (left, right) in DERIVED_JOINTS.items():
        a, b = landmarks[left], landmarks[right]
        joints[name] = {
            "x": 0.5 * (float(a.x) + float(b.x)) * width,
            "y": 0.5 * (float(a.y) + float(b.y)) * height,
            # A midpoint is only as trustworthy as its weaker endpoint.
            "confidence": min(_confidence(a), _confidence(b)),
        }
    return {"joints": joints}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    import os

    confidence = float(os.environ.get("AUTOANIM_MP_CONFIDENCE", DEFAULT_CONFIDENCE))
    model = Path(sys.argv[1]).resolve(strict=True)
    frames = [Path(value) for value in sys.argv[2:]]

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from PIL import Image

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=MAXIMUM_PEOPLE,
        min_pose_detection_confidence=confidence,
        min_pose_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame in frames:
            array = np.asarray(Image.open(frame).convert("RGB"))
            height, width = array.shape[:2]
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
            )
            people = [
                {"index": index, **_person(landmarks, width, height)}
                for index, landmarks in enumerate(result.pose_landmarks)
            ]
            print(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "detector": DETECTOR,
                        "frame_index": int(frame.stem),
                        "width": width,
                        "height": height,
                        "image_path": str(frame),
                        "people": people,
                    },
                    separators=(",", ":"),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
