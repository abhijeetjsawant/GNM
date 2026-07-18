"""MediaPipe face extraction with a versioned GNM-68 correspondence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .errors import AutoAnimError


MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MAPPING_NAME = "mediapipe478_to_gnm68_v1"
MEDIAPIPE_TO_GNM68 = np.asarray(
    [
        234, 93, 150, 136, 172, 58, 132, 149, 152, 377, 400, 378, 379, 365, 397, 288, 454,
        70, 63, 105, 66, 107,
        336, 296, 334, 293, 300,
        168, 6, 197, 195,
        64, 98, 2, 327, 294,
        33, 160, 158, 133, 153, 144,
        362, 385, 387, 263, 373, 380,
        61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
        78, 81, 13, 311, 308, 402, 14, 178,
    ],
    dtype=np.int32,
)


@dataclass(frozen=True, slots=True)
class DetectedFace:
    image_bgr: np.ndarray
    landmarks: np.ndarray
    all_landmarks: np.ndarray
    blendshapes: dict[str, float]
    face_width: float
    mapped_in_bounds_fraction: float
    strong_expression_score: float


def validate_model(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise AutoAnimError(
            "DEPENDENCY_MISSING",
            f"MediaPipe face_landmarker.task is missing: {path}",
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise AutoAnimError(
            "DEPENDENCY_MISSING",
            "MediaPipe model checksum does not match the pinned asset",
            {"expected": MODEL_SHA256, "actual": digest, "path": str(path)},
        )
    return path


class FaceExtractor:
    def __init__(self, model_path: str | Path):
        self.model_path = validate_model(model_path)

    def detect(self, image_path: str | Path) -> DetectedFace:
        image_path = Path(image_path)
        if not image_path.is_file() or image_path.stat().st_size > 100 * 1024 * 1024:
            raise AutoAnimError("INPUT_INVALID", f"Invalid image input: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise AutoAnimError("MEDIA_INVALID", "OpenCV could not decode the image")
        height, width = image.shape[:2]
        if width > 12_000 or height > 12_000 or width * height > 40_000_000:
            raise AutoAnimError("LIMIT_EXCEEDED", "Image exceeds dimension/pixel limits")
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=2,
        )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            result = detector.detect(media_image)
        count = len(result.face_landmarks)
        if count == 0:
            raise AutoAnimError("FACE_NOT_FOUND", "No face was detected")
        if count != 1:
            raise AutoAnimError("MULTIPLE_FACES", f"Expected one face; detected {count}")
        all_landmarks = np.asarray(
            [(point.x * width, point.y * height) for point in result.face_landmarks[0]],
            dtype=np.float32,
        )
        mapped = all_landmarks[MEDIAPIPE_TO_GNM68]
        margin_x, margin_y = 0.05 * width, 0.05 * height
        inside = (
            (mapped[:, 0] >= -margin_x)
            & (mapped[:, 0] <= width + margin_x)
            & (mapped[:, 1] >= -margin_y)
            & (mapped[:, 1] <= height + margin_y)
        )
        in_bounds = float(np.mean(inside))
        if in_bounds < 0.90:
            raise AutoAnimError(
                "FIT_REJECTED",
                "Too many mapped landmarks lie outside the image",
                {"mapped_in_bounds_fraction": in_bounds},
            )
        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            blendshapes = {
                category.category_name: float(category.score)
                for category in result.face_blendshapes[0]
            }
        prefixes = ("mouth", "eyeBlink", "eyeSquint", "eyeWide", "browDown", "browInnerUp")
        strong_score = max(
            (score for name, score in blendshapes.items() if name.startswith(prefixes)),
            default=0.0,
        )
        return DetectedFace(
            image_bgr=image,
            landmarks=mapped,
            all_landmarks=all_landmarks,
            blendshapes=blendshapes,
            face_width=float(np.ptp(mapped[:, 0])),
            mapped_in_bounds_fraction=in_bounds,
            strong_expression_score=strong_score,
        )
