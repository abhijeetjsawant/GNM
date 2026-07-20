"""Persisted, fail-closed provenance for one exact video detector run.

Capture v1 intentionally does not retain detector configuration or hashes of the
RGB buffers passed to MediaPipe.  :class:`~autoanim_gnm.video_capture.VideoCaptureRun`
does retain those facts in memory.  This module serializes that envelope as a
bounded canonical JSON artifact so later verification does not reconstruct
detector provenance from VisualTrack or Observation artifacts.

The document SHA-256 detects accidental standalone mutation.  Authenticity still
comes from sealing the artifact in the JobStore ledger; the digest is not a
publisher signature.  This schema never authorizes motion or production use.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .video_capture import CAPTURE_SCHEMA_VERSION, CaptureTrack, VideoCaptureRun


VIDEO_CAPTURE_RUN_SCHEMA_VERSION = "autoanim.video-capture-run/1.0"
VIDEO_CAPTURE_RUN_KIND = "video_capture_run"
DETECTOR_IMPLEMENTATION = "mediapipe.tasks.vision.FaceLandmarker"
DETECTOR_INGRESS_HASH_DOMAIN = "rgb8_hwc_contiguous_exact_mp_image_input"
MAX_VIDEO_CAPTURE_RUN_BYTES = 2 * 1024 * 1024
MAX_VIDEO_CAPTURE_RUN_FRAMES = 7_200

_SHA256_HEX = frozenset("0123456789abcdef")


class VideoCaptureRunArtifactError(ValueError):
    """A persisted detector-run document is malformed or fails reconstruction."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VideoCaptureRunArtifactError(
            "INVALID_DOCUMENT",
            "Video capture-run evidence must be finite canonical JSON",
        ) from exc


def _payload_sha256(document: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(document))
    payload.pop("document_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def video_capture_run_payload_sha256(document: Mapping[str, Any]) -> str:
    """Return the canonical payload digest, excluding ``document_sha256``."""

    return _payload_sha256(document)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value).astype("<i8", copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _configuration_sha256(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(configuration))).hexdigest()


def _ordered_binding_sha256(
    source_pts: np.ndarray, frame_hashes: tuple[str, ...]
) -> str:
    binding = {
        "source_pts": [int(value) for value in np.asarray(source_pts)],
        "detector_ingress_rgb_sha256": list(frame_hashes),
    }
    return hashlib.sha256(_canonical_json(binding)).hexdigest()


def _document_for_run(run: VideoCaptureRun) -> dict[str, Any]:
    if not isinstance(run, VideoCaptureRun):
        raise VideoCaptureRunArtifactError(
            "INVALID_RUN", "Expected one in-memory VideoCaptureRun"
        )
    if type(run.num_faces) is not int or run.num_faces != 1:
        raise VideoCaptureRunArtifactError(
            "INVALID_CONFIGURATION", "Detector num_faces must be the integer 1"
        )
    frame_hashes = tuple(run.detector_ingress_rgb_sha256)
    if not 0 < run.track.frame_count <= MAX_VIDEO_CAPTURE_RUN_FRAMES:
        raise VideoCaptureRunArtifactError(
            "FRAME_LIMIT", "Video capture-run frame count is outside schema bounds"
        )
    configuration = run.detector_configuration()
    document: dict[str, Any] = {
        "schema_version": VIDEO_CAPTURE_RUN_SCHEMA_VERSION,
        "kind": VIDEO_CAPTURE_RUN_KIND,
        "source": {
            "capture_schema_version": run.track.schema_version,
            "name": run.track.provenance.source_name,
            "sha256": run.track.provenance.source_sha256,
            "bytes": run.track.provenance.source_bytes,
            "frame_count": run.track.frame_count,
            "source_start_pts": int(run.track.source_pts[0]),
            "source_pts_sha256": _array_sha256(run.track.source_pts),
            "time_base": [
                run.track.provenance.time_base_numerator,
                run.track.provenance.time_base_denominator,
            ],
        },
        "model": {
            "name": run.track.provenance.model_name,
            "sha256": run.track.provenance.model_sha256,
        },
        "detector": {
            "implementation": DETECTOR_IMPLEMENTATION,
            "runtime": {"mediapipe": run.track.provenance.mediapipe_version},
            "configuration": configuration,
            "configuration_sha256": _configuration_sha256(configuration),
        },
        "detector_ingress": {
            "hash_algorithm": "sha256",
            "hash_domain": run.hash_domain,
            "ordered_rgb_sha256": list(frame_hashes),
            "ordered_pts_and_rgb_sha256": _ordered_binding_sha256(
                run.track.source_pts, frame_hashes
            ),
        },
        "claims": {
            "detector_ingress_pixels_retained": False,
            "detector_ingress_hashes_retained": True,
            "changes_final_gnm_motion": False,
            "production_validated": False,
        },
    }
    document["document_sha256"] = _payload_sha256(document)
    return document


def build_video_capture_run_document(run: VideoCaptureRun) -> dict[str, Any]:
    """Build one deterministic, non-authorizing CaptureRun evidence document."""

    document = _document_for_run(run)
    # Exercise the same strict reconstruction path used for persisted evidence.
    _reconstruct_document(document, expected_capture=run.track)
    return document


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VideoCaptureRunArtifactError(
                "DUPLICATE_KEY", f"Duplicate JSON member {key!r}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise VideoCaptureRunArtifactError(
        "NONFINITE_NUMBER", f"Non-finite JSON number {value!r} is forbidden"
    )


def _object(value: Any, *, field: str, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise VideoCaptureRunArtifactError(
            "INVALID_FIELDS", f"{field} has missing or unknown members"
        )
    return value


def _strict_integer(value: Any, *, field: str, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        qualifier = "positive " if positive else ""
        raise VideoCaptureRunArtifactError(
            "INVALID_TYPE", f"{field} must be a {qualifier}integer"
        )
    return value


def _strict_float(value: Any, *, field: str) -> float:
    if type(value) is not float or not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise VideoCaptureRunArtifactError(
            "INVALID_TYPE", f"{field} must be a finite JSON float in [0,1]"
        )
    return value


def _string(value: Any, *, field: str, maximum: int = 255) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise VideoCaptureRunArtifactError(
            "INVALID_TYPE", f"{field} must be a non-empty trimmed string"
        )
    return value


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA256_HEX
    ):
        raise VideoCaptureRunArtifactError(
            "INVALID_DIGEST", f"{field} must be a lowercase SHA-256 digest"
        )
    return value


def _validate_configuration(value: Any) -> dict[str, Any]:
    field = "detector.configuration"
    configuration = _object(
        value,
        field=field,
        keys=(
            "num_faces",
            "min_face_detection_confidence",
            "min_face_presence_confidence",
            "min_tracking_confidence",
            "running_mode",
            "output_face_blendshapes",
            "output_facial_transformation_matrices",
            "detector_ingress_hash_domain",
        ),
    )
    if _strict_integer(configuration["num_faces"], field=f"{field}.num_faces") != 1:
        raise VideoCaptureRunArtifactError(
            "INVALID_CONFIGURATION", "Detector num_faces must equal 1"
        )
    for name in (
        "min_face_detection_confidence",
        "min_face_presence_confidence",
        "min_tracking_confidence",
    ):
        _strict_float(configuration[name], field=f"{field}.{name}")
    if (
        configuration["running_mode"] != "VIDEO"
        or type(configuration["output_face_blendshapes"]) is not bool
        or configuration["output_face_blendshapes"] is not True
        or type(configuration["output_facial_transformation_matrices"]) is not bool
        or configuration["output_facial_transformation_matrices"] is not True
        or configuration["detector_ingress_hash_domain"]
        != DETECTOR_INGRESS_HASH_DOMAIN
    ):
        raise VideoCaptureRunArtifactError(
            "INVALID_CONFIGURATION", "Detector configuration is not exact"
        )
    return configuration


def _reconstruct_document(
    document: Mapping[str, Any], *, expected_capture: CaptureTrack
) -> VideoCaptureRun:
    if not isinstance(expected_capture, CaptureTrack):
        raise VideoCaptureRunArtifactError(
            "INVALID_EXPECTATION", "expected_capture must be one CaptureTrack"
        )
    root = _object(
        dict(document),
        field="video_capture_run",
        keys=(
            "schema_version",
            "kind",
            "source",
            "model",
            "detector",
            "detector_ingress",
            "claims",
            "document_sha256",
        ),
    )
    if (
        root["schema_version"] != VIDEO_CAPTURE_RUN_SCHEMA_VERSION
        or root["kind"] != VIDEO_CAPTURE_RUN_KIND
    ):
        raise VideoCaptureRunArtifactError(
            "UNSUPPORTED_SCHEMA", "Unsupported video capture-run schema"
        )
    declared_document_sha = _sha256(
        root["document_sha256"], field="document_sha256"
    )
    if declared_document_sha != _payload_sha256(root):
        raise VideoCaptureRunArtifactError(
            "DOCUMENT_HASH_MISMATCH", "Video capture-run payload hash does not match"
        )

    source = _object(
        root["source"],
        field="source",
        keys=(
            "capture_schema_version",
            "name",
            "sha256",
            "bytes",
            "frame_count",
            "source_start_pts",
            "source_pts_sha256",
            "time_base",
        ),
    )
    if source["capture_schema_version"] != CAPTURE_SCHEMA_VERSION:
        raise VideoCaptureRunArtifactError(
            "UNSUPPORTED_SCHEMA", "CaptureRun evidence must bind Capture v1"
        )
    source_name = _string(source["name"], field="source.name")
    if Path(source_name).name != source_name:
        raise VideoCaptureRunArtifactError(
            "INVALID_NAME", "source.name must be a path-free basename"
        )
    _sha256(source["sha256"], field="source.sha256")
    _strict_integer(source["bytes"], field="source.bytes", positive=True)
    frame_count = _strict_integer(
        source["frame_count"], field="source.frame_count", positive=True
    )
    if frame_count > MAX_VIDEO_CAPTURE_RUN_FRAMES:
        raise VideoCaptureRunArtifactError(
            "FRAME_LIMIT", "Video capture-run frame count exceeds its schema bound"
        )
    _strict_integer(source["source_start_pts"], field="source.source_start_pts")
    _sha256(source["source_pts_sha256"], field="source.source_pts_sha256")
    time_base = source["time_base"]
    if not isinstance(time_base, list) or len(time_base) != 2:
        raise VideoCaptureRunArtifactError(
            "INVALID_TYPE", "source.time_base must contain two positive integers"
        )
    for index, value in enumerate(time_base):
        _strict_integer(value, field=f"source.time_base.{index}", positive=True)

    model = _object(root["model"], field="model", keys=("name", "sha256"))
    model_name = _string(model["name"], field="model.name")
    if Path(model_name).name != model_name:
        raise VideoCaptureRunArtifactError(
            "INVALID_NAME", "model.name must be a path-free basename"
        )
    _sha256(model["sha256"], field="model.sha256")

    detector = _object(
        root["detector"],
        field="detector",
        keys=(
            "implementation",
            "runtime",
            "configuration",
            "configuration_sha256",
        ),
    )
    if detector["implementation"] != DETECTOR_IMPLEMENTATION:
        raise VideoCaptureRunArtifactError(
            "INVALID_CONFIGURATION", "Detector implementation is not exact"
        )
    runtime = _object(
        detector["runtime"], field="detector.runtime", keys=("mediapipe",)
    )
    _string(runtime["mediapipe"], field="detector.runtime.mediapipe", maximum=160)
    configuration = _validate_configuration(detector["configuration"])
    declared_configuration_sha = _sha256(
        detector["configuration_sha256"], field="detector.configuration_sha256"
    )
    if declared_configuration_sha != _configuration_sha256(configuration):
        raise VideoCaptureRunArtifactError(
            "CONFIGURATION_HASH_MISMATCH",
            "Detector configuration hash does not match",
        )

    ingress = _object(
        root["detector_ingress"],
        field="detector_ingress",
        keys=(
            "hash_algorithm",
            "hash_domain",
            "ordered_rgb_sha256",
            "ordered_pts_and_rgb_sha256",
        ),
    )
    if (
        ingress["hash_algorithm"] != "sha256"
        or ingress["hash_domain"] != DETECTOR_INGRESS_HASH_DOMAIN
        or ingress["hash_domain"]
        != configuration["detector_ingress_hash_domain"]
    ):
        raise VideoCaptureRunArtifactError(
            "INVALID_HASH_DOMAIN", "Detector ingress hash domain is not exact"
        )
    raw_hashes = ingress["ordered_rgb_sha256"]
    if not isinstance(raw_hashes, list) or len(raw_hashes) != frame_count:
        raise VideoCaptureRunArtifactError(
            "INVALID_HASH_SEQUENCE",
            "Detector ingress hashes must have exactly one ordered value per frame",
        )
    frame_hashes = tuple(
        _sha256(value, field=f"detector_ingress.ordered_rgb_sha256.{index}")
        for index, value in enumerate(raw_hashes)
    )
    declared_ordered_sha = _sha256(
        ingress["ordered_pts_and_rgb_sha256"],
        field="detector_ingress.ordered_pts_and_rgb_sha256",
    )
    if declared_ordered_sha != _ordered_binding_sha256(
        expected_capture.source_pts, frame_hashes
    ):
        raise VideoCaptureRunArtifactError(
            "ORDERED_BINDING_MISMATCH",
            "Detector ingress hashes are not bound to the expected source PTS order",
        )

    claims = _object(
        root["claims"],
        field="claims",
        keys=(
            "detector_ingress_pixels_retained",
            "detector_ingress_hashes_retained",
            "changes_final_gnm_motion",
            "production_validated",
        ),
    )
    expected_claims = {
        "detector_ingress_pixels_retained": False,
        "detector_ingress_hashes_retained": True,
        "changes_final_gnm_motion": False,
        "production_validated": False,
    }
    if any(type(value) is not bool for value in claims.values()) or claims != expected_claims:
        raise VideoCaptureRunArtifactError(
            "UNSUPPORTED_CLAIM", "Video capture-run claims are not fail-closed"
        )

    reconstructed = VideoCaptureRun(
        track=expected_capture,
        detector_ingress_rgb_sha256=frame_hashes,
        num_faces=configuration["num_faces"],
        confidence_thresholds=(
            configuration["min_face_detection_confidence"],
            configuration["min_face_presence_confidence"],
            configuration["min_tracking_confidence"],
        ),
        hash_domain=ingress["hash_domain"],
    )
    if root != _document_for_run(reconstructed):
        raise VideoCaptureRunArtifactError(
            "RECONSTRUCTION_MISMATCH",
            "Video capture-run evidence does not exactly reconstruct from Capture v1",
        )
    return reconstructed


def _read_document(
    source: str | Path | bytes | bytearray | Mapping[str, Any],
) -> dict[str, Any]:
    payload: bytes | None = None
    if isinstance(source, Mapping):
        document = deepcopy(dict(source))
        encoded = _canonical_json(document)
        if not encoded or len(encoded) + 1 > MAX_VIDEO_CAPTURE_RUN_BYTES:
            raise VideoCaptureRunArtifactError(
                "DOCUMENT_SIZE", "Video capture-run evidence exceeds its byte bound"
            )
        return document
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            size = path.stat().st_size
            if size <= 0 or size > MAX_VIDEO_CAPTURE_RUN_BYTES:
                raise VideoCaptureRunArtifactError(
                    "DOCUMENT_SIZE", "Video capture-run evidence exceeds its byte bound"
                )
            payload = path.read_bytes()
        except VideoCaptureRunArtifactError:
            raise
        except OSError as exc:
            raise VideoCaptureRunArtifactError(
                "DOCUMENT_MISSING", "Video capture-run evidence is unavailable"
            ) from exc
    else:
        payload = bytes(source)
    if not payload or len(payload) > MAX_VIDEO_CAPTURE_RUN_BYTES:
        raise VideoCaptureRunArtifactError(
            "DOCUMENT_SIZE", "Video capture-run evidence exceeds its byte bound"
        )
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except VideoCaptureRunArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise VideoCaptureRunArtifactError(
            "INVALID_JSON", "Video capture-run evidence must be strict UTF-8 JSON"
        ) from exc
    if not isinstance(document, dict):
        raise VideoCaptureRunArtifactError(
            "INVALID_DOCUMENT", "Video capture-run root must be an object"
        )
    if payload != _canonical_json(document) + b"\n":
        raise VideoCaptureRunArtifactError(
            "NONCANONICAL_JSON",
            "Video capture-run evidence must use canonical JSON with one final newline",
        )
    return document


def load_video_capture_run(
    source: str | Path | bytes | bytearray | Mapping[str, Any],
    *,
    expected_capture: CaptureTrack,
) -> VideoCaptureRun:
    """Strictly load and reconstruct persisted run evidence against Capture v1."""

    return _reconstruct_document(
        _read_document(source), expected_capture=expected_capture
    )


def write_video_capture_run(path: str | Path, run: VideoCaptureRun) -> Path:
    """Atomically write one canonical persisted detector-run artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = build_video_capture_run_document(run)
    payload = _canonical_json(document) + b"\n"
    if len(payload) > MAX_VIDEO_CAPTURE_RUN_BYTES:
        raise VideoCaptureRunArtifactError(
            "DOCUMENT_SIZE", "Video capture-run evidence exceeds its byte bound"
        )
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, destination)
    except Exception:
        handle.close()
        temporary.unlink(missing_ok=True)
        raise
    return destination


__all__ = [
    "DETECTOR_INGRESS_HASH_DOMAIN",
    "MAX_VIDEO_CAPTURE_RUN_BYTES",
    "MAX_VIDEO_CAPTURE_RUN_FRAMES",
    "VIDEO_CAPTURE_RUN_KIND",
    "VIDEO_CAPTURE_RUN_SCHEMA_VERSION",
    "VideoCaptureRunArtifactError",
    "build_video_capture_run_document",
    "load_video_capture_run",
    "video_capture_run_payload_sha256",
    "write_video_capture_run",
]
