"""Provider-neutral regional evidence for the VisualTrack V2a shadow lane.

The contract intentionally carries measurements and their uncertainty state
separately.  Missing calibration, occlusion, or residual evidence is encoded as
NaN plus an explicit UNKNOWN state; it is never silently promoted to zero.
The current adapter binds VisualTrack v1 exactly and grants no motion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from .errors import AutoAnimError
from .serialization import write_npz
from .visual_track import (
    CONFIDENCE_UNKNOWN,
    MOTION_AUTHORITY,
    OCCLUSION_MISSING,
    OCCLUSION_UNKNOWN,
    REGION_MISSING,
    REGION_NAMES,
    REGION_PROVISIONAL_OBSERVED,
    VisualTrack,
)


VISUAL_TRACK_PROVIDER_SCHEMA_VERSION = "autoanim.visual-track-provider-result/1.0"
VISUAL_TRACK_PROVIDER_POLICY = "exact_pts_regional_shadow_no_motion_effect_v2a"
MAX_PROVIDER_BYTES = 128 * 1024 * 1024
MAX_PROVIDER_UNCOMPRESSED_BYTES = 384 * 1024 * 1024

REGION_EVIDENCE_UNKNOWN = np.uint8(0)
REGION_EVIDENCE_OBSERVED = np.uint8(1)
_SHA256_HEX = frozenset("0123456789abcdef")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON member: {key}")
        result[key] = value
    return result


def parse_canonical_json(value: str) -> dict[str, Any]:
    parsed = json.loads(
        value,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON number: {item}")
        ),
    )
    if not isinstance(parsed, dict) or canonical_json(parsed) != value:
        raise ValueError("Provider metadata must be canonical JSON")
    return parsed


def valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _SHA256_HEX
    )


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _require_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Provider {label} members do not match the schema")
    return value


@dataclass(frozen=True, slots=True)
class VisualTrackProviderResult:
    """Immutable, exact-clock regional evidence from one provider adapter."""

    metadata_json: str
    source_pts: np.ndarray
    evidence_rgb_sha256: tuple[str, ...]
    detected: np.ndarray
    cut_candidate: np.ndarray
    shot_epoch_index: np.ndarray
    tracking_epoch_index: np.ndarray
    subject_epoch_index: np.ndarray
    region_names: tuple[str, ...]
    region_observation_state: np.ndarray
    region_support_score: np.ndarray
    region_confidence: np.ndarray
    region_confidence_state: np.ndarray
    region_occlusion_probability: np.ndarray
    region_occlusion_state: np.ndarray
    region_residual: np.ndarray
    region_residual_state: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "source_pts": (self.source_pts, np.int64),
            "detected": (self.detected, np.bool_),
            "cut_candidate": (self.cut_candidate, np.bool_),
            "shot_epoch_index": (self.shot_epoch_index, np.int32),
            "tracking_epoch_index": (self.tracking_epoch_index, np.int32),
            "subject_epoch_index": (self.subject_epoch_index, np.int32),
            "region_observation_state": (self.region_observation_state, np.uint8),
            "region_support_score": (self.region_support_score, np.float32),
            "region_confidence": (self.region_confidence, np.float32),
            "region_confidence_state": (self.region_confidence_state, np.uint8),
            "region_occlusion_probability": (
                self.region_occlusion_probability,
                np.float32,
            ),
            "region_occlusion_state": (self.region_occlusion_state, np.uint8),
            "region_residual": (self.region_residual, np.float32),
            "region_residual_state": (self.region_residual_state, np.uint8),
        }
        for name, (value, dtype) in arrays.items():
            object.__setattr__(self, name, _readonly(value, np.dtype(dtype)))
        object.__setattr__(self, "region_names", tuple(self.region_names))
        object.__setattr__(
            self, "evidence_rgb_sha256", tuple(self.evidence_rgb_sha256)
        )
        metadata = parse_canonical_json(self.metadata_json)
        count = len(self.source_pts)
        region_count = len(self.region_names)
        if count <= 0 or self.region_names != REGION_NAMES:
            raise ValueError("Provider frame or regional schema is invalid")
        for name in (
            "detected",
            "cut_candidate",
            "shot_epoch_index",
            "tracking_epoch_index",
            "subject_epoch_index",
        ):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"Provider {name} has invalid shape")
        for name in (
            "region_observation_state",
            "region_support_score",
            "region_confidence",
            "region_confidence_state",
            "region_occlusion_probability",
            "region_occlusion_state",
            "region_residual",
            "region_residual_state",
        ):
            if getattr(self, name).shape != (count, region_count):
                raise ValueError(f"Provider {name} has invalid shape")
        if len(self.evidence_rgb_sha256) != count or any(
            not valid_sha256(value) for value in self.evidence_rgb_sha256
        ):
            raise ValueError("Provider frame hashes are invalid")
        self._validate_metadata(metadata)
        self._validate_values(metadata)

    @property
    def metadata(self) -> dict[str, Any]:
        return parse_canonical_json(self.metadata_json)

    @property
    def frame_count(self) -> int:
        return len(self.source_pts)

    def _validate_metadata(self, metadata: dict[str, Any]) -> None:
        _require_keys(
            metadata,
            {
                "schema_version",
                "kind",
                "status",
                "policy",
                "motion_authority",
                "consumed_by_retargeting",
                "provider",
                "bindings",
                "timeline",
                "regions",
                "claims",
            },
            "metadata",
        )
        provider = _require_keys(
            metadata["provider"],
            {"adapter_id", "implementation", "model_name", "model_sha256", "runtime", "profile"},
            "provider",
        )
        bindings = _require_keys(
            metadata["bindings"],
            {
                "capture_source_sha256",
                "capture_model_sha256",
                "source_pts_sha256",
                "visual_track_v1_sha256",
                "visual_track_v1_summary_sha256",
            },
            "bindings",
        )
        timeline = _require_keys(
            metadata["timeline"],
            {
                "frame_count",
                "source_start_pts",
                "source_time_base",
                "strictly_increasing",
                "exact_pts_retained",
            },
            "timeline",
        )
        regions = _require_keys(
            metadata["regions"],
            {
                "order",
                "support_score_semantics",
                "confidence_semantics",
                "occlusion_semantics",
                "residual_semantics",
                "unknown_is_not_zero",
            },
            "regions",
        )
        claims = _require_keys(
            metadata["claims"],
            {
                "changes_final_gnm_motion",
                "confidence_calibrated",
                "occlusion_validated",
                "identity_continuity_validated",
                "tongue_observed",
                "production_validated",
            },
            "claims",
        )
        if (
            metadata["schema_version"] != VISUAL_TRACK_PROVIDER_SCHEMA_VERSION
            or metadata["kind"] != "visual_track_provider_result"
            or metadata["status"] != "shadow_unqualified"
            or metadata["policy"] != VISUAL_TRACK_PROVIDER_POLICY
            or metadata["motion_authority"] != MOTION_AUTHORITY
            or metadata["consumed_by_retargeting"] is not False
            or timeline["frame_count"] != self.frame_count
            or timeline["source_start_pts"] != int(self.source_pts[0])
            or timeline["strictly_increasing"] is not True
            or timeline["exact_pts_retained"] is not True
            or regions["order"] != list(REGION_NAMES)
            or regions["support_score_semantics"]
            != "provider_candidate_score_not_probability_until_calibrated"
            or regions["unknown_is_not_zero"] is not True
            or provider["profile"] != {"state": "absent", "sha256": None}
        ):
            raise ValueError("Provider metadata is inconsistent or not fail-closed")
        if not isinstance(provider["runtime"], dict) or not provider["runtime"]:
            raise ValueError("Provider runtime binding is invalid")
        time_base = timeline["source_time_base"]
        if (
            not isinstance(time_base, list)
            or len(time_base) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in time_base
            )
            or time_base[0] <= 0
            or time_base[1] <= 0
        ):
            raise ValueError("Provider source time base is invalid")
        for key in (
            "capture_source_sha256",
            "capture_model_sha256",
            "source_pts_sha256",
            "visual_track_v1_sha256",
            "visual_track_v1_summary_sha256",
        ):
            if not valid_sha256(bindings[key]):
                raise ValueError(f"Provider binding {key} is invalid")
        if (
            not valid_sha256(provider["model_sha256"])
            or provider["model_sha256"] != bindings["capture_model_sha256"]
        ):
            raise ValueError("Provider model binding is inconsistent")
        if bindings["source_pts_sha256"] != array_sha256(self.source_pts):
            raise ValueError("Provider source PTS hash does not reconstruct")
        if any(value is not False for value in claims.values()):
            raise ValueError("Provider claims must remain false")

    def _validate_values(self, metadata: dict[str, Any]) -> None:
        if self.frame_count > 1 and np.any(np.diff(self.source_pts) <= 0):
            raise ValueError("Provider source PTS must be strictly increasing")
        if self.cut_candidate[0]:
            raise ValueError("Provider first frame cannot start with a cut")
        expected_shot = np.cumsum(self.cut_candidate.astype(np.int32), dtype=np.int32)
        expected_tracking = np.full(self.frame_count, -1, dtype=np.int32)
        tracking_epoch = -1
        for index, present in enumerate(self.detected):
            if not bool(present):
                continue
            if (
                index == 0
                or not bool(self.detected[index - 1])
                or bool(self.cut_candidate[index])
            ):
                tracking_epoch += 1
            expected_tracking[index] = tracking_epoch
        if not np.array_equal(self.shot_epoch_index, expected_shot) or not np.array_equal(
            self.tracking_epoch_index, expected_tracking
        ):
            raise ValueError("Provider shot or tracking epochs are inconsistent")
        if np.any(self.subject_epoch_index != -1):
            raise ValueError("Unbound provider subject epochs must remain unknown")
        if np.any(self.region_observation_state > 2):
            raise ValueError("Provider observation states are invalid")
        support_finite = np.isfinite(self.region_support_score)
        if np.any(self.region_support_score[support_finite] < 0) or np.any(
            self.region_support_score[support_finite] > 1
        ):
            raise ValueError("Provider support scores must lie in [0,1]")
        expected_support = (
            self.region_observation_state == REGION_PROVISIONAL_OBSERVED
        )
        if not np.array_equal(support_finite, expected_support):
            raise ValueError(
                "Provider candidate support must be finite exactly for observed regions"
            )
        if not np.isnan(self.region_confidence).all() or np.any(
            self.region_confidence_state != CONFIDENCE_UNKNOWN
        ):
            raise ValueError("Uncalibrated provider confidence must remain unknown")
        if not np.isnan(self.region_occlusion_probability).all():
            raise ValueError("Unavailable provider occlusion must remain NaN")
        expected_occlusion = np.where(
            self.detected[:, None], OCCLUSION_UNKNOWN, OCCLUSION_MISSING
        ).astype(np.uint8)
        expected_occlusion = np.broadcast_to(
            expected_occlusion, self.region_occlusion_state.shape
        )
        if not np.array_equal(self.region_occlusion_state, expected_occlusion):
            raise ValueError("Provider occlusion states are inconsistent")
        if not np.isnan(self.region_residual).all() or np.any(
            self.region_residual_state != REGION_EVIDENCE_UNKNOWN
        ):
            raise ValueError("Unavailable provider residuals must remain unknown")
        if np.any(self.region_observation_state[~self.detected] != REGION_MISSING):
            raise ValueError("Missing frames cannot contain regional observations")

    def validate_visual_track(self, track: VisualTrack) -> None:
        source = track.metadata["source"]
        track_provider = track.metadata["provider"]
        bindings = self.metadata["bindings"]
        if (
            bindings["capture_source_sha256"] != source["sha256"]
            or bindings["capture_model_sha256"] != track_provider["model_sha256"]
        ):
            raise ValueError("Provider metadata differs from VisualTrack v1")
        if not np.array_equal(self.source_pts, track.source_pts):
            raise ValueError("Provider PTS differ from VisualTrack v1")
        if self.evidence_rgb_sha256 != track.evidence_rgb_sha256:
            raise ValueError("Provider frame hashes differ from VisualTrack v1")
        for name in (
            "detected",
            "cut_candidate",
            "shot_epoch_index",
            "tracking_epoch_index",
            "subject_epoch_index",
            "region_observation_state",
            "region_support_score",
        ):
            if not np.array_equal(
                getattr(self, name), getattr(track, name), equal_nan=True
            ):
                raise ValueError(f"Provider {name} differs from VisualTrack v1")

    def validate_visual_track_artifacts(
        self,
        *,
        visual_track_path: str | Path,
        visual_track_summary_path: str | Path,
    ) -> None:
        bindings = self.metadata["bindings"]
        if (
            bindings["visual_track_v1_sha256"] != file_sha256(visual_track_path)
            or bindings["visual_track_v1_summary_sha256"]
            != file_sha256(visual_track_summary_path)
        ):
            raise ValueError("Provider does not bind the supplied VisualTrack artifacts")


def build_visual_track_provider_result(
    track: VisualTrack,
    *,
    visual_track_v1_sha256: str,
    visual_track_v1_summary_sha256: str,
) -> VisualTrackProviderResult:
    """Adapt exact VisualTrack v1 evidence without inventing V2 measurements."""

    if not valid_sha256(visual_track_v1_sha256) or not valid_sha256(
        visual_track_v1_summary_sha256
    ):
        raise ValueError("VisualTrack v1 artifact hashes must be SHA-256")
    v1 = track.metadata
    source = v1["source"]
    provider = v1["provider"]
    count = track.frame_count
    region_count = len(REGION_NAMES)
    metadata = {
        "schema_version": VISUAL_TRACK_PROVIDER_SCHEMA_VERSION,
        "kind": "visual_track_provider_result",
        "status": "shadow_unqualified",
        "policy": VISUAL_TRACK_PROVIDER_POLICY,
        "motion_authority": MOTION_AUTHORITY,
        "consumed_by_retargeting": False,
        "provider": {
            "adapter_id": provider["adapter_id"],
            "implementation": provider["implementation"],
            "model_name": provider["model_name"],
            "model_sha256": provider["model_sha256"],
            "runtime": provider["runtime"],
            "profile": {"state": "absent", "sha256": None},
        },
        "bindings": {
            "capture_source_sha256": source["sha256"],
            "capture_model_sha256": provider["model_sha256"],
            "source_pts_sha256": array_sha256(track.source_pts),
            "visual_track_v1_sha256": visual_track_v1_sha256,
            "visual_track_v1_summary_sha256": visual_track_v1_summary_sha256,
        },
        "timeline": {
            "frame_count": count,
            "source_start_pts": int(track.source_pts[0]),
            "source_time_base": source["source_time_base"],
            "strictly_increasing": True,
            "exact_pts_retained": True,
        },
        "regions": {
            "order": list(REGION_NAMES),
            "support_score_semantics": (
                "provider_candidate_score_not_probability_until_calibrated"
            ),
            "confidence_semantics": (
                "unknown_until_independently_calibrated_probability_of_named_error_bound"
            ),
            "occlusion_semantics": "unknown_not_zero",
            "residual_semantics": "unknown_not_zero",
            "unknown_is_not_zero": True,
        },
        "claims": {
            "changes_final_gnm_motion": False,
            "confidence_calibrated": False,
            "occlusion_validated": False,
            "identity_continuity_validated": False,
            "tongue_observed": False,
            "production_validated": False,
        },
    }
    occlusion_state = np.broadcast_to(
        np.where(track.detected[:, None], OCCLUSION_UNKNOWN, OCCLUSION_MISSING),
        (count, region_count),
    )
    return VisualTrackProviderResult(
        metadata_json=canonical_json(metadata),
        source_pts=track.source_pts,
        evidence_rgb_sha256=track.evidence_rgb_sha256,
        detected=track.detected,
        cut_candidate=track.cut_candidate,
        shot_epoch_index=track.shot_epoch_index,
        tracking_epoch_index=track.tracking_epoch_index,
        subject_epoch_index=track.subject_epoch_index,
        region_names=REGION_NAMES,
        region_observation_state=track.region_observation_state,
        region_support_score=track.region_support_score,
        region_confidence=np.full((count, region_count), np.nan, dtype=np.float32),
        region_confidence_state=np.full(
            (count, region_count), CONFIDENCE_UNKNOWN, dtype=np.uint8
        ),
        region_occlusion_probability=np.full(
            (count, region_count), np.nan, dtype=np.float32
        ),
        region_occlusion_state=occlusion_state,
        region_residual=np.full((count, region_count), np.nan, dtype=np.float32),
        region_residual_state=np.full(
            (count, region_count), REGION_EVIDENCE_UNKNOWN, dtype=np.uint8
        ),
    )


_ARRAY_DTYPES = {
    "source_pts": np.dtype(np.int64),
    "evidence_rgb_sha256": np.dtype("<U64"),
    "detected": np.dtype(np.bool_),
    "cut_candidate": np.dtype(np.bool_),
    "shot_epoch_index": np.dtype(np.int32),
    "tracking_epoch_index": np.dtype(np.int32),
    "subject_epoch_index": np.dtype(np.int32),
    "region_names": np.dtype("<U32"),
    "region_observation_state": np.dtype(np.uint8),
    "region_support_score": np.dtype(np.float32),
    "region_confidence": np.dtype(np.float32),
    "region_confidence_state": np.dtype(np.uint8),
    "region_occlusion_probability": np.dtype(np.float32),
    "region_occlusion_state": np.dtype(np.uint8),
    "region_residual": np.dtype(np.float32),
    "region_residual_state": np.dtype(np.uint8),
}


def write_visual_track_provider_result(
    path: str | Path, result: VisualTrackProviderResult
) -> Path:
    return write_npz(
        path,
        metadata_json=np.asarray(result.metadata_json),
        source_pts=result.source_pts,
        evidence_rgb_sha256=np.asarray(result.evidence_rgb_sha256, dtype="<U64"),
        detected=result.detected,
        cut_candidate=result.cut_candidate,
        shot_epoch_index=result.shot_epoch_index,
        tracking_epoch_index=result.tracking_epoch_index,
        subject_epoch_index=result.subject_epoch_index,
        region_names=np.asarray(result.region_names, dtype="<U32"),
        region_observation_state=result.region_observation_state,
        region_support_score=result.region_support_score,
        region_confidence=result.region_confidence,
        region_confidence_state=result.region_confidence_state,
        region_occlusion_probability=result.region_occlusion_probability,
        region_occlusion_state=result.region_occlusion_state,
        region_residual=result.region_residual,
        region_residual_state=result.region_residual_state,
    )


def load_visual_track_provider_result(
    path: str | Path,
    *,
    expected_visual_track: VisualTrack | None = None,
    expected_visual_track_path: str | Path | None = None,
    expected_visual_track_summary_path: str | Path | None = None,
) -> VisualTrackProviderResult:
    artifact = Path(path)
    try:
        if artifact.stat().st_size <= 0 or artifact.stat().st_size > MAX_PROVIDER_BYTES:
            raise ValueError("Provider artifact size is outside its bounds")
        with zipfile.ZipFile(artifact) as archive:
            names = [item.filename for item in archive.infolist()]
            expected = {"metadata_json.npy", *{f"{name}.npy" for name in _ARRAY_DTYPES}}
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError("Provider artifact members do not match the schema")
            if sum(item.file_size for item in archive.infolist()) > MAX_PROVIDER_UNCOMPRESSED_BYTES:
                raise ValueError("Provider artifact exceeds its resource limit")
        with np.load(artifact, allow_pickle=False) as payload:
            arrays = {name: np.array(payload[name], copy=True) for name in _ARRAY_DTYPES}
            metadata_array = np.asarray(payload["metadata_json"])
        if metadata_array.shape != () or metadata_array.dtype.kind != "U":
            raise ValueError("Provider metadata member is invalid")
        for name, dtype in _ARRAY_DTYPES.items():
            if arrays[name].dtype != dtype:
                raise ValueError(f"Provider {name} dtype does not match the schema")
        result = VisualTrackProviderResult(
            metadata_json=str(metadata_array.item()),
            source_pts=arrays["source_pts"],
            evidence_rgb_sha256=tuple(arrays["evidence_rgb_sha256"].tolist()),
            detected=arrays["detected"],
            cut_candidate=arrays["cut_candidate"],
            shot_epoch_index=arrays["shot_epoch_index"],
            tracking_epoch_index=arrays["tracking_epoch_index"],
            subject_epoch_index=arrays["subject_epoch_index"],
            region_names=tuple(arrays["region_names"].tolist()),
            region_observation_state=arrays["region_observation_state"],
            region_support_score=arrays["region_support_score"],
            region_confidence=arrays["region_confidence"],
            region_confidence_state=arrays["region_confidence_state"],
            region_occlusion_probability=arrays["region_occlusion_probability"],
            region_occlusion_state=arrays["region_occlusion_state"],
            region_residual=arrays["region_residual"],
            region_residual_state=arrays["region_residual_state"],
        )
        if expected_visual_track is not None:
            result.validate_visual_track(expected_visual_track)
        artifact_paths = (
            expected_visual_track_path,
            expected_visual_track_summary_path,
        )
        if any(value is not None for value in artifact_paths):
            if any(value is None for value in artifact_paths):
                raise ValueError(
                    "Both VisualTrack artifact paths are required for binding validation"
                )
            result.validate_visual_track_artifacts(
                visual_track_path=expected_visual_track_path,
                visual_track_summary_path=expected_visual_track_summary_path,
            )
        return result
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise AutoAnimError("MEDIA_INVALID", f"Invalid VisualTrack provider result: {exc}") from exc


__all__ = [
    "REGION_EVIDENCE_OBSERVED",
    "REGION_EVIDENCE_UNKNOWN",
    "VISUAL_TRACK_PROVIDER_POLICY",
    "VISUAL_TRACK_PROVIDER_SCHEMA_VERSION",
    "VisualTrackProviderResult",
    "array_sha256",
    "build_visual_track_provider_result",
    "canonical_json",
    "file_sha256",
    "load_visual_track_provider_result",
    "valid_sha256",
    "write_visual_track_provider_result",
]
