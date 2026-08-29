"""Candidate construction and transparent diagnostics for learned body acting."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np

from .body import DETAILED_HUMANOID, BodyTrack
from .body_projection import (
    BodyProjectionDiagnostics,
    RootStabilizationDiagnostics,
    constrain_restrained_root_travel,
    project_generated_foot_contacts,
)
from .speech_motion import NativeSpeechMotion, retarget_smplx55_to_autoanim55


@dataclass(frozen=True, slots=True)
class SpeechMotionCandidate:
    seed: int
    raw_track: BodyTrack
    projected_track: BodyTrack
    root_stabilization: RootStabilizationDiagnostics
    projection: BodyProjectionDiagnostics
    diagnostics: dict[str, float | int | bool]

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "raw_track_sha256": sha256(
                self.raw_track.canonical_json_bytes()
            ).hexdigest(),
            "projected_track_sha256": sha256(
                self.projected_track.canonical_json_bytes()
            ).hexdigest(),
            "root_stabilization": self.root_stabilization.as_dict(),
            "projection": self.projection.as_dict(),
            "diagnostics": dict(self.diagnostics),
            "approved": False,
        }


def _rotation_span(track: BodyTrack) -> np.ndarray:
    reference = track.local_rotations_xyzw[0]
    dots = np.clip(
        np.abs(np.sum(track.local_rotations_xyzw * reference[None, :, :], axis=2)),
        0.0,
        1.0,
    )
    return 2.0 * np.arccos(np.min(dots, axis=0))


def _slerp_samples(
    values: np.ndarray,
    source_ticks: np.ndarray,
    target_ticks: np.ndarray,
) -> np.ndarray:
    """Shortest-arc quaternion interpolation over an integer source clock."""

    right = np.searchsorted(source_ticks, target_ticks, side="right")
    right = np.clip(right, 1, len(source_ticks) - 1)
    left = right - 1
    span = (source_ticks[right] - source_ticks[left]).astype(np.float64)
    alpha = ((target_ticks - source_ticks[left]).astype(np.float64) / span).reshape(
        (-1,) + (1,) * (values.ndim - 1)
    )
    first = np.asarray(values[left], dtype=np.float64)
    second = np.asarray(values[right], dtype=np.float64).copy()
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0.0, -second, second)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    near = sine < 1e-7
    safe_sine = np.where(near, 1.0, sine)
    first_weight = np.where(
        near, 1.0 - alpha, np.sin((1.0 - alpha) * angle) / safe_sine
    )
    second_weight = np.where(near, alpha, np.sin(alpha * angle) / safe_sine)
    output = first_weight * first + second_weight * second
    output /= np.linalg.norm(output, axis=-1, keepdims=True)
    for frame in range(1, len(output)):
        flip = np.sum(output[frame] * output[frame - 1], axis=-1) < 0.0
        output[frame, flip] *= -1.0
    return output.astype(np.float32)


def resample_generated_body_track(
    track: BodyTrack, target_ticks: np.ndarray
) -> BodyTrack:
    """Put learned motion on the authoritative face clock before composition.

    Contacts are derived state, so they are cleared here and must be projected
    again after interpolation. This avoids asserting stationary feet from a
    different sample grid.
    """

    ticks = np.asarray(target_ticks)
    if (
        ticks.dtype.kind not in "iu"
        or ticks.ndim != 1
        or len(ticks) < 2
        or ticks[0] != 0
        or ticks[-1] != track.duration_ticks
        or np.any(np.diff(ticks) <= 0)
    ):
        raise ValueError("Target ticks must span the exact body duration")
    ticks = ticks.astype(np.int64, copy=False)
    source = track.ticks.astype(np.float64)
    query = ticks.astype(np.float64)

    def linear(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        flat = array.reshape(len(source), -1)
        output = np.stack(
            [np.interp(query, source, flat[:, channel]) for channel in range(flat.shape[1])],
            axis=1,
        )
        return output.reshape((len(query),) + array.shape[1:]).astype(np.float32)

    direction = linear(track.gaze_direction_body)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    provenance = sha256()
    provenance.update(b"autoanim.body-track-resample/1.0\0")
    provenance.update(track.source_plan_sha256.encode("ascii"))
    provenance.update(np.ascontiguousarray(ticks).tobytes())
    return BodyTrack(
        duration_ticks=track.duration_ticks,
        ticks_per_second=track.ticks_per_second,
        sample_rate_hz=track.sample_rate_hz,
        joint_names=track.joint_names,
        ticks=ticks,
        root_translation_m=linear(track.root_translation_m),
        local_rotations_xyzw=_slerp_samples(
            track.local_rotations_xyzw, track.ticks, ticks
        ),
        foot_contacts=np.zeros((len(ticks), 2), dtype=np.bool_),
        gaze_direction_body=direction,
        gaze_strength=np.clip(linear(track.gaze_strength), 0.0, 1.0),
        gnm_eye_rotations_xyzw=_slerp_samples(
            track.gnm_eye_rotations_xyzw, track.ticks, ticks
        ),
        source_plan_sha256=provenance.hexdigest(),
    )


def _diagnostics(track: BodyTrack) -> dict[str, float | int | bool]:
    span = _rotation_span(track)
    index = {name: value for value, name in enumerate(track.joint_names)}
    finger_indices = [
        value
        for name, value in index.items()
        if any(token in name for token in ("Thumb", "Index", "Middle", "Ring", "Little"))
    ]
    named = {
        "left_wrist_span_rad": span[index["LeftHand"]],
        "right_wrist_span_rad": span[index["RightHand"]],
        "chest_span_rad": span[index["Chest"]],
        "pelvis_span_rad": span[index["Hips"]],
    }
    root_range = np.ptp(track.root_translation_m, axis=0)
    return {
        "nonconstant_joint_count": int(np.sum(span > 1e-4)),
        "active_finger_joint_count": int(np.sum(span[finger_indices] > 1e-4)),
        "root_travel_m": float(np.linalg.norm(root_range)),
        **{name: float(value) for name, value in named.items()},
        "hand_generation_pass": bool(np.sum(span[finger_indices] > 1e-4) >= 20),
        "body_motion_pass": bool(
            max(named["left_wrist_span_rad"], named["right_wrist_span_rad"]) > 1e-4
            and named["chest_span_rad"] > 1e-4
            and named["pelvis_span_rad"] > 1e-4
        ),
    }


def build_speech_motion_candidates(
    motions: Iterable[NativeSpeechMotion],
) -> tuple[SpeechMotionCandidate, ...]:
    """Retarget, stabilize, diagnose and rank candidates without approving one."""

    candidates: list[SpeechMotionCandidate] = []
    seen_seeds: set[int] = set()
    seen_hashes: set[str] = set()
    for motion in motions:
        if motion.candidate_seed in seen_seeds:
            raise ValueError("Candidate seeds must be unique")
        seen_seeds.add(motion.candidate_seed)
        raw = retarget_smplx55_to_autoanim55(motion)
        digest = sha256()
        digest.update(np.ascontiguousarray(raw.root_translation_m).tobytes())
        digest.update(np.ascontiguousarray(raw.local_rotations_xyzw).tobytes())
        raw_hash = digest.hexdigest()
        if raw_hash in seen_hashes:
            raise ValueError("GestureLSM produced duplicate candidates")
        seen_hashes.add(raw_hash)
        restrained, root_stabilization = constrain_restrained_root_travel(raw)
        projected, projection = project_generated_foot_contacts(restrained)
        candidates.append(
            SpeechMotionCandidate(
                seed=motion.candidate_seed,
                raw_track=raw,
                projected_track=projected,
                root_stabilization=root_stabilization,
                projection=projection,
                diagnostics=_diagnostics(projected),
            )
        )
    candidates.sort(
        key=lambda item: (
            bool(item.diagnostics["body_motion_pass"]),
            bool(item.diagnostics["hand_generation_pass"]),
            int(item.diagnostics["nonconstant_joint_count"]),
        ),
        reverse=True,
    )
    return tuple(candidates)


def candidate_report(candidates: Iterable[SpeechMotionCandidate]) -> dict[str, object]:
    values = tuple(candidates)
    return {
        "schema_version": "autoanim.speech-motion-candidate-set/1.0",
        "production_validated": False,
        "automatic_approval": False,
        "candidate_count": len(values),
        "candidates": [candidate.as_dict() for candidate in values],
    }


__all__ = [
    "SpeechMotionCandidate",
    "build_speech_motion_candidates",
    "candidate_report",
    "resample_generated_body_track",
]
