"""Fail-closed 2D reprojection audits for learned body-motion providers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SOMA_ARM_JOINTS = {
    "left": (11, 12, 13, 14),
    "right": (39, 40, 41, 42),
}
SOMA_ARM_JOINT_NAMES = ("shoulder", "upper_arm", "forearm", "hand")


@dataclass(frozen=True, slots=True)
class ArmReprojectionAudit:
    passed: bool
    confidence_threshold: float
    p95_error_limit_bbox_height: float
    sign_mismatch_limit: float
    sides: dict[str, dict[str, object]]
    failure_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "autoanim.arm-reprojection-audit/1.0",
            "passed": self.passed,
            "confidence_threshold": self.confidence_threshold,
            "p95_error_limit_bbox_height": self.p95_error_limit_bbox_height,
            "sign_mismatch_limit": self.sign_mismatch_limit,
            "sides": self.sides,
            "failure_reasons": list(self.failure_reasons),
        }


def audit_soma_arm_reprojection(
    observed_keypoints_xyc: np.ndarray,
    projected_keypoints_xy: np.ndarray,
    bounding_boxes_xyxy: np.ndarray,
    *,
    confidence_threshold: float = 0.8,
    p95_error_limit_bbox_height: float = 0.08,
    sign_mismatch_limit: float = 0.10,
    minimum_raised_frames: int = 4,
) -> ArmReprojectionAudit:
    """Compare high-confidence 2D arm evidence with reconstructed SOMA joints.

    Pixel errors are normalized by the detected person's box height. The
    shoulder-to-wrist sign check catches the especially damaging case where a
    provider turns a clearly raised arm into a lowered one even if its global
    camera fit makes raw pixel errors hard to interpret.
    """

    observed = np.asarray(observed_keypoints_xyc, dtype=np.float64)
    projected = np.asarray(projected_keypoints_xy, dtype=np.float64)
    boxes = np.asarray(bounding_boxes_xyxy, dtype=np.float64)
    frame_count = observed.shape[0] if observed.ndim else 0
    if observed.shape != (frame_count, 77, 3):
        raise ValueError("Observed SOMA keypoints must have shape [frames, 77, 3]")
    if projected.shape != (frame_count, 77, 2):
        raise ValueError("Projected SOMA keypoints must have shape [frames, 77, 2]")
    if boxes.shape != (frame_count, 4):
        raise ValueError("Body bounding boxes must have shape [frames, 4]")
    if frame_count < 2:
        raise ValueError("Arm reprojection audit requires at least two frames")
    if not all(
        np.isfinite(value).all() for value in (observed, projected, boxes)
    ):
        raise ValueError("Arm reprojection inputs must be finite")
    if not 0.0 < confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be in (0, 1]")
    if not 0.0 < p95_error_limit_bbox_height < 1.0:
        raise ValueError("Reprojection p95 limit must be in (0, 1)")
    if not 0.0 <= sign_mismatch_limit < 1.0:
        raise ValueError("Sign mismatch limit must be in [0, 1)")
    if minimum_raised_frames < 1:
        raise ValueError("Minimum raised-frame count must be positive")

    heights = boxes[:, 3] - boxes[:, 1]
    if np.any(heights <= 1.0):
        raise ValueError("Every body bounding box must have positive pixel height")

    sides: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for side, indices in SOMA_ARM_JOINTS.items():
        joint_metrics: dict[str, object] = {}
        for name, index in zip(SOMA_ARM_JOINT_NAMES, indices, strict=True):
            visible = observed[:, index, 2] >= confidence_threshold
            visible_count = int(np.count_nonzero(visible))
            if not visible_count:
                joint_metrics[name] = {
                    "visible_frames": 0,
                    "p95_error_bbox_height": None,
                }
                continue
            error = np.linalg.norm(
                projected[visible, index] - observed[visible, index, :2],
                axis=1,
            ) / heights[visible]
            p95 = float(np.percentile(error, 95.0))
            joint_metrics[name] = {
                "visible_frames": visible_count,
                "mean_confidence": float(np.mean(observed[visible, index, 2])),
                "p95_error_bbox_height": p95,
            }
            if p95 > p95_error_limit_bbox_height:
                failures.append(f"{side}_{name}_p95_reprojection")

        shoulder, _, _, hand = indices
        sign_visible = (
            (observed[:, shoulder, 2] >= confidence_threshold)
            & (observed[:, hand, 2] >= confidence_threshold)
        )
        observed_rise = observed[:, shoulder, 1] - observed[:, hand, 1]
        projected_rise = projected[:, shoulder, 1] - projected[:, hand, 1]
        raised = sign_visible & (observed_rise > 0.02 * heights)
        raised_count = int(np.count_nonzero(raised))
        mismatch_fraction: float | None = None
        if raised_count >= minimum_raised_frames:
            mismatch_fraction = float(np.mean(projected_rise[raised] <= 0.0))
            if mismatch_fraction > sign_mismatch_limit:
                failures.append(f"{side}_raised_hand_sign_mismatch")
        sides[side] = {
            "joints": joint_metrics,
            "raised_observation_frames": raised_count,
            "raised_sign_mismatch_fraction": mismatch_fraction,
        }

    return ArmReprojectionAudit(
        passed=not failures,
        confidence_threshold=confidence_threshold,
        p95_error_limit_bbox_height=p95_error_limit_bbox_height,
        sign_mismatch_limit=sign_mismatch_limit,
        sides=sides,
        failure_reasons=tuple(failures),
    )


__all__ = [
    "ArmReprojectionAudit",
    "SOMA_ARM_JOINTS",
    "SOMA_ARM_JOINT_NAMES",
    "audit_soma_arm_reprojection",
]
