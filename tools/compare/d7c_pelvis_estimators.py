#!/usr/bin/env python3
"""D7c: the two RIG-REST pelvis estimators and the pelvis lever guard, instrument-side.

These are written HERE, before `src/` moves, because the card requires selector S to run and
choose between them BEFORE the source change is made. The src change then implements exactly
these as `_pelvis_world_frames`' `D_rig_rest_hipline` and `E_rig_rest_kabsch` branches, and
`tests/test_pelvis_rest.py` asserts the two agree BIT-FOR-BIT on S's own frozen draws --
otherwise S selected one estimator and production ships another.

WHAT IS AND IS NOT A CONSTANT HERE. Every number these functions use comes from the caller's
own `rest` (the per-performer skeleton D3 stamps on the track) and from the observation. The
only literal is `SEGMENT_LENGTH_CEILING_FRACTION`, which is D8b's 0.15 and is NOT re-selected
in this step; the guard's denominator is the subject's OWN pre-guard median lever, frozen from
the input array before any frame is discarded, so D8b's moving-denominator defect (CLAUDE.md:
`captured_limb_stability.py` recomputes the performer's median per build) cannot enter.

(b) `D_rig_rest_hipline` -- the observed hip line is the EXACT primary axis and Spine1 sets
    only the pitch about it. `_frame_alignment` normalises both source axes, so the rest's
    LENGTHS do not weight this fit at all; the rig's rest supplies two DIRECTIONS.

(a) `E_rig_rest_kabsch` -- the C branch's un-centred rotation-only SVD, with the SOMA-77 rest
    template replaced by the rig's own rest offsets and the `root` landmark replaced by the
    observed hip midpoint. Here the rest lengths DO weight the fit, which is the whole
    difference: a same-length rotation of the observed hip line costs (b) its full angle and
    costs (a) less, because the 197 mm spine lever pulls against it.

Neither reads SOMA-77's `root` landmark (a pelvis-interior joint 74-77 mm above the hip
midpoint on this take, SOMA's own convention), and neither reads any `SOMA77_REST_*` constant.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "scripts"):
    if str(ROOT / _relative) not in sys.path:
        sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

from autoanim_gnm import commercial_multiview as cm  # noqa: E402

RIG_MODES = ("D_rig_rest_hipline", "E_rig_rest_kabsch")


def rest_template_about_hip_midpoint(rest: dict[str, np.ndarray]) -> np.ndarray:
    """`{Spine, LeftUpperLeg, RightUpperLeg}` rest offsets taken about the leg-root midpoint.

    This is the rig's OWN pelvis triangle. On an exact rig it is congruent to the observed
    one by construction: forward kinematics puts the leg roots at
    ``root + rest[Hips] + R . rest[L|R]`` and `Spine` at ``root + rest[Hips] + R . rest[Spine]``,
    so subtracting the leg-root midpoint from both leaves ``R`` acting on exactly these three
    vectors. That congruence is what makes O1 an exactness clause and not a fit quality.
    """
    mid = 0.5 * (np.asarray(rest["LeftUpperLeg"], dtype=np.float64)
                 + np.asarray(rest["RightUpperLeg"], dtype=np.float64))
    return np.stack((np.asarray(rest["Spine"], dtype=np.float64) - mid,
                     np.asarray(rest["LeftUpperLeg"], dtype=np.float64) - mid,
                     np.asarray(rest["RightUpperLeg"], dtype=np.float64) - mid))


def pelvis_lever_guard(
    spine: np.ndarray,
    hip_mid: np.ndarray,
    ceiling_fraction: float,
) -> dict:
    """Discard the Spine1 sample on frames whose pelvis LEVER is off the subject's median.

    A NEW MECHANISM, not D8b/D8c's demote. D8b demotes a landmark's RAYS and lets the
    sequence solve keep them; this discards a solved WORLD POINT and hands the gap to the
    spine's existing `np.interp` path. The shared 0.15 ceiling validates nothing about it,
    which is why S scores the guard against synthetic truth with its gaps.

    The denominator is FROZEN: the median is taken over the finite frames of the input array
    as it arrives, before any frame is discarded, so the median cannot move with the mask it
    produces. `np.median` over an array carrying NaN returns NaN and would make the ceiling
    infinite and the mask empty -- a silent pass -- so the finite mask is explicit.

    Returns the mask and the frozen figures; it does NOT modify its inputs.
    """
    spine = np.asarray(spine, dtype=np.float64)
    hip_mid = np.asarray(hip_mid, dtype=np.float64)
    finite = np.isfinite(spine).all(axis=1)
    lever = np.full(len(spine), np.nan)
    lever[finite] = np.linalg.norm(spine[finite] - hip_mid[finite], axis=1)
    if not finite.any():
        return {"median_m": float("nan"), "off": np.zeros(len(spine), bool),
                "lever_m": lever, "finite": finite, "ceiling_fraction": ceiling_fraction}
    median = float(np.median(lever[finite]))
    off = np.zeros(len(spine), bool)
    if median > 0.0:
        off[finite] = np.abs(lever[finite] / median - 1.0) > ceiling_fraction
    return {"median_m": median, "off": off, "lever_m": lever, "finite": finite,
            "ceiling_fraction": ceiling_fraction}


def rig_rest_pelvis_frames(
    points: np.ndarray,
    spine: np.ndarray,
    rest: dict[str, np.ndarray],
    *,
    mode: str,
    guard: bool,
    ceiling_fraction: float = cm.SEGMENT_LENGTH_CEILING_FRACTION,
    minimum_resolved_fraction: float = cm.PELVIS_MINIMUM_RESOLVED_FRACTION,
) -> tuple[np.ndarray | None, dict]:
    """`Hips`' world rotation per frame from the RIG's own rest. The reference implementation.

    `points` and `spine` are in the RIG's Y-up world, exactly as `_pelvis_world_frames`
    receives them. The order of operations matches the shipped function line for line:
    the guard (new), then the resolved-fraction whole-subject fallback, then the `np.interp`
    fill, then the per-frame construction, then the hemisphere walk.
    """
    if mode not in RIG_MODES:
        raise ValueError(f"not a rig-rest mode: {mode!r}")
    points = np.asarray(points, dtype=np.float64)
    spine = np.asarray(spine, dtype=np.float64)
    frames = len(points)
    left_hip = points[:, cm.JOINT_INDEX["left_hip"]]
    right_hip = points[:, cm.JOINT_INDEX["right_hip"]]
    hip_mid = 0.5 * (left_hip + right_hip)

    report: dict = {"mode": mode, "guard": bool(guard)}
    if guard:
        guarded = pelvis_lever_guard(spine, hip_mid, ceiling_fraction)
        spine = spine.copy()
        spine[guarded["off"]] = np.nan
        report["lever_guard"] = {
            "ceiling_fraction": ceiling_fraction,
            "pre_guard_median_mm": round(1e3 * guarded["median_m"], 4),
            "demoted_frames": [int(i) for i in np.flatnonzero(guarded["off"])],
            "demoted_count": int(guarded["off"].sum()),
        }
    else:
        report["lever_guard"] = {"ceiling_fraction": None, "demoted_frames": [],
                                 "demoted_count": 0, "note": "guard disabled (ablation arm)"}

    valid = np.isfinite(spine).all(axis=1)
    resolved = float(valid.mean())
    report["resolved_fraction"] = resolved
    if resolved < minimum_resolved_fraction or int(valid.sum()) < 2:
        report["status"] = "fell_back_to_torso_frame"
        return None, report

    filled = spine.copy()
    axis = np.arange(frames, dtype=np.float64)
    for component in range(3):
        filled[:, component] = np.interp(axis, axis[valid], spine[valid, component])

    mid = 0.5 * (np.asarray(rest["LeftUpperLeg"], dtype=np.float64)
                 + np.asarray(rest["RightUpperLeg"], dtype=np.float64))
    quaternions = np.zeros((frames, 4), dtype=np.float64)
    if mode == "D_rig_rest_hipline":
        source_primary = (np.asarray(rest["LeftUpperLeg"], dtype=np.float64)
                          - np.asarray(rest["RightUpperLeg"], dtype=np.float64))
        source_secondary = np.asarray(rest["Spine"], dtype=np.float64) - mid
        for frame in range(frames):
            # `_frame` orthogonalises its SECONDARY in place (`second -= first * dot`,
            # :2078 -- `np.asarray` does not copy a float64 array), so a persistent source
            # vector handed to it every frame would be mutated on the first one. Hand it a
            # fresh copy; the src branch does the same, and the bit-parity test pins it.
            quaternions[frame] = cm._frame_alignment(
                source_primary, np.array(source_secondary),
                left_hip[frame] - right_hip[frame], filled[frame] - hip_mid[frame])
    else:
        template = rest_template_about_hip_midpoint(rest)
        for frame in range(frames):
            observed = np.stack((filled[frame] - hip_mid[frame],
                                 left_hip[frame] - hip_mid[frame],
                                 right_hip[frame] - hip_mid[frame]))
            u, _, vt = np.linalg.svd(template.T @ observed)
            sign = float(np.sign(np.linalg.det(vt.T @ u.T)))
            rotation = vt.T @ np.diag((1.0, 1.0, sign)) @ u.T
            quaternions[frame] = Rotation.from_matrix(rotation).as_quat()

    for frame in range(1, frames):
        if float(np.dot(quaternions[frame], quaternions[frame - 1])) < 0.0:
            quaternions[frame] *= -1.0
    report["status"] = "solved"
    report["smoothing_frames"] = 0
    report["interpolated_frames"] = int(frames - int(valid.sum()))
    return quaternions, report


def frozen_pitch_hipline_frames(points: np.ndarray) -> np.ndarray:
    """S's control LAW, and it carries no constant and no rest.

    The observed hip line is the exact primary axis, exactly as (b) makes it; the up axis is
    WORLD +Y orthogonalised against that hip line. It therefore follows the hip line on every
    frame and its pitch about that line is frozen at zero relative to gravity -- it never
    reads Spine1, so missing data cannot arise. It is the control that leg-root placement
    cannot discriminate: it puts the leg roots on the captured hips exactly as (b) does.
    """
    points = np.asarray(points, dtype=np.float64)
    left_hip = points[:, cm.JOINT_INDEX["left_hip"]]
    right_hip = points[:, cm.JOINT_INDEX["right_hip"]]
    quaternions = np.zeros((len(points), 4))
    for frame in range(len(points)):
        across = left_hip[frame] - right_hip[frame]
        # `_frame_alignment`'s own `_frame` orthogonalises the secondary against the primary,
        # so handing it world +Y as the secondary IS "world vertical, orthogonalised". The
        # array is rebuilt per frame because `_frame` mutates its secondary in place.
        quaternions[frame] = cm._frame_alignment((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                                 across, np.array((0.0, 1.0, 0.0)))
    for frame in range(1, len(points)):
        if float(np.dot(quaternions[frame], quaternions[frame - 1])) < 0.0:
            quaternions[frame] *= -1.0
    return quaternions


def world_vertical_frames(points: np.ndarray) -> np.ndarray:
    """D7's own control: world +Y is the PRIMARY axis, the observed hip line the secondary."""
    points = np.asarray(points, dtype=np.float64)
    left_hip = points[:, cm.JOINT_INDEX["left_hip"]]
    right_hip = points[:, cm.JOINT_INDEX["right_hip"]]
    quaternions = np.zeros((len(points), 4))
    for frame in range(len(points)):
        quaternions[frame] = cm._frame_alignment(
            (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
            np.array((0.0, 1.0, 0.0)), left_hip[frame] - right_hip[frame])
    for frame in range(1, len(points)):
        if float(np.dot(quaternions[frame], quaternions[frame - 1])) < 0.0:
            quaternions[frame] *= -1.0
    return quaternions
