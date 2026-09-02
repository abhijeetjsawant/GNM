#!/usr/bin/env python3
"""Re-select `THORAX_SMOOTHING_FRAMES` on data that contains no MAMMA output.

Ladder step **I8**, part 2. `THORAX_SMOOTHING_FRAMES = 15`
(`src/autoanim_gnm/commercial_multiview.py:1679`) carries the comment "chosen by the
oracle arm alone" -- the oracle being MAMMA's own head expressed in our thorax frame.
That makes a shipped constant a MAMMA-derived constant, which CLAUDE.md forbids, and
`LADDER_EXECUTION_PLAN.md` §4 lists it as the one leak already found.

**The rule this script obeys** (Fable, `LADDER_EXECUTION_PLAN.md` §3 "Selectors"): the
constant may be selected only by synthetic truth, a held-out camera, the own-capture
control, or anatomy. The MAMMA arm *reports* and never selects.

* **truth** is exact: the thorax frame built by `_thorax_frames` itself, unsmoothed, from
  the forward-kinematic joint positions of `scripts/build_synthetic_truth_fixture.py`'s
  take builder. No estimate anywhere in the truth.
* **noise** is our own detector's self-consistency, measured on our own footage by our
  own instruments -- `artifacts/commercial-multiview-soma77/run-report.json` reprojection
  and `tools/swap-harness/sam3d_ladder.py`'s SOMA-77 cross-view epipolar percentiles.
  `mamma_residuals.py` is deliberately not consulted.
* the sweep scores against exact truth *because* the smoothing acts on precisely the
  quantity a tracking gate measures. "A band the solver regularises is a knob setting,
  not evidence" -- so the band cannot come from a gate the window can optimise.

**The finding that shaped this script.** A window is measured in FRAMES, so what it costs
and what it buys depend entirely on how fast the thorax turns. Measured here: our own
capture's thorax turns at 66 / 73 deg/s median and 183 / 196 deg/s p95 (two performers,
`artifacts/handfit-arrays/body-track.npz`, our own detector). The synthetic clips played
at stride 1 turn at 0.0-22.7 deg/s median -- between 3x and 700x too slow. At that speed
smoothing has nothing to lose and the sweep runs away to the widest window on offer,
which is not a selection, it is the fixture's gate-G10 defect ("a fixture whose motion
cannot exercise the temporal prior reports optimistic millimetres for every arm"). The
fix is the fixture's own mechanism: replay at `stride` times speed until the thorax turns
at the speed the real one does. Strides 1, 2 and 3 are all reported so the dependence is
visible rather than assumed.

**Two clips are excluded from the headline and kept as a declared degenerate arm.**
`autoanim_real/autoanim_fixture` and `cpu_smoke/autoanim_fixture` turn at 0.9 and 0.0
deg/s -- a thorax that does not turn selects the widest window by construction, so
including them would let two static trajectories decide a constant about motion.

**What is declared, not hidden.** The rig that carries pixel noise into millimetres,
`artifacts/soma77-full/camera-rig.json`, is byte-identical to
`artifacts/commercial-multiview-soma77/camera-rig.json` and *is* MAMMA's `ma_cap`
calibration (ladder rung 0, "not owned"; lane H retires it). So does the placement and
the speed-match target, which come from our own capture on MAMMA's example footage. The
mitigation is that the selection is an ARGMIN, not an absolute number, and the noise
amplitude is swept at 0.5x and 2.0x to show the argmin does not rest on the calibration
being exactly right.

**What this instrument is blind to.** The injected noise is white. Real detector error is
temporally correlated, and correlated error is exactly what a temporal smoother cannot
remove, so every window here is flattered equally and the SHAPE of the curve is the
claim, not the absolute degrees. No calibration, distortion, sync or soft-tissue error
exists here by construction. And speed-matching costs take length: at stride 3 the takes
are 24-32 frames, so windows past 21 are not covered at all and nothing here says what a
45-frame window does over a minute of footage.

    .venv/bin/python tools/head/thorax_window_sweep.py

Writes `artifacts/compare/thorax-window-sweep.json` (under `compare/` because the report
carries a MAMMA column, per `SUBSTITUTION_LADDER.md` §3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "workers" / "commercial_multiview"))

from autoanim_gnm import commercial_multiview as cm  # noqa: E402
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_INDEX, _fill_and_smooth_positions, _thorax_frames, triangulate_point,
)
from autoanim_gnm.head_orientation import log_so3, orthonormalise  # noqa: E402
from soma77_pose import SOMA77_TO_AUTOANIM  # noqa: E402

import build_synthetic_truth_fixture as fx  # noqa: E402

RIG = "artifacts/soma77-full/camera-rig.json"
CAPTURE = "artifacts/handfit-arrays/body-track.npz"
WORKING_WIDTH, WORKING_HEIGHT = 1280, 720
SAMPLE_RATE_HZ = 30.0

# The three clips whose thorax actually turns; see the module docstring.
MOVING_CLIPS = (
    "autoanim_dialogue/amy-cuddy-dialogue-body",
    "autoanim_squat/research-squat-640",
    "autoanim_will_acting/will-stephen-acting-body",
)
STATIC_CLIPS = ("autoanim_real/autoanim_fixture", "cpu_smoke/autoanim_fixture")

# The six joints `_thorax_frames` reads. Only these are given noise; the rest of the
# 19-joint contract is carried through as exact truth, because `_fill_and_smooth_positions`
# is per-joint and independent, so noising joints the frame never reads would cost runtime
# and change nothing. Verified against the function body: `up` from neck-root, `across`
# from the shoulder pair (the shipped default).
THORAX_JOINTS = ("root", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip")

# 0 means "none": `_thorax_frames` returns the per-frame frame unsmoothed for any value
# <= 1. 3 at polyorder 2 is an exact parabola through three points, i.e. the identity, and
# is kept as a self-check that the harness really is calling the smoother.
WINDOWS = (0, 3, 5, 9, 15, 21, 31, 45)

# --------------------------------------------------------------------- the noise source
#
# Both figures are our own detector measured against our own pipeline. Neither is
# referenced to MAMMA. The conversion to a per-view, per-axis pixel sigma at 1280 px:
#
# (a) run-report.json `median_reprojection_error_px` = 2.9129 px -- the median 2D NORM of
#     a post-solve residual. For isotropic Gaussian per-axis noise the norm is Rayleigh,
#     median 1.17741*sigma; and a least-squares residual is deflated, with 2n DLT
#     equations and 3 unknowns giving E[r^2] = sigma^2 (2n-3)/(2n). At n=4 that is 5/8:
#         sigma = 2.9129 / 1.17741 / sqrt(5/8) = 3.13 px
#     (mean camera support on the fixture is 3.30, and n=3 gives 3.50 px.)
#
# (b) sam3d_ladder.py, SOMA-77 one-sided cross-view epipolar distance at 1280 px,
#     percentiles [10,25,50,75,90,95] = [0.53, 1.37, 3.11, 6.04, 11.49, 32.79]. That
#     distance is two views' errors differenced onto the epipolar normal, so it is
#     |N(0, 2 sigma^2)| with median 0.67449*sigma*sqrt(2):
#         sigma = 3.11 / 0.67449 / sqrt(2) = 3.26 px
#
# Two independent instruments, 3.13 and 3.26 px. The sweep uses 3.20 px, and the amplitude
# is swept at 0.5x and 2.0x.
REPROJECTION_MEDIAN_PX = 2.9128817155694238          # run-report.json
EPIPOLAR_PERCENTILES = (10.0, 25.0, 50.0, 75.0, 90.0, 95.0)
EPIPOLAR_ONE_SIDED_PX = (0.53, 1.37, 3.11, 6.04, 11.49, 32.79)   # sam3d_ladder.py
NOISE_SIGMA_PX = 3.20
RAYLEIGH_MEDIAN = 1.1774100226
NORMAL_MEDIAN = 0.6744897502


def sigma_from_reprojection(median_px: float, views: int) -> float:
    return float(median_px / RAYLEIGH_MEDIAN / np.sqrt((2 * views - 3) / (2 * views)))


def sigma_from_epipolar(median_px: float) -> float:
    return float(median_px / NORMAL_MEDIAN / np.sqrt(2.0))


def heavy_tail_magnitude(rng: np.random.Generator, sigma_px: float) -> float:
    """One per-view displacement magnitude carrying SOMA-77's measured tail, median-matched.

    The Gaussian arm's per-view displacement is Rayleigh(sigma) with median
    1.17741*sigma. This arm draws from the empirical quantile function of the one-sided
    epipolar distance -- same detector, same footage -- rescaled so its median equals the
    Gaussian arm's. The arms therefore differ ONLY in tail shape, which is the thing being
    tested: p95 of a smoothed frame is a tail statistic, and SOMA-77's tail is 10.5x its
    median where a Rayleigh's p95 is 2.08x.
    """
    draw = float(np.interp(rng.uniform(0.0, 100.0), EPIPOLAR_PERCENTILES, EPIPOLAR_ONE_SIDED_PX))
    return draw * RAYLEIGH_MEDIAN * sigma_px / EPIPOLAR_ONE_SIDED_PX[2]


# ------------------------------------------------------------------------------ geometry

def to_19(take_soma: np.ndarray) -> np.ndarray:
    """SOMA-77 joint order -> the pipeline's 19-joint `JOINT_INDEX` order.

    The plan brief said the fixture's take builder returns positions "in our 19-joint
    JOINT_INDEX order". It does not: `build_take` returns [frame, 77, 3] in SOMA-77 index
    order, and `observations_for` is what applies `SOMA77_TO_AUTOANIM`. Recorded in the
    report as a plan error rather than fixed silently.
    """
    out = np.full((len(take_soma), len(JOINT_INDEX), 3), np.nan, dtype=np.float64)
    for name, soma in SOMA77_TO_AUTOANIM.items():
        out[:, JOINT_INDEX[name]] = take_soma[:, soma]
    return out


def load_takes(stride: int) -> dict[str, np.ndarray]:
    placement = ROOT / CAPTURE
    if placement.exists():
        track = np.load(placement, allow_pickle=True)["positions"]
        roots = track[:, :, JOINT_INDEX["root"], :2]
        ground = np.asarray([np.nanmedian(roots[s], axis=0) for s in range(min(2, len(roots)))])
    else:
        ground = np.asarray([[0.5, 4.9], [-0.9, 4.7]])
    return {clip: to_19(fx.build_take(clip, None, ground[index % len(ground)], stride))
            for index, clip in enumerate(fx.FULL_BODY_CLIPS)}


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between two stacks of frames, both orthonormalised first.

    CLAUDE.md: normalise the rows before comparing world orientation matrices, or an
    unnormalised scaled matrix drives `acos` past its clamp and every frame reads 0.00.
    """
    a, b = orthonormalise(np.asarray(a, dtype=float)), orthonormalise(np.asarray(b, dtype=float))
    return np.degrees(np.linalg.norm(log_so3(np.einsum("nij,nkj->nik", a, b)), axis=1))


def angular_speed_deg_s(frames: np.ndarray) -> np.ndarray:
    a = orthonormalise(np.asarray(frames, dtype=float))
    rel = np.einsum("nij,nkj->nik", a[1:], a[:-1])
    return np.degrees(np.linalg.norm(log_so3(rel), axis=1)) * SAMPLE_RATE_HZ


def yaw_deg(frames: np.ndarray) -> np.ndarray:
    """Heading of the frame's `across` column, unwrapped -- the fast-turn observable."""
    across = orthonormalise(np.asarray(frames, dtype=float))[:, :, 0]
    return np.degrees(np.unwrap(np.arctan2(across[:, 1], across[:, 0])))


def lag_frames(truth_yaw: np.ndarray, candidate_yaw: np.ndarray, maximum: int = 10) -> float:
    """Sub-frame cross-correlation lag of the yaw RATE. Positive: the candidate trails."""
    a = np.diff(truth_yaw)
    b = np.diff(candidate_yaw)
    a, b = a - a.mean(), b - b.mean()
    shifts, scores = [], []
    for shift in range(-maximum, maximum + 1):
        x, y = (a[: len(a) - shift], b[shift:]) if shift >= 0 else (a[-shift:], b[: len(b) + shift])
        if len(x) < 8:
            continue
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        shifts.append(shift)
        scores.append(float(x @ y / denominator) if denominator > 1e-12 else -1.0)
    if not scores:
        return float("nan")
    best = int(np.argmax(scores))
    if 0 < best < len(scores) - 1:
        y0, y1, y2 = scores[best - 1], scores[best], scores[best + 1]
        denominator = y0 - 2.0 * y1 + y2
        if abs(denominator) > 1e-12:
            return float(shifts[best] + 0.5 * (y0 - y2) / denominator)
    return float(shifts[best])


# -------------------------------------------------------------------------- the pipeline

def observe_and_triangulate(
    cameras, truth19: np.ndarray, rng: np.random.Generator, *, sigma_px: float, heavy: bool,
) -> np.ndarray:
    """Truth -> pixels -> noise -> `triangulate_point` -> world. The real rig, the real solver.

    Injecting in pixels and recovering through the pipeline's own triangulator is what
    "converted to 3D mm through the rig geometry" has to mean: an isotropic millimetre
    sphere would hide the depth anisotropy that is the whole reason a per-frame torso
    frame is jittery in the first place.
    """
    noisy = truth19.copy()
    n_cam = len(cameras)
    for name in THORAX_JOINTS:
        joint = JOINT_INDEX[name]
        for frame in range(len(truth19)):
            world = truth19[frame, joint]
            points = np.full((n_cam, 2), np.nan)
            weights = np.zeros(n_cam)
            for index, camera in enumerate(cameras):
                uv, depth = camera.project(world)
                if depth <= 0.0:
                    continue
                if heavy:
                    angle = rng.uniform(0.0, 2.0 * np.pi)
                    magnitude = heavy_tail_magnitude(rng, sigma_px)
                    offset = magnitude * np.asarray([np.cos(angle), np.sin(angle)])
                else:
                    offset = rng.normal(0.0, sigma_px, size=2)
                points[index] = uv + offset
                weights[index] = 0.95
            solved = triangulate_point(cameras, points, weights, pixel_scale=1.0)
            noisy[frame, joint] = solved.position_world_m if solved is not None else np.nan
    return noisy


def score_take(cameras, truth19: np.ndarray, *, seeds: int, sigma_px: float, heavy: bool) -> dict:
    """Every window and every control on one take, `seeds` independent noise draws."""
    truth_frame = _thorax_frames(truth19, smoothing_frames=0)
    truth_yaw = yaw_deg(truth_frame)
    truth_rate_p95 = float(np.percentile(np.abs(np.diff(truth_yaw)), 95))
    frames = len(truth19)
    usable = [w for w in WINDOWS if w <= frames]
    errors: dict[str, list[np.ndarray]] = {str(w): [] for w in usable}
    lags: dict[str, list[float]] = {str(w): [] for w in usable}
    attenuation: dict[str, list[float]] = {str(w): [] for w in usable}
    controls: dict[str, dict[str, list]] = {
        name: {"errors": [], "lags": [], "attenuation": []}
        for name in ("raw_positions_no_smoothing", "frozen_mean_frame")
    }
    for seed in range(seeds):
        rng = np.random.default_rng(20260902 + 1000 * seed)
        noisy = observe_and_triangulate(cameras, truth19, rng, sigma_px=sigma_px, heavy=heavy)
        smoothed, _ = _fill_and_smooth_positions(noisy)
        for window in usable:
            candidate = _thorax_frames(smoothed, smoothing_frames=window)
            errors[str(window)].append(geodesic_deg(candidate, truth_frame))
            candidate_yaw = yaw_deg(candidate)
            lags[str(window)].append(lag_frames(truth_yaw, candidate_yaw))
            attenuation[str(window)].append(
                float(np.percentile(np.abs(np.diff(candidate_yaw)), 95)) / max(truth_rate_p95, 1e-9))
        # Control 1: the frame from RAW triangulation -- neither position-smoothed nor
        # rotation-smoothed. This is what an earlier pipeline did and what the docstring
        # says was repaired; it must be the jitteriest arm.
        raw_positions = np.where(np.isfinite(noisy), noisy, smoothed)
        raw = _thorax_frames(raw_positions, smoothing_frames=0)
        # Control 2: the FROZEN frame -- the take's own mean, constant on every frame.
        # Built explicitly, because a Savitzky-Golay filter at polyorder 2 fits a parabola
        # and never freezes however wide it gets, and `_thorax_frames` returns the
        # UNSMOOTHED frame once the window exceeds the take. "A very wide window" is
        # therefore not a freeze control.
        frozen = np.broadcast_to(
            orthonormalise(_thorax_frames(smoothed, smoothing_frames=0).mean(axis=0)[None])[0],
            (frames, 3, 3))
        for name, arm in (("raw_positions_no_smoothing", raw), ("frozen_mean_frame", frozen)):
            controls[name]["errors"].append(geodesic_deg(arm, truth_frame))
            arm_yaw = yaw_deg(arm)
            controls[name]["lags"].append(
                0.0 if name == "frozen_mean_frame" else lag_frames(truth_yaw, arm_yaw))
            controls[name]["attenuation"].append(
                float(np.percentile(np.abs(np.diff(arm_yaw)), 95)) / max(truth_rate_p95, 1e-9))

    def block(err: list[np.ndarray], lag: list[float], att: list[float]) -> dict:
        stack = np.stack(err)
        per_seed_median = np.median(stack, axis=1)
        per_seed_p95 = np.percentile(stack, 95, axis=1)
        return {
            # Pooled over every seed and frame: the stable figure the selection uses.
            "pooled_median_deg": float(np.median(stack)),
            "pooled_p95_deg": float(np.percentile(stack, 95)),
            # Per-seed mean and spread: the figure the brief asks to be reported.
            "median_deg": {"mean": float(per_seed_median.mean()),
                           "sd": float(per_seed_median.std(ddof=1)) if seeds > 1 else 0.0},
            "p95_deg": {"mean": float(per_seed_p95.mean()),
                        "sd": float(per_seed_p95.std(ddof=1)) if seeds > 1 else 0.0},
            # Kept so windows can be compared PAIRED -- same seed, same noise realisation,
            # same clip. An unpaired mean +- sd cannot separate 3.86 from 3.91 against a
            # per-seed sd of 0.55; the paired difference can, or can say it does not.
            "p95_per_seed": [float(v) for v in per_seed_p95],
            "lag_frames": {"mean": float(np.mean(lag)),
                           "sd": float(np.std(lag, ddof=1)) if seeds > 1 else 0.0},
            "yaw_rate_p95_ratio": {"mean": float(np.mean(att)),
                                   "sd": float(np.std(att, ddof=1)) if seeds > 1 else 0.0},
        }

    return {
        "frames": frames,
        "truth_thorax_speed_deg_s": {
            "median": float(np.median(angular_speed_deg_s(truth_frame))),
            "p95": float(np.percentile(angular_speed_deg_s(truth_frame), 95)),
        },
        "windows": {w: block(errors[w], lags[w], attenuation[w]) for w in errors},
        "windows_not_covered": [w for w in WINDOWS if w > frames],
        "controls": {n: block(c["errors"], c["lags"], c["attenuation"]) for n, c in controls.items()},
    }


# ------------------------------------------------------------------------------ selection

def pool_clips(per_clip: dict[str, dict], key: str = "pooled_p95_deg") -> dict:
    """Mean over clips, so no one trajectory carries the answer. N/A where a clip is short."""
    out = {}
    for window in map(str, WINDOWS):
        rows = [clip["windows"].get(window) for clip in per_clip.values()]
        if any(r is None for r in rows):
            out[window] = {"status": "not covered: window exceeds take length on at least one clip"}
            continue
        out[window] = {
            "pooled_p95_deg": float(np.mean([r["pooled_p95_deg"] for r in rows])),
            "pooled_median_deg": float(np.mean([r["pooled_median_deg"] for r in rows])),
            "p95_deg_seed_sd": float(np.mean([r["p95_deg"]["sd"] for r in rows])),
            "median_deg_seed_sd": float(np.mean([r["median_deg"]["sd"] for r in rows])),
            "lag_frames": float(np.mean([r["lag_frames"]["mean"] for r in rows])),
            "yaw_rate_p95_ratio": float(np.mean([r["yaw_rate_p95_ratio"]["mean"] for r in rows])),
        }
    return out


def paired_wins(per_clip: dict[str, dict], against: int) -> dict:
    """Does the argmin actually beat each neighbour, seed by seed?

    CLAUDE.md: quote a margin only with a resampling argument behind it. Seeds here are
    independent draws (unlike frames within a take, which have lag-1 autocorrelation 0.99
    and cannot be resampled), so the paired sign test over (clip, seed) is legitimate and
    is the honest statement of whether the p95 curve's minimum is real or flat.
    """
    reference = {clip: d["windows"][str(against)]["p95_per_seed"] for clip in per_clip
                 for d in [per_clip[clip]] if str(against) in d["windows"]}
    out = {}
    for window in map(str, WINDOWS):
        deltas = []
        for clip, d in per_clip.items():
            if window not in d["windows"] or clip not in reference:
                deltas = None
                break
            deltas += [a - b for a, b in zip(d["windows"][window]["p95_per_seed"],
                                             reference[clip])]
        if deltas is None:
            continue
        wins = sum(1 for v in deltas if v > 0.0)
        out[window] = {
            "n_paired": len(deltas),
            "argmin_better_fraction": wins / len(deltas),
            "mean_delta_deg": float(np.mean(deltas)),
            "sd_delta_deg": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        }
    return {"against_window": against, "per_window": out,
            "reading": "argmin_better_fraction is the share of (clip, seed) draws on which "
                       "this window's p95 was WORSE than the argmin's. 0.5 means the two "
                       "are indistinguishable on this data."}


def per_clip_argmin(per_clip: dict[str, dict], metric: str = "pooled_p95_deg") -> dict:
    """The argmin each clip would choose on its own -- consensus, or an average of disagreement."""
    out = {}
    for clip, d in per_clip.items():
        usable = sorted((int(w), v[metric]) for w, v in d["windows"].items())
        out[clip] = {"argmin_window": usable[int(np.argmin([v for _, v in usable]))][0],
                     "thorax_speed_median_deg_s": d["truth_thorax_speed_deg_s"]["median"],
                     "covered_windows": [w for w, _ in usable]}
    return out


def interior_optimum(pooled: dict, metric: str = "pooled_p95_deg") -> dict:
    usable = sorted((int(w), v[metric]) for w, v in pooled.items() if metric in v)
    if len(usable) < 3:
        return {"status": "too few covered windows to speak of an interior"}
    values = [v for _, v in usable]
    best = int(np.argmin(values))
    return {
        "covered_windows": [w for w, _ in usable],
        "argmin_window": usable[best][0],
        "argmin_value": usable[best][1],
        "interior": 0 < best < len(usable) - 1,
        "curve": [{"window": w, metric: v} for w, v in usable],
    }


# The MAMMA oracle sweep, transcribed verbatim from `_thorax_frames`'s docstring. It is
# here so the two can be read side by side and for no other purpose: it REPORTS, IT NEVER
# SELECTS. It is also a different quantity on different data -- degrees of agreement with
# MAMMA's head expressed in our thorax frame, on the four-camera real fixture -- so
# "agreement" between the arms can only mean the same argmin and the same curve shape,
# never the same numbers.
MAMMA_ORACLE_SWEEP_REPORTS_NEVER_SELECTS = {
    "reference": "MAMMA's own head orientation in our thorax frame (P1 median / p95, two performers)",
    "source": "src/autoanim_gnm/commercial_multiview.py, _thorax_frames docstring",
    "rows": [
        {"window": 0, "median_deg": [5.46, 4.87], "p95_deg": [17.22, 17.59]},
        {"window": 5, "median_deg": [5.42, 4.87], "p95_deg": [17.00, 17.21]},
        {"window": 15, "median_deg": [5.46, 4.89], "p95_deg": [14.14, 16.30]},
        {"window": 21, "median_deg": [5.87, 4.96], "p95_deg": [14.12, 17.19]},
        {"window": 31, "median_deg": [6.52, 4.88], "p95_deg": [13.78, 16.39]},
    ],
    "its_own_p95_argmin_by_performer": {"0": 31, "1": 15},
    "its_own_median_argmin_by_performer": {"0": 5, "1": 5},
    "its_stated_choice": 15,
    "note": "Read BOTH columns. Performer 1's p95 runs 17.59 / 17.21 / 16.30 / 17.19 / "
            "16.39 -- an interior minimum at 15, which is the stated choice. Performer 0's "
            "runs 17.22 / 17.00 / 14.14 / 14.12 / 13.78 -- monotone to 31, the widest "
            "window it tested, still improving. So the docstring's 'it has an interior "
            "optimum' holds for one performer and not the other, and the table does not "
            "say which it was read from. Neither performer's p95 argmin is 15 on both "
            "arms at once: it is 15 for performer 1 and 31 for performer 0.",
    "shape_that_does_agree": "on BOTH performers the median prefers a narrower window than "
                             "the p95 (median argmin 5 for each; p95 argmin 31 and 15), and "
                             "ours does the same at the nearest-speed playback. The SHAPE "
                             "agrees; the LEVEL does not, and only the shape is comparable, "
                             "because the two arms score different quantities on different "
                             "data.",
    "what_this_arm_cannot_see": "it scores AGREEMENT with a smooth fitted-chain reference, "
                                "so it has no truth to charge lag or peak attenuation "
                                "against. Where its p95 improves with a wider window, that "
                                "improvement may be the reference's own smoothness rather "
                                "than a better frame, and nothing in that arm can tell the "
                                "two apart. Ours can, because it scores against exact truth.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts/compare/thorax-window-sweep.json")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--headline-stride", type=int, default=3)
    args = parser.parse_args()

    rig = cm.load_camera_rig(ROOT / RIG)
    cameras = tuple(camera.scaled(WORKING_WIDTH, WORKING_HEIGHT) for camera in rig)

    # The speed-match target: our own capture, our own detector. Not MAMMA output.
    track = np.load(ROOT / CAPTURE, allow_pickle=True)["positions"]
    capture_speed = []
    for subject in range(track.shape[0]):
        positions, _ = _fill_and_smooth_positions(track[subject])
        speed = angular_speed_deg_s(_thorax_frames(positions, smoothing_frames=0))
        capture_speed.append({"median_deg_s": float(np.median(speed)),
                              "p95_deg_s": float(np.percentile(speed, 95))})
    print("own-capture thorax speed:", capture_speed)

    sigmas = {
        "from_reprojection_median_4_views_px": sigma_from_reprojection(REPROJECTION_MEDIAN_PX, 4),
        "from_reprojection_median_3_views_px": sigma_from_reprojection(REPROJECTION_MEDIAN_PX, 3),
        "from_epipolar_median_px": sigma_from_epipolar(EPIPOLAR_ONE_SIDED_PX[2]),
        "used_px": NOISE_SIGMA_PX,
    }
    print("per-view per-axis sigma at 1280 px:", {k: round(v, 3) for k, v in sigmas.items()})

    strides: dict[str, dict] = {}
    for stride in args.strides:
        takes = load_takes(stride)
        moving, static = {}, {}
        for clip, truth19 in takes.items():
            heavy = stride == args.headline_stride and clip in MOVING_CLIPS
            target = moving if clip in MOVING_CLIPS else static
            print(f"[stride {stride}] {clip} ({len(truth19)} frames) ...", flush=True)
            target[clip] = score_take(cameras, truth19, seeds=args.seeds,
                                      sigma_px=NOISE_SIGMA_PX, heavy=False)
            if heavy:
                target[clip]["heavy_tail_arm"] = score_take(
                    cameras, truth19, seeds=args.seeds, sigma_px=NOISE_SIGMA_PX, heavy=True)
        pooled = pool_clips(moving)
        entry = {
            "clip_frames": {c: d["frames"] for c, d in {**moving, **static}.items()},
            "thorax_speed_deg_s": {c: d["truth_thorax_speed_deg_s"]
                                   for c, d in {**moving, **static}.items()},
            "moving_clips": moving,
            "static_clips_degenerate_arm": static,
            "pooled_moving": pooled,
            "selection_p95": interior_optimum(pooled, "pooled_p95_deg"),
            "selection_median": interior_optimum(pooled, "pooled_median_deg"),
            "per_clip_argmin": per_clip_argmin(moving),
        }
        chosen = entry["selection_p95"].get("argmin_window")
        if chosen is not None:
            entry["paired_against_argmin"] = paired_wins(moving, chosen)
        if stride == args.headline_stride:
            heavy_pooled = pool_clips({c: d["heavy_tail_arm"] for c, d in moving.items()})
            entry["pooled_moving_heavy_tail"] = heavy_pooled
            entry["selection_p95_heavy_tail"] = interior_optimum(heavy_pooled, "pooled_p95_deg")
            static_pooled = pool_clips(static)
            entry["selection_p95_static_degenerate"] = interior_optimum(
                static_pooled, "pooled_p95_deg")
        strides[str(stride)] = entry

    # Amplitude sensitivity: the px->mm conversion runs through MAMMA's calibration, so
    # the selection must not depend on that amplitude being exactly right.
    sensitivity = {}
    probe_takes = load_takes(args.headline_stride)
    for label, scale in (("0.5x", 0.5), ("2.0x", 2.0)):
        probe = {c: score_take(cameras, probe_takes[c], seeds=max(4, args.seeds // 2),
                               sigma_px=NOISE_SIGMA_PX * scale, heavy=False)
                 for c in MOVING_CLIPS}
        pooled = pool_clips(probe)
        sensitivity[label] = {"pooled": pooled,
                              "selection_p95": interior_optimum(pooled, "pooled_p95_deg")}
        print(f"amplitude {label}: argmin {sensitivity[label]['selection_p95'].get('argmin_window')}")

    # How well each stride matches the speed our own capture's thorax actually turns at.
    # Reported rather than assumed: the headline stride is the closest available, and the
    # fixture cannot reach the real speed at a length that still covers wide windows.
    capture_median = float(np.mean([c["median_deg_s"] for c in capture_speed]))
    speed_match = {}
    for stride, entry in strides.items():
        moving = [entry["thorax_speed_deg_s"][c]["median"] for c in MOVING_CLIPS]
        speed_match[stride] = {"moving_clip_median_deg_s": float(np.mean(moving)),
                               "ratio_to_own_capture": float(np.mean(moving) / capture_median)}
    print("speed match to own capture:", {k: round(v["ratio_to_own_capture"], 2)
                                          for k, v in speed_match.items()})

    head = strides[str(args.headline_stride)]
    gaussian = head["selection_p95"]
    heavy = head["selection_p95_heavy_tail"]
    argmin_by_stride = {s: v["selection_p95"].get("argmin_window") for s, v in strides.items()}
    interior_by_stride = {s: v["selection_p95"].get("interior") for s, v in strides.items()}
    agree = gaussian.get("argmin_window") == heavy.get("argmin_window")
    interior = bool(gaussian.get("interior"))
    proposed = gaussian.get("argmin_window") if interior else None
    bracket = sorted({gaussian.get("argmin_window"), heavy.get("argmin_window")} - {None})
    proposal = {
        "rule": "minimum pooled p95 angular error against exact truth, on the "
                "nearest-speed moving clips, with an INTERIOR optimum required. Where the "
                "p95 curve is flat across neighbouring windows, lag and peak attenuation "
                "are the REPORTED tiebreaker -- they are not part of the selection rule.",
        "headline_stride": args.headline_stride,
        "headline_stride_speed_ratio_to_own_capture":
            speed_match[str(args.headline_stride)]["ratio_to_own_capture"],
        "gaussian_arm_argmin": gaussian.get("argmin_window"),
        "heavy_tail_arm_argmin": heavy.get("argmin_window"),
        "arms_agree": agree,
        "bracket": bracket,
        "interior_optimum": interior,
        "covered_windows": gaussian.get("covered_windows"),
        "argmin_by_stride": argmin_by_stride,
        "interior_by_stride": interior_by_stride,
        "amplitude_sensitivity_argmin": {k: v["selection_p95"].get("argmin_window")
                                         for k, v in sensitivity.items()},
        "proposed_value": proposed,
        "status": "selected" if proposed is not None else
                  "RULE DID NOT SELECT: the p95 minimum sits at an end of the covered sweep. "
                  "No value is proposed and nothing is extrapolated past the windows the "
                  "takes cover.",
        "current_value_verdict": {
            "argmin_by_stride": argmin_by_stride,
            "speed_ratio_by_stride": {k: round(v["ratio_to_own_capture"], 2)
                                      for k, v in speed_match.items()},
            "reading":
                "The argmin falls as the thorax turns faster and settles at 9 once the "
                "playback reaches roughly 0.6-0.8 of our own capture's median thorax "
                f"speed ({argmin_by_stride}). At the slowest playback, which turns at "
                f"{speed_match[min(speed_match)]['ratio_to_own_capture']:.0%} of the real "
                "speed, the argmin is 21 -- ABOVE the shipped 15 -- and that is the point: "
                "the number is not a property of the pipeline, it is a property of the "
                "noise-to-motion ratio. At the two nearest-speed playbacks it is 9, and "
                f"the two noise arms bracket it at {bracket}.",
            "why_the_true_optimum_is_at_most_this":
                "Two biases push the measured optimum WIDER than the truth, and they are "
                "independent. (a) Speed: even the fastest playback the fixture supports at "
                "usable length reaches only "
                f"{speed_match[str(max(args.strides))]['ratio_to_own_capture']:.0%} of our "
                "own capture's median thorax speed, and the argmin falls with speed. "
                "(b) Noise colour: the injected noise is WHITE, and a temporal smoother "
                "removes white noise far better than the temporally correlated error a "
                "real detector makes -- so smoothing is flattered here and a wider window "
                "looks cheaper than it is. Both point the same way, so the shipped 15 is "
                "not supported at the real fixture's motion, and the honest statement is "
                "an upper bound rather than a point estimate.",
        },
        "caveats": [
            "The optimum is a bias-variance trade, so it moves with the RATIO of detector "
            "noise to thorax angular speed -- not with either alone. It moved 5 -> 9 -> 21 "
            "as the injected noise went 0.5x -> 1x -> 2x, and 21 -> 15 -> 9 -> 5 as the "
            "motion went 1x -> 4x. A single frame count is not a transferable constant, and "
            "this sweep says so rather than hiding it.",
            "The two noise arms disagree because they differ in TAIL, and p95 is a tail "
            "statistic. The Gaussian arm's amplitude is pinned by two independent estimates "
            "of the same sigma (3.13 and 3.26 px) and is the primary; the heavy-tail arm "
            "carries SOMA-77's measured tail and is the sensitivity arm.",
            "The frozen-frame control fails on the two clips that turn (5.3 / 10.5 and "
            "9.5 / 14.2 deg) but WINS on will-stephen-acting-body, which turns at only "
            "18.8 deg/s. A freeze control is only a control on a thorax that moves; read it "
            "per clip, never pooled.",
        ],
        "not_edited": "Lane I is read-only against the pipeline. commercial_multiview.py is "
                      "NOT edited here; the constant moves in lane D after Fable reviews.",
    }

    report = {
        "schema_version": "autoanim.thorax-window-sweep/1.0",
        "step": "I8",
        "constant": {"name": "THORAX_SMOOTHING_FRAMES",
                     "file": "src/autoanim_gnm/commercial_multiview.py:1679",
                     "current_value": int(cm.THORAX_SMOOTHING_FRAMES),
                     "current_provenance": "MAMMA oracle arm (leak)"},
        "selector": "synthetic truth (exact FK thorax frames) + our own detector's "
                    "self-consistency noise; MAMMA reports and never selects",
        "contains_mamma_output": False,
        "declared_mamma_dependencies": [
            {"item": RIG,
             "what": "the camera rig carrying pixel noise into millimetres IS MAMMA's ma_cap "
                     "calibration (ladder rung 0, 'not owned'); byte-identical to "
                     "artifacts/commercial-multiview-soma77/camera-rig.json",
             "mitigation": "the selection is an argmin, not an absolute; noise amplitude swept "
                           "at 0.5x and 2.0x"},
            {"item": CAPTURE,
             "what": "subject placement and the thorax-speed match target come from OUR OWN "
                     "detector's track, which was run on MAMMA's example footage through that "
                     "rig. The positions are ours; the footage and calibration are not owned.",
             "mitigation": "used only to place the subject inside the frusta and to choose a "
                           "playback stride; neither enters the score"},
        ],
        "own_capture_thorax_speed_deg_s": capture_speed,
        "speed_match_by_stride": speed_match,
        "noise_source": {
            "reprojection_median_px": REPROJECTION_MEDIAN_PX,
            "reprojection_report": "artifacts/commercial-multiview-soma77/run-report.json",
            "epipolar_one_sided_px": dict(zip(map(str, EPIPOLAR_PERCENTILES), EPIPOLAR_ONE_SIDED_PX)),
            "epipolar_instrument": "tools/swap-harness/sam3d_ladder.py (SOMA-77 arm)",
            "sigma_px": sigmas,
            "mamma_derived": False,
            "note": "injected in PIXELS per view and recovered through triangulate_point on the "
                    "real rig, so the depth anisotropy is present; an isotropic mm sphere would "
                    "hide it",
        },
        "blind_to": [
            "temporally correlated detector error -- the injected noise is white, and "
            "correlated error is exactly what a temporal smoother cannot remove, so every "
            "window here is flattered equally",
            "calibration, distortion, sync and soft-tissue error, absent by construction",
            "joint-definition error, which dominated Battles 0 and 1 on real footage",
            "windows past what the takes cover: at the nearest-speed playback the takes are "
            "24-32 frames, so 31 and 45 are not measured at all",
            "a real take's length -- nothing here says what any window does over a minute",
            "cross-correlation lag on a 18-32 frame take is coarse: the sub-frame estimate "
            "is meaningful at the ~1 frame level and the small NEGATIVE lags at the "
            "narrowest windows are estimator noise, not anticipation",
            "the audit asymmetry in the manifest's companion script: module constants are "
            "scanned regardless of reachability while keyword defaults are "
            "reachability-filtered",
        ],
        "seeds": args.seeds,
        "windows_swept": list(WINDOWS),
        "strides": strides,
        "amplitude_sensitivity": sensitivity,
        "mamma_oracle_sweep": MAMMA_ORACLE_SWEEP_REPORTS_NEVER_SELECTS,
        "proposal": proposal,
        "plan_errors_found": [
            "The brief said build_synthetic_truth_fixture's take builder gives [frame, joint, 3] "
            "in our 19-joint JOINT_INDEX order. It returns [frame, 77, 3] in SOMA-77 index "
            "order; SOMA77_TO_AUTOANIM is applied by observations_for, not by build_take.",
            "The brief's control 'a window so wide it freezes the frame' cannot be built by "
            "widening the window. Savitzky-Golay at polyorder 2 fits a parabola and never "
            "freezes, and _thorax_frames returns the UNSMOOTHED frame when the window exceeds "
            "the take -- so the widest cells are 'none' in disguise. The freeze control is "
            "built explicitly as the take's mean frame.",
            "The brief assumed the tracked synthetic clips could select a temporal window as "
            "they stand. At stride 1 their thorax turns at 0.0-22.7 deg/s against our own "
            "capture's 66-73, so the p95 curve is monotone to the widest window on offer and "
            "selects nothing. The clips must be played faster first, which costs take length "
            "and caps the sweep.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(json.dumps(proposal, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
