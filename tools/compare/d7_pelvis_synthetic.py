#!/usr/bin/env python3
"""D7, step 2: which pelvis frame, and which temporal window -- SELECTED ON SYNTHETIC TRUTH.

INSTRUMENT AND SELECTOR. This is the only arm that selects anything in D7. The MAMMA arm
reports and never selects; the real take's rigidity pass (`d7_pelvis_rigidity.py`) says
whether the landmark may be trusted at all and selects nothing either.

THE TRUTH. `scripts/build_synthetic_truth_fixture.py` poses SOMASKEL77 with real GEM-X
rotations, so a TRUE pelvis frame exists on every frame: the rotation `R` with
`posed[child] - posed[Hips] == R . (rest[child] - rest[Hips])` for the three children of
`Hips` -- `Spine1`, `LeftLeg`, `RightLeg`. It is recovered by Kabsch and it is EXACT
(4.2e-8 m residual), which is why:

  * the CLEAN arm cannot select. Every rigid pelvis candidate reads ~0 on it by
    construction. What the clean arm proves is the CONSTRUCTION, and it separates the
    pelvis candidates from the controls and from the LUMBAR arms, which is its job.
  * the NOISY arm is the selector. Lever length is the whole question there:
    `root->Spine1` is 43 mm and `mid(hips)->Spine1` is 108 mm against the same position
    noise.

THE NOISE is I7/I8's own: `tools/head/thorax_window_sweep.py`'s measured heavy-tail
frame-correlated model at `NOISE_SIGMA_PX = 3.20 px`, itself two independent estimates of
OUR OWN detector (3.13 and 3.26 px). Never MAMMA's residual. It is injected in PIXELS and
recovered through the real `triangulate_point`, because an isotropic millimetre sphere
would hide the depth anisotropy that makes a per-frame frame jittery in the first place.

SAME DENOMINATOR. Every arm reads the same noised positions on the same frames; the
joints noised are every joint any arm reads (SOMA-77 0, 1, 2, 5, 67, 72), so no arm
enjoys a cleaner input than another.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7_pelvis_synthetic.py

Writes `artifacts/compare/d7-pelvis-frame/synthetic.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "scripts", "workers/commercial_multiview"):
    sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

import build_synthetic_truth_fixture as fx  # noqa: E402
import thorax_window_sweep as sweep  # noqa: E402
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_INDEX, load_camera_rig, triangulate_point,
)
from autoanim_gnm.head_orientation import log_so3, orthonormalise, rodrigues  # noqa: E402
from autoanim_gnm.soma_motion import SOMASKEL77_NAMES  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d7-pelvis-frame"
RIG = ROOT / "artifacts/commercial-multiview-soma77/camera-rig.json"
WORKING_WIDTH, WORKING_HEIGHT = 1280, 720
SAMPLE_RATE_HZ = 30
N = {name: index for index, name in enumerate(SOMASKEL77_NAMES)}
S2W = fx.SOMA_TO_WORLD

SEEDS = tuple(20260904 + 1000 * k for k in range(5))
WINDOWS = (0, 3, 5, 9, 15, 21, 31)
BENT_TILT_DEG = 20.0
SQUAT = "autoanim_squat/research-squat-640"
FAST_STRIDE = 3

# SOMA-77 joints every arm reads. Noised together so no arm has a cleaner input.
NOISED = (N["Hips"], N["Spine1"], N["Spine2"], N["Neck2"], N["LeftLeg"], N["RightLeg"])
# Which of them the pipeline would smooth: the ones inside the 19-joint contract. `Spine1`
# and `Spine2` come through the toe-landmark feed, which is triangulated per frame and
# never Savitzky-Golay filtered, so the asymmetry is modelled rather than assumed away.
SMOOTHED = (N["Hips"], N["Neck2"], N["LeftLeg"], N["RightLeg"])
# Whether the spine point gets the SAME Savitzky-Golay treatment the 19-contract joints
# get. It is a choice, not a fact: the toe feed (`_toe_world_for_subject`, the precedent
# this extends) triangulates per frame and never smooths, so the shipped code could go
# either way -- and leaving the spine raw while the controls' inputs are smoothed would
# break SAME DENOMINATOR by handing every control a cleaner input than the candidates.
# Both arms are measured and both are reported.
SPINE_SMOOTHING_ARMS = (True, False)

SHIPPED = json.loads((OUT_DIR / "rest-pelvis-constants.json").read_text())


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def frame_of(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """`commercial_multiview._frame`, vectorised over frames. Columns (primary, ., .)."""
    first = unit(np.asarray(primary, dtype=np.float64))
    second = np.asarray(secondary, dtype=np.float64) - first * np.einsum(
        "...i,...i->...", first, secondary)[..., None]
    second = unit(second)
    third = unit(np.cross(first, second))
    second = unit(np.cross(third, first))
    return np.stack((first, second, third), axis=-1)


def alignment(source_primary, source_secondary, target_primary, target_secondary):
    """`_frame_alignment` as a rotation matrix: target_frame @ source_frame.T."""
    source = frame_of(np.broadcast_to(np.asarray(source_primary, float), np.shape(target_primary)),
                      np.broadcast_to(np.asarray(source_secondary, float), np.shape(target_primary)))
    target = frame_of(target_primary, target_secondary)
    return np.einsum("...ij,...kj->...ik", target, source)


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = orthonormalise(np.asarray(a, float)), orthonormalise(np.asarray(b, float))
    return np.degrees(np.linalg.norm(log_so3(np.einsum("nij,nkj->nik", a, b)), axis=1))


def truth_frames(posed: np.ndarray, rest_world: np.ndarray) -> np.ndarray:
    """Exact `Hips` world rotation per frame, by Kabsch on the three rigid children."""
    kids = [N["Spine1"], N["LeftLeg"], N["RightLeg"]]
    a = rest_world[kids] - rest_world[N["Hips"]]
    out = np.zeros((len(posed), 3, 3))
    residual = 0.0
    for f in range(len(posed)):
        b = posed[f, kids] - posed[f, N["Hips"]]
        u, _, vt = np.linalg.svd(a.T @ b)
        rot = (vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T)
        out[f] = rot
        residual = max(residual, float(np.abs(rot @ a.T - b.T).max()))
    return out, residual


def source_axes(rest_world: np.ndarray, which: str) -> tuple[np.ndarray, np.ndarray]:
    """(primary, secondary) source vectors in world Z-up for one candidate."""
    hips, s1 = rest_world[N["Hips"]], rest_world[N["Spine1"]]
    left, right = rest_world[N["LeftLeg"]], rest_world[N["RightLeg"]]
    mid = 0.5 * (left + right)
    across = left - right
    if which == "A_root_to_spine1":
        return unit(s1 - hips), unit(across)
    if which == "B_hipmid_to_spine1":
        return unit(s1 - mid), unit(across)
    if which == "lumbar_root_to_spine2":
        return unit(rest_world[N["Spine2"]] - hips), unit(across)
    raise ValueError(which)


# INSTRUMENT-SIDE ONLY, and never shipped: the lumbar arms are reported, never selected,
# so their source direction is derived here from the same five clip rests rather than
# registered as a runtime constant.
def _lumbar_rest_source() -> np.ndarray:
    rows = []
    for clip in fx.FULL_BODY_CLIPS:
        rest = np.load(fx.MOTION_ROOT / clip / "soma_motion.npz",
                       allow_pickle=True)["rest_joint_positions_m"].astype(np.float64)
        rows.append(unit(rest[N["Spine2"]] - rest[N["Hips"]]))
    return S2W @ unit(np.median(np.stack(rows), axis=0))


def shipped_source(which: str) -> tuple[np.ndarray, np.ndarray]:
    """The SHIPPED constants, carried from SOMA rest axes (Y-up) into world Z-up."""
    across = S2W @ np.asarray(SHIPPED["hip_across_rest"], float)
    if which == "A_root_to_spine1":
        return S2W @ np.asarray(SHIPPED["root_to_spine1"], float), across
    if which == "B_hipmid_to_spine1":
        return S2W @ np.asarray(SHIPPED["hipmid_to_spine1"], float), across
    if which == "lumbar_root_to_spine2":
        return _lumbar_rest_source(), across
    raise ValueError(which)


def kabsch_template(rest_world: np.ndarray | None) -> np.ndarray:
    if rest_world is not None:
        kids = [N["Spine1"], N["LeftLeg"], N["RightLeg"]]
        return rest_world[kids] - rest_world[N["Hips"]]
    t = SHIPPED["kabsch_template_m"]
    return (S2W @ np.stack([np.asarray(t["root_to_spine1_m"], float),
                            np.asarray(t["root_to_leftleg"], float),
                            np.asarray(t["root_to_rightleg"], float)]).T).T


def candidate_frames(pos: np.ndarray, which: str, rest_world: np.ndarray | None) -> np.ndarray:
    """`pos` is [frame, 77, 3] world Z-up (only the read joints need be finite)."""
    hips, s1, s2 = pos[:, N["Hips"]], pos[:, N["Spine1"]], pos[:, N["Spine2"]]
    left, right = pos[:, N["LeftLeg"]], pos[:, N["RightLeg"]]
    mid = 0.5 * (left + right)
    across = left - right
    if which == "C_kabsch_pelvis":
        template = kabsch_template(rest_world)
        out = np.zeros((len(pos), 3, 3))
        for f in range(len(pos)):
            b = np.stack([s1[f] - hips[f], left[f] - hips[f], right[f] - hips[f]])
            u, _, vt = np.linalg.svd(template.T @ b)
            out[f] = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
        return out
    if which == "thorax_as_pelvis":
        return alignment((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), pos[:, N["Neck2"]] - mid, across)
    if which == "world_vertical":
        return alignment((0.0, 0.0, 1.0), (1.0, 0.0, 0.0),
                         np.broadcast_to(np.asarray((0.0, 0.0, 1.0)), hips.shape), across)
    if which == "no_rest_correction":
        return alignment((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), s1 - mid, across)
    if which == "lumbar_spine_line":
        pts = np.stack([hips, s1, s2], axis=1)
        centred = pts - pts.mean(axis=1, keepdims=True)
        out = np.zeros((len(pos), 3, 3))
        for f in range(len(pos)):
            _, _, vt = np.linalg.svd(centred[f])
            axis = vt[0] * np.sign(vt[0] @ (s2[f] - hips[f]))
            out[f] = frame_of(axis, across[f])
        # the source frame is the same line taken on the rest
        rest = rest_world if rest_world is not None else None
        if rest is None:
            src = frame_of(S2W @ np.asarray(SHIPPED["root_to_spine1"], float),
                           S2W @ np.asarray(SHIPPED["hip_across_rest"], float))
        else:
            pts_r = np.stack([rest[N["Hips"]], rest[N["Spine1"]], rest[N["Spine2"]]])
            _, _, vt = np.linalg.svd(pts_r - pts_r.mean(axis=0))
            axis = vt[0] * np.sign(vt[0] @ (rest[N["Spine2"]] - rest[N["Hips"]]))
            src = frame_of(axis, rest[N["LeftLeg"]] - rest[N["RightLeg"]])
        return np.einsum("nij,kj->nik", out, src)
    if which in ("A_root_to_spine1", "B_hipmid_to_spine1", "lumbar_root_to_spine2"):
        primary = {"A_root_to_spine1": s1 - hips,
                   "B_hipmid_to_spine1": s1 - mid,
                   "lumbar_root_to_spine2": s2 - hips}[which]
        if rest_world is not None:
            sp, ss = source_axes(rest_world, which)
        else:
            sp, ss = shipped_source(which)
        return alignment(sp, ss, primary, across)
    raise ValueError(which)


CANDIDATES = ("A_root_to_spine1", "B_hipmid_to_spine1", "C_kabsch_pelvis")
CONTROLS = ("thorax_as_pelvis", "world_vertical", "no_rest_correction")
LUMBAR = ("lumbar_root_to_spine2", "lumbar_spine_line")
ALL_ARMS = CANDIDATES + CONTROLS + LUMBAR


def smooth_frames(frames: np.ndarray, window: int) -> np.ndarray:
    """`_thorax_frames`' own rotation smoothing: tangent space about the take's mean."""
    if window <= 1 or len(frames) < window:
        return frames
    mean = orthonormalise(frames.mean(axis=0)[None])[0]
    tangent = log_so3(np.einsum("nij,kj->nik", frames, mean))
    length = window if window % 2 else window - 1
    tangent = savgol_filter(tangent, window_length=length, polyorder=2, axis=0, mode="interp")
    return orthonormalise(np.einsum("nij,jk->nik", rodrigues(tangent), mean))


def yaw_deg(frames: np.ndarray) -> np.ndarray:
    across = orthonormalise(np.asarray(frames, float))[:, :, 0]
    return np.degrees(np.unwrap(np.arctan2(across[:, 1], across[:, 0])))


def observe(cameras, truth: np.ndarray, rng, *, smooth_spine: bool,
            sigma_scale: float = 1.0) -> np.ndarray:
    """Truth -> pixels -> heavy-tail frame-correlated noise -> `triangulate_point` -> world."""
    noisy = truth.copy()
    held = {}
    for joint in NOISED:
        for frame in range(len(truth)):
            points = np.full((len(cameras), 2), np.nan)
            weights = np.zeros(len(cameras))
            for index, camera in enumerate(cameras):
                uv, depth = camera.project(truth[frame, joint])
                if depth <= 0.0:
                    continue
                key = (index, joint)
                if key not in held or held[key][0] <= 0:
                    angle = rng.uniform(0.0, 2.0 * np.pi)
                    magnitude = sweep.heavy_tail_magnitude(
                        rng, sweep.NOISE_SIGMA_PX * sigma_scale)
                    held[key] = [int(rng.integers(2, 6)),
                                 magnitude * np.asarray([np.cos(angle), np.sin(angle)])]
                held[key][0] -= 1
                points[index] = uv + held[key][1]
                weights[index] = 0.95
            solved = triangulate_point(cameras, points, weights, pixel_scale=1.0)
            noisy[frame, joint] = solved.position_world_m if solved is not None else np.nan
    # the pipeline's own treatment: the 19-contract joints are filled and Savitzky-Golay
    # smoothed; the spine points come through the toe-style feed and are only gap-filled.
    axis = np.arange(len(truth), dtype=np.float64)
    for joint in NOISED:
        valid = np.isfinite(noisy[:, joint]).all(axis=1)
        if valid.sum() < 2:
            continue
        for k in range(3):
            noisy[:, joint, k] = np.interp(axis, axis[valid], noisy[valid, joint, k])
        smooth_this = joint in SMOOTHED or (smooth_spine and joint in (N["Spine1"], N["Spine2"]))
        if smooth_this and len(truth) >= 7:
            noisy[:, joint] = savgol_filter(noisy[:, joint], window_length=9, polyorder=2,
                                            axis=0, mode="interp")
    return noisy


def tilt_deg(pos: np.ndarray) -> np.ndarray:
    mid = 0.5 * (pos[:, N["LeftLeg"]] + pos[:, N["RightLeg"]])
    trunk = unit(pos[:, N["Neck2"]] - mid)
    return np.degrees(np.arccos(np.clip(trunk @ np.asarray([0.0, 0.0, 1.0]), -1.0, 1.0)))


def load(stride: int = 1) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ground = np.asarray([[0.5, 4.9], [-0.9, 4.7]])
    out = {}
    for index, clip in enumerate(fx.FULL_BODY_CLIPS):
        take = fx.build_take(clip, None, ground[index % 2], stride)
        rest = np.load(fx.MOTION_ROOT / clip / "soma_motion.npz",
                       allow_pickle=True)["rest_joint_positions_m"].astype(np.float64) @ S2W.T
        out[clip] = (take, rest)
    return out


def clean_arm() -> dict:
    block: dict = {
        "reference": "exact synthetic truth -- the posed SOMASKEL77 Hips world rotation, by "
                     "Kabsch of the clip's OWN rest pelvis offsets onto the posed ones. "
                     "MAMMA-FREE.",
        "what_it_proves": "the CONSTRUCTION, and the separation of the pelvis candidates "
                          "from the controls and the lumbar arms. It CANNOT select between "
                          "the rigid candidates: they are exact by construction.",
        "clips": {},
    }
    for clip, (take, rest) in load().items():
        truth, residual = truth_frames(take, rest)
        tilt = tilt_deg(take)
        bent = tilt > BENT_TILT_DEG
        row = {"frames": int(len(take)), "kabsch_residual_m": residual,
               "tilt_median_deg": round(float(np.median(tilt)), 2),
               "bent_frames": int(bent.sum()), "arms": {}}
        for arm in ALL_ARMS:
            own = geodesic_deg(candidate_frames(take, arm, rest), truth)
            shipped = geodesic_deg(candidate_frames(take, arm, None), truth)
            row["arms"][arm] = {
                "own_rest_median_deg": round(float(np.median(own)), 4),
                "own_rest_max_deg": round(float(own.max()), 4),
                "shipped_constants_median_deg": round(float(np.median(shipped)), 3),
                "bent_own_rest_median_deg": round(float(np.median(own[bent])), 4) if bent.any() else None,
                "bent_shipped_median_deg": round(float(np.median(shipped[bent])), 3) if bent.any() else None,
            }
        block["clips"][clip] = row
    return block


def segment_sd_mm(pos: np.ndarray) -> dict:
    """The rigidity instrument's own figure, on synthetic noise.

    THIS IS THE INSTRUMENT'S CALIBRATION. `d7_pelvis_rigidity.py` measured the same two
    segments on the REAL take at 5.09 / 6.61 mm (subject 0) and 8.44 / 11.10 mm
    (subject 1). If the synthetic arm's noise produces a much larger spread than that, the
    noise model is PESSIMISTIC and every angular figure on it is a lower bound on quality
    -- which is a statement about the instrument, not about the pelvis.
    """
    mid = 0.5 * (pos[:, N["LeftLeg"]] + pos[:, N["RightLeg"]])
    root, s1 = pos[:, N["Hips"]], pos[:, N["Spine1"]]
    return {
        "root_to_spine1_sd_mm": round(float(np.linalg.norm(s1 - root, axis=1).std() * 1000.0), 2),
        "midhips_to_spine1_sd_mm": round(float(np.linalg.norm(s1 - mid, axis=1).std() * 1000.0), 2),
        "real_take_measured_sd_mm": {"subject_00": [5.09, 6.61], "subject_01": [8.44, 11.10]},
    }


def noisy_arm(stride: int, label: str, *, smooth_spine: bool) -> dict:
    rig = load_camera_rig(RIG)
    cameras = tuple(c.scaled(WORKING_WIDTH, WORKING_HEIGHT) for c in rig)
    block: dict = {
        "reference": "the same exact truth; I7/I8's measured heavy-tail FRAME-CORRELATED "
                     "noise injected in PIXELS at NOISE_SIGMA_PX = "
                     f"{sweep.NOISE_SIGMA_PX} px and recovered through the real "
                     "`triangulate_point`. Our own detector's amplitude; never MAMMA's.",
        "seeds": list(SEEDS), "stride": stride, "spine_input_smoothed": smooth_spine,
        "clips": {},
    }
    for clip, (take, rest) in load(stride).items():
        truth, _ = truth_frames(take, rest)
        tilt = tilt_deg(take)
        bent = tilt > BENT_TILT_DEG
        truth_yaw = yaw_deg(truth)
        truth_rate = float(np.percentile(np.abs(np.diff(truth_yaw)), 95))
        arms: dict = {arm: {str(w): [] for w in WINDOWS} for arm in ALL_ARMS}
        bent_arms: dict = {arm: {str(w): [] for w in WINDOWS} for arm in ALL_ARMS}
        lags: dict = {str(w): [] for w in WINDOWS}
        atten: dict = {str(w): [] for w in WINDOWS}
        calibration = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            noisy = observe(cameras, take, rng, smooth_spine=smooth_spine)
            calibration.append(segment_sd_mm(noisy))
            for arm in ALL_ARMS:
                base = candidate_frames(noisy, arm, None)
                for window in WINDOWS:
                    if window and len(take) < window:
                        continue
                    error = geodesic_deg(smooth_frames(base, window), truth)
                    arms[arm][str(window)].append(error)
                    if bent.any():
                        bent_arms[arm][str(window)].append(error[bent])
            reference_arm = candidate_frames(noisy, "B_hipmid_to_spine1", None)
            for window in WINDOWS:
                if window and len(take) < window:
                    continue
                smoothed = smooth_frames(reference_arm, window)
                candidate_yaw = yaw_deg(smoothed)
                lags[str(window)].append(sweep.lag_frames(truth_yaw, candidate_yaw))
                atten[str(window)].append(
                    float(np.percentile(np.abs(np.diff(candidate_yaw)), 95)) / max(truth_rate, 1e-9))

        def stat(pool: list[np.ndarray]) -> dict | None:
            if not pool:
                return None
            values = np.concatenate(pool)
            return {"median_deg": round(float(np.median(values)), 3),
                    "p95_deg": round(float(np.percentile(values, 95)), 3),
                    "n": int(values.size)}

        block["clips"][clip] = {
            "frames": int(len(take)), "bent_frames": int(bent.sum()),
            "tilt_median_deg": round(float(np.median(tilt)), 2),
            "truth_yaw_rate_p95_deg_per_frame": round(truth_rate, 3),
            "all_frames": {arm: {w: stat(v) for w, v in windows.items()}
                           for arm, windows in arms.items()},
            "bent_frames_only": {arm: {w: stat(v) for w, v in windows.items()}
                                 for arm, windows in bent_arms.items()},
            "noise_calibration_vs_the_real_take": {
                "root_to_spine1_sd_mm": round(float(np.mean(
                    [c["root_to_spine1_sd_mm"] for c in calibration])), 2),
                "midhips_to_spine1_sd_mm": round(float(np.mean(
                    [c["midhips_to_spine1_sd_mm"] for c in calibration])), 2),
                "real_take_measured_sd_mm": calibration[0]["real_take_measured_sd_mm"],
            },
            "window_lag_frames": {w: round(float(np.median(v)), 3) for w, v in lags.items() if v},
            "window_attenuation": {w: round(float(np.median(v)), 4) for w, v in atten.items() if v},
        }
    return block


SIGMA_SCALES = (1.0, 0.75, 0.5, 0.35, 0.25)


def calibration_sweep() -> dict:
    """AN INSTRUMENT REPAIR, AND IT WAS NOT PRE-REGISTERED. Recorded as post-hoc.

    The noisy arm at the pre-registered sigma produces a `mid(hips)->Spine1` length spread
    of ~19 mm, where `d7_pelvis_rigidity.py` measures 6.61 and 11.10 mm on the REAL take
    with the REAL detector. The noise model is therefore 1.7-2.9x too harsh, and every
    angular figure on it is a pessimistic bound. CLAUDE.md: *when a fit is short, suspect
    the instrument and the free parameters before reaching for more input.*

    This sweeps the pixel sigma and reports, at each scale, the synthetic segment spread
    beside the real take's -- so the reader can read every arm at the noise level the real
    detector actually delivers. It is MAMMA-FREE: the calibration target is our own
    capture's own reference-free length invariant. The PRE-REGISTERED verdict stands on
    scale 1.0 and is recorded as it fell.
    """
    rig = load_camera_rig(RIG)
    cameras = tuple(c.scaled(WORKING_WIDTH, WORKING_HEIGHT) for c in rig)
    take, rest = load(1)[SQUAT]
    truth, _ = truth_frames(take, rest)
    bent = tilt_deg(take) > BENT_TILT_DEG
    rows = {}
    for scale in SIGMA_SCALES:
        sds, errors = [], {arm: {str(w): [] for w in WINDOWS} for arm in ALL_ARMS}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            noisy = observe(cameras, take, rng, smooth_spine=True, sigma_scale=scale)
            sds.append(segment_sd_mm(noisy))
            for arm in ALL_ARMS:
                base = candidate_frames(noisy, arm, None)
                for window in WINDOWS:
                    if window and len(take) < window:
                        continue
                    errors[arm][str(window)].append(
                        geodesic_deg(smooth_frames(base, window), truth)[bent])
        rows[f"sigma_x{scale}"] = {
            "sigma_px": round(sweep.NOISE_SIGMA_PX * scale, 3),
            "synthetic_midhips_to_spine1_sd_mm": round(float(np.mean(
                [v["midhips_to_spine1_sd_mm"] for v in sds])), 2),
            "synthetic_root_to_spine1_sd_mm": round(float(np.mean(
                [v["root_to_spine1_sd_mm"] for v in sds])), 2),
            "bent_median_deg": {
                arm: {w: round(float(np.median(np.concatenate(v))), 3)
                      for w, v in windows.items() if v}
                for arm, windows in errors.items()},
        }
    return {
        "status": "POST-HOC INSTRUMENT REPAIR, not pre-registered",
        "why": calibration_sweep.__doc__,
        "real_take_measured_midhips_to_spine1_sd_mm": {"subject_00": 6.61, "subject_01": 11.10},
        "scales": rows,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "instrument": "D7 pelvis-frame candidate and window selection on synthetic truth",
        "selection_rule": (
            "PRE-REGISTERED: the candidate with the LOWEST MEDIAN orientation error on the "
            "NOISY arm, on the squat clip's bent frames (tilt > 20 deg), pooled over 5 "
            "seeds; a tie within 0.5 deg broken by the smaller convention spread. The "
            "clean arm cannot select; the MAMMA arm selects nothing at any point."),
        "clean": clean_arm(),
        "noisy_stride_1": noisy_arm(1, "real-time", smooth_spine=True),
        "noisy_stride_1_spine_unsmoothed": noisy_arm(1, "real-time", smooth_spine=False),
        "noisy_fast": noisy_arm(FAST_STRIDE, "fast", smooth_spine=True),
        "noise_calibration_sweep": calibration_sweep(),
    }
    squat_bent = report["noisy_stride_1"]["clips"][SQUAT]["bent_frames_only"]
    scores = {arm: squat_bent[arm]["0"]["median_deg"] for arm in CANDIDATES}
    winner = min(scores, key=scores.get)
    report["selected_candidate"] = {"winner": winner, "median_deg_by_candidate": scores}
    fast_squat = report["noisy_fast"]["clips"][SQUAT]
    report["selected_window"] = {
        "p95_by_window": {w: v["p95_deg"] for w, v in
                          report["noisy_stride_1"]["clips"][SQUAT]["bent_frames_only"][winner].items()
                          if v},
        "lag_frames_fast_clip": fast_squat["window_lag_frames"],
        "attenuation_fast_clip": fast_squat["window_attenuation"],
    }
    (OUT_DIR / "synthetic.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["clean"]["clips"][SQUAT]["arms"], indent=1))
    print("\nNOISY, squat, bent frames, window 0:")
    for arm in ALL_ARMS:
        row = squat_bent[arm]["0"]
        print(f"  {arm:26s} median {row['median_deg']:7.3f}  p95 {row['p95_deg']:8.3f}")
    print(f"\nselected candidate: {winner}  {scores}")
    print(f"window lag (fast clip):  {report['selected_window']['lag_frames_fast_clip']}")
    print(f"window attenuation:      {report['selected_window']['attenuation_fast_clip']}")
    print(f"window p95 (real speed): {report['selected_window']['p95_by_window']}")
    print("\nNOISE CALIBRATION SWEEP (squat, bent frames, window 0):")
    for name, row in report["noise_calibration_sweep"]["scales"].items():
        arms = row["bent_median_deg"]
        print(f"  {name:12s} sigma {row['sigma_px']:5.2f}px  midhips->S1 sd "
              f"{row['synthetic_midhips_to_spine1_sd_mm']:6.2f} mm  |  "
              f"C {arms['C_kabsch_pelvis']['0']:6.2f}  B {arms['B_hipmid_to_spine1']['0']:6.2f}  "
              f"worldvert {arms['world_vertical']['0']:6.2f}  thorax {arms['thorax_as_pelvis']['0']:6.2f}")
    print(f"\nwrote {OUT_DIR / 'synthetic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
