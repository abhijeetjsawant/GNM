#!/usr/bin/env python3
"""I5 -- the hand fit's held-out-camera protocol, as a JSON report.

    .venv/bin/python tools/hands/hand_fit_report.py            # everything
    .venv/bin/python tools/hands/hand_fit_report.py --hand subj0-l   # one hand
    .venv/bin/python tools/hands/hand_fit_report.py --assemble       # report only

Rung 8 of `docs/SUBSTITUTION_LADDER.md` has had its instrument written down in a
document since 2026-08-29 and nowhere else, which the ladder's own rule calls
*instrument missing*. This is that instrument: it wraps `fit_hand_sequence` --
never re-implements it -- runs leave-one-camera-out on all four hands and all
four cameras, and writes `artifacts/hands-lane/hand-fit-heldout.json`.


THE SELECTION RULE, which is the first thing this file exists to write down
--------------------------------------------------------------------------
**Every weight, prior, threshold and initialisation reported here is selected by
the held-out camera and by nothing else.** Fit on N-1 cameras, score reprojection
on the camera the fit never saw, rotate over all four, and read the fold spread
as well as the mean.

Explicitly NOT selectors, and the reason each is excluded:

* **MAMMA agreement is never a selector.** MAMMA is a research-licensed measuring
  instrument that never ships (CLAUDE.md). Its fingertips appear here on their own
  reference string as an *agreement* figure and are never optimised toward.
* **The temporal term is never a selector.** `fit_hand_sequence` regularises
  wrist-relative joint *acceleration* -- which is, to within the frame in which it
  is expressed, the quantity this report calls **jitter**. A band on a quantity
  the solver optimises directly is a knob setting, not evidence (CLAUDE.md). Jitter
  is reported because it is the thrash figure the rung asks for; it rejects no
  control, and every control below names the figure that does reject it.
* **In-sample reprojection is never a selector.** It is the data term.

The gauge problem this protocol was built to escape: SMPL-X axis-angle spread and
MHR Euler-channel spread are incomparable quantities, so the angle standard
deviation that increment 5 reported (18-24 deg/DoF against MAMMA's 5.6) means
nothing as a ratio (`BATTLE1_INCREMENT5_ARTICULATION_VARIANCE.md`, "The old
comparison was never valid"). The repo replaced it with wrist-local fingertip
**position**, which is parameterisation-invariant, and that is the thrash figure
carried here.


THE ARMS
--------
* **candidate** -- `fit_hand_sequence` at its *shipped defaults*
  (`smooth_weight=2.0`, `pose_smooth_weight=0.25`, `cross_view_weights` at 9 px).
  Not arm C (prior 1.0) or arm D (prior 4.0) of increment 6: those are documented
  configurations that were never the default, and this report scores the code as
  it stands. Changing the default is solver work and belongs behind this gate.
* **CONTROL frozen rest hand** -- every angle zero, wrist rotation identity,
  anchored at the true wrist. Articulation sd 0 by construction.
* **CONTROL frozen medoid pose** -- the candidate's own most-central wrist-relative
  configuration, held for the whole take. This is the *well-placed* constant and
  the harder control: it is a real, anatomically valid hand in a plausible pose.
* **CONTROL lag k** (k = 1, 3, 5) -- the candidate's wrist-relative configuration
  from frame f-k, anchored at frame f's true wrist. Bone lengths exact.
* **CONTROL duplicated frames** -- f -> 2*floor(f/2); every odd frame copied from
  its predecessor.
* **CONTROL low-pass** -- a 15-frame (0.5 s) zero-phase box filter on the
  wrist-relative configuration, and its causal (lagged) twin.
* **CONTROL prior-dominated** -- a real refit at `pose_smooth_weight=1000`.
* **ORACLE** -- MAMMA's own hand joints, mapped slot-for-slot onto the MHR chain
  and projected into the held-out camera, scored against the identical held-out
  observations. MAMMA saw all four cameras, so this is not a held-out score: it is
  the **floor the protocol imposes** -- SOMA-77's own 2D error plus the
  joint-convention gap between MHR's hand and SMPL-X's, which is reported per slot
  in millimetres beside it so the reader can see how much of the floor is
  convention.

Every arm is scored on the **identical** (frame, joint) test set: the retained
cross-view gate `keep[:, held_out]`, which is geometry among cameras and not a
function of any solver. Same denominator, per the standing rule.


WHAT THIS INSTRUMENT IS BLIND TO
--------------------------------
* **Depth along the held-out camera's rays.** Reprojection into one camera cannot
  score it, and at ~4.7 m on a ~20 px hand that is the direction fingers are least
  determined in. The 35 mm-class figures are transverse.
* **Accuracy.** There is no ground truth on disk (`gt_joints` is a byte-copy of
  `pred_joints`). Held-out reprojection scores *view generalisation*; the MAMMA
  columns score *agreement with another estimator*.
* **Anything the test set's own gate removed.** `keep` is computed with the
  held-out camera included, so an observation the held-out view disagrees with the
  other three about is not scored. The ungated denominator is reported beside every
  gated one.
* **Contact precision.** An open hand and a fist differ by 60-80 mm at the tips, so
  grasp state clears this noise and touch does not.
* **One take.** 150 frames, two performers, five seconds, lag-1 autocorrelation of
  the residual series reported per fold. Block bootstrap only; no generalisation
  beyond this take.

FIRST CANDIDATE FIX, recorded and deliberately not attempted here: `smooth_weight`
multiplies a second difference in **radians** against a data block in **pixels**,
which measured 1.9 % of the objective. Reconciling those units is solver work; it
belongs after this report exists as its gate, not inside the step that builds it.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))

from autoanim_gnm import commercial_multiview as cm  # noqa: E402
from autoanim_gnm import hand_fit as hf  # noqa: E402
from subject_map import mamma_index_for  # noqa: E402

ARRAYS = ROOT / "artifacts/handfit-arrays"
RIG = ROOT / "artifacts/commercial-multiview-soma77/camera-rig.json"
MA3D = ROOT / (
    "artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/"
    "pushing_and_lifting_from_ground"
)
OUT = ROOT / "artifacts/hands-lane"
CACHE = OUT / "cache"
REPORT = OUT / "hand-fit-heldout.json"

SCHEMA_VERSION = "autoanim.hand-fit-heldout/1.0"
CAMERAS = ("A001", "B001", "C001", "D001")
# SOMA-77 runs at 1280 wide and every pixel-denominated figure in this lane is
# quoted at that width. The retained observations are in that space (x up to
# ~925, y up to ~465 on 1280x720), so the rig is scaled to match rather than the
# observations being scaled to the rig.
WORKING_WIDTH, WORKING_HEIGHT = 1280, 720
HANDS = ("subj0-l", "subj0-r", "subj1-l", "subj1-r")

# Half a second per block. Per-frame agreement in this lane has lag-1
# autocorrelation ~0.99, so ordinary resampling is invalid and every margin is
# quoted from a moving-block bootstrap with candidate and control on identical draws.
BLOCK_FRAMES = 15
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20260902

GATE_THRESHOLD_PX = 9.0          # the retained `keep` mask's threshold
LOWPASS_FRAMES = 15
PRIOR_DOMINATED_WEIGHT = 1000.0
# A hand with under a millimetre of fingertip excursion is not moving. The
# threshold is reference-free on purpose: MAMMA's own amplitude would be a
# band read off the instrument that is forbidden from selecting anything.
COLLAPSED_AMPLITUDE_MM = 1.0

# MHR chain slots, by name, in the order `build_hand_chain` returns them.
KNUCKLES = ("index1", "middle1", "ring1", "pinky1")
TIPS = ("thumb_null", "index_null", "middle_null", "ring_null", "pinky_null")

# SMPL-X hand joints in MAMMA's 127-joint `pred_joints`. DERIVED, not recalled:
# `derive_smplx_hand_layout` re-establishes the block and the tip permutation from
# the file itself on every run and refuses to continue if they move. The tip
# permutation is the part worth deriving -- the naive reading (66=index ... 70=thumb,
# matching the 25..39 chain order) is WRONG; optimal assignment on median
# tip-to-distal distance gives thumb, index, middle, ring, pinky, consistently on
# both bodies and both sides.
SMPLX_WRIST = {"l": 20, "r": 21}
SMPLX_CHAIN_BASE = {"l": 25, "r": 40}
SMPLX_TIP_BASE = {"l": 66, "r": 71}
SMPLX_CHAIN_ORDER = ("index", "middle", "pinky", "ring", "thumb")


# ------------------------------------------------------------------ provenance

def sha256_of(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return "unknown"


# ----------------------------------------------------------------------- input

def cameras() -> tuple:
    return tuple(c.scaled(WORKING_WIDTH, WORKING_HEIGHT) for c in cm.load_camera_rig(RIG))


def load_hand(name: str) -> dict:
    blob = np.load(ARRAYS / f"{name}.npz", allow_pickle=True)
    return {
        "observations": blob["observations"].astype(np.float64),
        "wrist": blob["wrist"].astype(np.float64),
        "keep": blob["keep"].astype(bool),
        "joints": [str(j) for j in blob["joints"]],
        "angles": blob["angles"].astype(np.float64),
        "positions": blob["positions"].astype(np.float64),
        "subject": int(name[4]),
        "side": name[-1],
    }


def slot_index(joints: list[str], side: str, short: str) -> int:
    return joints.index(f"{side}_{short}")


# --------------------------------------------------- wrapping reproduction check

def gate_reproduction(cams, hand: dict) -> dict:
    """Does re-running the retained gate reproduce the retained mask exactly?

    The standing rule is *wrap the pipeline, never re-implement it* -- a careful
    hand replication of an association loop drifted 9-19 mm where wrapping the real
    function reproduced it at 0.0. This report's test set is the retained `keep`
    mask, so the anchor is that `gate_observations_by_cross_view_agreement`, called
    on the retained observations at the documented threshold, returns that mask
    bit-for-bit. If it does not, the arrays and the code have drifted apart and no
    figure below is comparable to the documents.
    """
    recomputed = hf.gate_observations_by_cross_view_agreement(
        cams, hand["observations"], GATE_THRESHOLD_PX)
    disagree = int((recomputed != hand["keep"]).sum())
    return {
        "threshold_px": GATE_THRESHOLD_PX,
        "retained_kept": int(hand["keep"].sum()),
        "recomputed_kept": int(recomputed.sum()),
        "observations_disagreeing": disagree,
        "reproduced": disagree == 0,
    }


# -------------------------------------------------------------------- the fits

def training_weights(cams, observations: np.ndarray, held_out: int) -> np.ndarray:
    """`cross_view_weights` computed on the training cameras ONLY.

    The obvious wiring -- weight all four cameras, then zero the held-out column --
    leaks: the held-out view's pixels would have set the sigma of every training
    observation through the median epipolar distance. Fitting on three and scoring
    on the fourth then would not be leave-one-camera-out. So the weights are
    computed on the three-camera sub-rig and expanded back with a zero column,
    which is what makes `used[:, held_out]` False inside `fit_hand_sequence`.

    All four cameras stay in the call itself, so `_pixels_per_metre` -- and with it
    the position prior's strength -- is identical across the four folds.
    """
    train = [i for i in range(len(cams)) if i != held_out]
    partial = hf.cross_view_weights(
        [cams[i] for i in train], observations[:, train], threshold_px=GATE_THRESHOLD_PX)
    weights = np.zeros(observations.shape[:3], dtype=np.float64)
    weights[:, train] = partial
    return weights


def all_camera_weights(cams, observations: np.ndarray) -> np.ndarray:
    return hf.cross_view_weights(cams, observations, threshold_px=GATE_THRESHOLD_PX)


def cache_path(hand_name: str, arm: str, held_out: int | None) -> Path:
    return CACHE / f"{hand_name}-{arm}-{'all' if held_out is None else CAMERAS[held_out]}.npz"


def run_fit(cams, hand: dict, held_out: int | None, arm: str,
            *, compute: bool = True) -> np.ndarray | None:
    """One cached fit. Returns fitted joint positions [F,J,3] in metres.

    `compute=False` returns None rather than solving, so a report can be assembled
    from whatever the fit grid has reached. An arm that is absent is reported as
    absent; it is never filled with a placeholder.
    """
    path = cache_path(hand["name"], arm, held_out)
    if path.exists():
        return np.load(path)["positions"]
    if not compute:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    chain = hf.build_hand_chain(hand["side"])
    weights = (all_camera_weights(cams, hand["observations"]) if held_out is None
               else training_weights(cams, hand["observations"], held_out))
    kwargs: dict = {}
    if arm == "prior_dominated":
        kwargs["pose_smooth_weight"] = PRIOR_DOMINATED_WEIGHT
    started = time.time()
    _, positions, _ = hf.fit_hand_sequence(
        cams, chain, hand["wrist"], hand["observations"], weights=weights, **kwargs)
    # Written to a sibling and renamed, because a half-written .npz is
    # indistinguishable from a finished one to `path.exists()` and a later pass
    # would read a truncated array as if it were a fit.
    # `np.savez_compressed` appends `.npz` unless the name already ends in it, so
    # the temporary name has to end in `.npz` too or the rename below chases a file
    # that was never created.
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(
        partial, positions=positions, seconds=np.asarray(time.time() - started),
        observations_used=np.asarray(int((weights > 0.0).sum())))
    partial.rename(path)
    return positions


def fit_seconds(hand_name: str, arm: str, held_out: int | None) -> float | None:
    path = CACHE / f"{hand_name}-{arm}-{'all' if held_out is None else CAMERAS[held_out]}.npz"
    if not path.exists():
        return None
    return round(float(np.load(path)["seconds"]), 1)


# ------------------------------------------------------- degenerate arms, post hoc

def _local_configurations(positions: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    return positions - wrist[:, None, :]


def _reanchor(configurations: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    return configurations + wrist[:, None, :]


def frozen_rest(chain, wrist: np.ndarray) -> np.ndarray:
    """Angles zero, wrist rotation identity: the initialisation, held."""
    identity = np.asarray([0.0, 0.0, 0.0, 1.0])
    zero = np.zeros(chain.degrees_of_freedom)
    return np.stack([hf.forward_kinematics(chain, wrist[f], identity, zero)
                     for f in range(len(wrist))])


def frozen_medoid(positions: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """The candidate's own most central pose, held for the take.

    The medoid rather than the mean, so the constant is an exactly valid hand --
    a mean of configurations is not one, and a control that is anatomically
    impossible is a weaker control than one that is not.
    """
    local = _local_configurations(positions, wrist)
    flat = local.reshape(len(local), -1)
    distances = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2).sum(axis=1)
    medoid = local[int(np.argmin(distances))]
    return _reanchor(np.repeat(medoid[None], len(local), axis=0), wrist)


def lagged(positions: np.ndarray, wrist: np.ndarray, frames: int) -> np.ndarray:
    local = _local_configurations(positions, wrist)
    index = np.clip(np.arange(len(local)) - frames, 0, None)
    return _reanchor(local[index], wrist)


def duplicated(positions: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    local = _local_configurations(positions, wrist)
    index = (np.arange(len(local)) // 2) * 2
    return _reanchor(local[index], wrist)


def low_passed(positions: np.ndarray, wrist: np.ndarray, *, causal: bool) -> np.ndarray:
    """Box filter on the wrist-relative configuration.

    Bone lengths shrink slightly under an average of configurations (by 0.5-2 % on
    this take), which is itself part of what makes this degenerate; the arm is
    labelled as such rather than renormalised.
    """
    local = _local_configurations(positions, wrist)
    kernel = np.ones(LOWPASS_FRAMES) / LOWPASS_FRAMES
    flat = local.reshape(len(local), -1)
    padded = np.pad(flat, ((LOWPASS_FRAMES, LOWPASS_FRAMES), (0, 0)), mode="edge")
    smoothed = np.stack(
        [np.convolve(padded[:, k], kernel, mode="same") for k in range(flat.shape[1])],
        axis=1)[LOWPASS_FRAMES:LOWPASS_FRAMES + len(flat)]
    out = smoothed.reshape(local.shape)
    if causal:
        shift = LOWPASS_FRAMES // 2
        out = out[np.clip(np.arange(len(out)) - shift, 0, None)]
    return _reanchor(out, wrist)


# ----------------------------------------------------------------- held-out score

def reproject(camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project, tolerating non-finite rows.

    The oracle arm carries NaN in the two MHR slots SMPL-X has no counterpart for
    (`thumb0`, `pinky0`), and `CalibratedCamera.project` rightly refuses non-finite
    input. Those slots carry no SOMA-77 observation either, so they fall out of the
    test set anyway; they are projected as NaN rather than dropped, so every arm
    keeps the same [F,J] shape and the same denominator.
    """
    flat = points.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    uv = np.full((len(flat), 2), np.nan)
    depth = np.full(len(flat), np.nan)
    if finite.any():
        uv[finite], depth[finite] = camera.project(flat[finite])
    return uv.reshape(points.shape[:-1] + (2,)), depth.reshape(points.shape[:-1])


def held_out_residuals(camera, positions: np.ndarray, hand: dict,
                       held_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-observation reprojection residual on the fixed test set.

    Returns `(per_observation_px, frame_of_each)` over the retained gate's
    surviving (frame, joint) pairs in the held-out camera -- the same set for every
    arm, so no arm can move its own denominator.
    """
    test = hand["keep"][:, held_out]
    uv, depth = reproject(camera, positions)
    observed = hand["observations"][:, held_out, :, :2]
    error = np.linalg.norm(uv - observed, axis=2)
    error = np.where(np.isfinite(depth) & (depth > 0.0), error, np.nan)
    frames = np.repeat(np.arange(positions.shape[0])[:, None], positions.shape[1], axis=1)
    return error[test], frames[test]


def millimetres_at_the_subject(camera, positions: np.ndarray, hand: dict,
                               held_out: int) -> np.ndarray:
    """Per-observation pixels -> millimetres, using each point's own depth.

    A pixel is not a fixed distance: it is `depth / focal` metres at the subject,
    and the four cameras stand at different ranges. Converting with each
    observation's own depth rather than a take-wide constant keeps the mm column
    honest across folds.
    """
    test = hand["keep"][:, held_out]
    _, depth = reproject(camera, positions)
    focal = float(camera.intrinsics[0, 0] + camera.intrinsics[1, 1]) / 2.0
    return (depth[test] / focal) * 1000.0


def score_arm(camera, positions: np.ndarray, hand: dict, held_out: int) -> dict:
    pixels, frames = held_out_residuals(camera, positions, hand, held_out)
    scale = millimetres_at_the_subject(camera, positions, hand, held_out)
    finite = np.isfinite(pixels)
    pixels, frames, scale = pixels[finite], frames[finite], scale[finite]
    millimetres = pixels * scale
    return {
        "observations_scored": int(pixels.size),
        "median_px": round(float(np.median(pixels)), 2),
        "p95_px": round(float(np.percentile(pixels, 95)), 2),
        "median_mm_at_the_subject": round(float(np.median(millimetres)), 1),
        "p95_mm_at_the_subject": round(float(np.percentile(millimetres, 95)), 1),
        "_pixels": pixels,
        "_frames": frames,
    }


# ------------------------------------------------------------------- articulation

def wrist_local_frame(positions: np.ndarray, wrist: np.ndarray,
                      knuckles: list[int], index1: int, pinky1: int) -> np.ndarray:
    """Rotate joint positions into a basis built from POSITIONS ONLY.

    Neither estimator's solved rotation may enter: MHR's Euler channels and
    SMPL-X's axis-angle triples are the incomparable quantities this whole metric
    exists to avoid. Origin at the wrist, first axis to the knuckle centroid, third
    the palm normal, second completing the right-handed set -- built identically on
    both sides so the two are comparable.
    """
    origin = wrist
    forward = positions[:, knuckles].mean(axis=1) - origin
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    across = positions[:, index1] - positions[:, pinky1]
    normal = np.cross(forward, across)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    second = np.cross(normal, forward)
    basis = np.stack([forward, second, normal], axis=1)          # [F,3,3] rows
    return np.einsum("fac,fjc->fja", basis, positions - origin[:, None, :])


def articulation(local_tips_m: np.ndarray, hand_length_m: float) -> dict:
    """Amplitude, jitter, roughness over the five fingertips, in millimetres.

    * amplitude -- RMS distance of a tip from its own take mean, averaged over tips.
      *Is it moving.*
    * jitter -- median norm of the discrete second difference. *Is the motion
      smooth.* THE SOLVER OPTIMISES THIS (`pose_smooth_weight` penalises exactly
      this acceleration, in the wrist frame rather than this one), so it rejects
      nothing; it is the rung's thrash figure and is read only against the
      controls that are NOT smooth.
    * roughness -- jitter / amplitude, and 0/0 when a control has frozen, so it is
      suppressed below 1 mm of amplitude rather than reported as a large ratio of
      two vanishing numbers.
    """
    tips = local_tips_m * 1000.0
    centred = tips - tips.mean(axis=0, keepdims=True)
    amplitude = float(np.mean(np.sqrt((centred ** 2).sum(axis=2).mean(axis=0))))
    second = tips[:-2] - 2.0 * tips[1:-1] + tips[2:]
    jitter = float(np.median(np.linalg.norm(second, axis=2)))
    return {
        "amplitude_mm": round(amplitude, 2),
        "amplitude_percent_of_hand_length": round(
            100.0 * amplitude / (hand_length_m * 1000.0), 1),
        "jitter_mm": round(jitter, 3),
        "roughness": (round(jitter / amplitude, 4) if amplitude >= 1.0 else None),
        "roughness_suppressed_amplitude_below_1mm": amplitude < 1.0,
    }


def translation_only_jitter(positions: np.ndarray, wrist: np.ndarray,
                            tips: list[int]) -> dict:
    """The reading the wrist-local frame is blind to, plus the anchor's own.

    The knuckles hang rigidly off the wrist, so the local basis TURNS WITH THE
    WRIST -- which is what makes it the right frame for articulation and what makes
    it unable to see a thrashing wrist. Removing only wrist *translation* exposes
    it. And the wrist is an input to `fit_hand_sequence`, not a variable, so the
    body track's own translational jitter is a floor under every world-space hand
    figure: reported beside it, never subtracted from it.
    """
    relative = (positions[:, tips] - wrist[:, None, :]) * 1000.0
    second = relative[:-2] - 2.0 * relative[1:-1] + relative[2:]
    anchor = wrist * 1000.0
    anchor_second = anchor[:-2] - 2.0 * anchor[1:-1] + anchor[2:]
    return {
        "fingertip_jitter_wrist_translation_removed_only_mm":
            round(float(np.median(np.linalg.norm(second, axis=2))), 3),
        "wrist_anchor_translational_jitter_mm":
            round(float(np.median(np.linalg.norm(anchor_second, axis=1))), 3),
    }


# ------------------------------------------------------------------------ MAMMA

def derive_smplx_hand_layout(joints: np.ndarray) -> dict:
    """Re-establish MAMMA's hand joint block and its tip permutation from the file.

    Two independent structural readings, both required to agree with the layout the
    constants above assert, or this refuses to continue:

    1. exactly 20 joints of the 127 sit within 15 cm of each wrist, and they are the
       asserted chain block plus the asserted tip block;
    2. the tip permutation is recovered by optimal assignment on the median
       distance from each chain's distal joint to each tip -- not by index order,
       which is wrong here. The naive reading gives 66=index; the data gives
       66=thumb, on both bodies and both sides.
    """
    from scipy.optimize import linear_sum_assignment

    layout: dict = {}
    for side in ("l", "r"):
        wrist = SMPLX_WRIST[side]
        base, tip_base = SMPLX_CHAIN_BASE[side], SMPLX_TIP_BASE[side]
        distance = np.linalg.norm(joints[0] - joints[0, wrist], axis=1)
        near = {int(i) for i in np.flatnonzero(distance < 0.15)} - {wrist}
        expected = set(range(base, base + 15)) | set(range(tip_base, tip_base + 5))
        if near != expected:
            raise SystemExit(
                f"SMPL-X hand block for side {side} is not where this instrument "
                f"asserts it: {sorted(near)} against {sorted(expected)}")
        cost = np.zeros((5, 5))
        for chain in range(5):
            distal = joints[:, base + 3 * chain + 2]
            for tip in range(5):
                cost[chain, tip] = np.median(
                    np.linalg.norm(joints[:, tip_base + tip] - distal, axis=1))
        rows, columns = linear_sum_assignment(cost)
        for chain, tip in zip(rows, columns):
            chain, tip = int(chain), int(tip)
            name = SMPLX_CHAIN_ORDER[chain]
            layout[f"{side}_{name}"] = {
                "chain": [int(base + 3 * chain), int(base + 3 * chain) + 1,
                          int(base + 3 * chain) + 2],
                "tip": int(tip_base + tip),
                "median_distal_to_tip_mm": round(float(cost[chain, tip]) * 1000.0, 1),
            }
    return layout


def mamma_slots(layout: dict, joints_names: list[str], side: str) -> dict[int, int]:
    """MHR chain slot -> SMPL-X joint index, by chain semantics.

    MHR `X1,X2,X3,X_null` <-> SMPL-X `X1,X2,X3,tip` for the four fingers, and MHR
    `thumb1,thumb2,thumb3,thumb_null` <-> SMPL-X's three thumb joints and its tip.
    MHR's `thumb0` (carpometacarpal) and `pinky0` (metacarpal) have no SMPL-X
    counterpart -- and carry no SOMA-77 observation either, so they are absent from
    every figure rather than mapped approximately.
    """
    mapping: dict[int, int] = {}
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        entry = layout[f"{side}_{finger}"]
        for step, joint in enumerate(entry["chain"]):
            mapping[slot_index(joints_names, side, f"{finger}{step + 1}")] = joint
        mapping[slot_index(joints_names, side, f"{finger}_null")] = entry["tip"]
    return mapping


def mamma_hand_positions(body: int, side: str, layout: dict,
                         joints_names: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """MAMMA's hand on the MHR chain's own 22 slots, plus its wrist."""
    joints = np.load(MA3D / f"verts_joints_body_id-{body:02d}.npz",
                     allow_pickle=True)["pred_joints"].astype(np.float64)
    mapping = mamma_slots(layout, joints_names, side)
    positions = np.full((joints.shape[0], len(joints_names), 3), np.nan)
    for slot, joint in mapping.items():
        positions[:, slot] = joints[:, joint]
    return positions, joints[:, SMPLX_WRIST[side]], mapping


# -------------------------------------------------------------------- statistics

def block_draws(frames: int, draws: int, seed: int) -> np.ndarray:
    """Moving-block bootstrap indices, shared by every arm in a comparison.

    Identical draws or the comparison measures the draws instead of the arms.
    """
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(frames / BLOCK_FRAMES))
    starts = rng.integers(0, max(frames - BLOCK_FRAMES + 1, 1), size=(draws, blocks))
    offsets = np.arange(BLOCK_FRAMES)
    return np.clip((starts[:, :, None] + offsets).reshape(draws, -1), 0, frames - 1)


def bootstrap_medians(pixels: np.ndarray, frames: np.ndarray, index: np.ndarray,
                      total_frames: int) -> np.ndarray:
    order = np.argsort(frames, kind="stable")
    pixels, frames = pixels[order], frames[order]
    edges = np.searchsorted(frames, np.arange(total_frames + 1))
    medians = np.empty(len(index))
    for draw in range(len(index)):
        pool = np.concatenate(
            [pixels[edges[f]:edges[f + 1]] for f in index[draw]]) if len(index[draw]) else pixels
        medians[draw] = np.median(pool) if pool.size else np.nan
    return medians


def probability_candidate_beats(candidate: dict, other: dict, index: np.ndarray,
                                total_frames: int) -> float:
    """P(the candidate's bootstrapped median beats this arm's) on IDENTICAL draws.

    Identical draws or the comparison measures the resampling instead of the arms.
    The candidate's own draw medians are cached on its result so the ten comparisons
    in a fold share one pass.
    """
    if "_draw_medians" not in candidate:
        candidate["_draw_medians"] = bootstrap_medians(
            candidate["_pixels"], candidate["_frames"], index, total_frames)
    a = candidate["_draw_medians"]
    b = bootstrap_medians(other["_pixels"], other["_frames"], index, total_frames)
    good = np.isfinite(a) & np.isfinite(b)
    return round(float((a[good] < b[good]).mean()), 3)


def lag_one_autocorrelation(pixels: np.ndarray, frames: np.ndarray,
                            total_frames: int) -> float | None:
    series = np.full(total_frames, np.nan)
    for frame in range(total_frames):
        values = pixels[frames == frame]
        if values.size:
            series[frame] = np.median(values)
    good = np.isfinite(series[:-1]) & np.isfinite(series[1:])
    if good.sum() < 10:
        return None
    a, b = series[:-1][good], series[1:][good]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


# ------------------------------------------------------------------------- main

def arms_for(hand: dict, chain, candidate: np.ndarray) -> dict[str, np.ndarray]:
    wrist = hand["wrist"]
    return {
        "candidate": candidate,
        "CONTROL_frozen_rest_hand": frozen_rest(chain, wrist),
        "CONTROL_frozen_medoid_pose": frozen_medoid(candidate, wrist),
        "CONTROL_lag_1_frame": lagged(candidate, wrist, 1),
        "CONTROL_lag_3_frames": lagged(candidate, wrist, 3),
        "CONTROL_lag_5_frames": lagged(candidate, wrist, 5),
        "CONTROL_duplicated_frames": duplicated(candidate, wrist),
        "CONTROL_low_pass_zero_phase": low_passed(candidate, wrist, causal=False),
        "CONTROL_low_pass_causal_lagged": low_passed(candidate, wrist, causal=True),
    }


REJECTED_BY = {
    "CONTROL_frozen_rest_hand":
        "the held-out camera, and the collapsed-amplitude clause (0 mm by "
        "construction). Two independent clauses, neither of them a quantity the "
        "solver optimises on a camera it never saw.",
    "CONTROL_frozen_medoid_pose":
        "the held-out camera, and the collapsed-amplitude clause. This is the "
        "well-placed constant -- a real, anatomically valid hand pose taken from the "
        "candidate's own take -- so it is the harder of the two frozen arms and the "
        "one worth reading.",
    "CONTROL_lag_1_frame":
        "the held-out camera ALONE. Its jitter is identical to the candidate's -- a "
        "rigid time shift has the same second-difference distribution -- and its "
        "amplitude is identical too, so this arm is invisible to every figure except "
        "reprojection into a view the fit never saw. Whether one frame is enough for "
        "the held-out camera to see it is measured, not assumed.",
    "CONTROL_lag_3_frames":
        "the held-out camera alone; jitter and amplitude identical to the candidate's.",
    "CONTROL_lag_5_frames":
        "the held-out camera alone; jitter and amplitude identical to the candidate's.",
    "CONTROL_duplicated_frames":
        "the held-out camera. Frame doubling puts energy at the Nyquist frequency, so "
        "its jitter is WORSE than the candidate's -- a smoothness figure would happen "
        "to agree here, by the accident of this control's shape rather than because a "
        "smoothness figure can see a stalled solve.",
    "CONTROL_low_pass_zero_phase":
        "the held-out camera and the amplitude clause. Its jitter is far BETTER than "
        "the candidate's, so a jitter band would prefer it outright. Note what the "
        "measurement says: at the shipped `pose_smooth_weight` of 0.25 this arm is "
        "often NOT rejected, because half a second of zero-phase smoothing genuinely "
        "improves reprojection into an unseen camera. That is the shipped default "
        "being under-regularised, reported by the gate, not a hole in it.",
    "CONTROL_low_pass_causal_lagged":
        "the held-out camera. Jitter better than the candidate's and amplitude "
        "comparable, so reprojection is the only figure that can see the lag.",
    "CONTROL_prior_dominated":
        "the held-out camera ALONE, and this arm is the reason the rung needs one. A "
        "second-difference penalty at weight 1000 does not freeze the hand: zero "
        "acceleration is CONSTANT VELOCITY, so the solution drifts smoothly and "
        "measures LARGER amplitude than the candidate at a jitter better than MAMMA's. "
        "It passes a smoothness band, it passes an amplitude band, and it is wrong. "
        "Only reprojection into a camera the fit never saw can reject it.",
}


def measure_hand(name: str, cams, layout: dict, mapping_our_to_mamma: dict,
                 folds: list[int], compute: bool) -> dict | None:
    hand = load_hand(name)
    hand["name"] = name
    chain = hf.build_hand_chain(hand["side"])
    joints_names = hand["joints"]
    frames = hand["observations"].shape[0]

    knuckles = [slot_index(joints_names, hand["side"], k) for k in KNUCKLES]
    tips = [slot_index(joints_names, hand["side"], t) for t in TIPS]
    index1 = slot_index(joints_names, hand["side"], "index1")
    pinky1 = slot_index(joints_names, hand["side"], "pinky1")
    middle1 = slot_index(joints_names, hand["side"], "middle1")

    entry: dict = {
        "subject": hand["subject"],
        "side": hand["side"],
        "frames": frames,
        "gate_reproduction": gate_reproduction(cams, hand),
        "observations": {
            "present": int((np.isfinite(hand["observations"][..., :2]).all(axis=3)
                            & (hand["observations"][..., 2] > 0.0)).sum()),
            "kept_by_the_retained_cross_view_gate": int(hand["keep"].sum()),
            "slots_with_no_observation_at_all": [
                joints_names[j] for j in range(len(joints_names))
                if not hand["keep"][:, :, j].any()],
        },
        "folds": {},
        "articulation": {},
    }

    # ---- the four folds
    index = block_draws(frames, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    for held_out in folds:
        candidate = run_fit(cams, hand, held_out, "candidate", compute=compute)
        if candidate is None:
            continue
        arms = arms_for(hand, chain, candidate)
        prior = run_fit(cams, hand, held_out, "prior_dominated", compute=False)
        if prior is not None:
            arms["CONTROL_prior_dominated"] = prior
        mamma, mamma_wrist, _ = mamma_hand_positions(
            mapping_our_to_mamma[hand["subject"]], hand["side"], layout, joints_names)
        arms["ORACLE_mamma_hand_joints"] = mamma

        camera = cams[held_out]
        scored = {arm: score_arm(camera, positions, hand, held_out)
                  for arm, positions in arms.items()}
        fold: dict = {
            "held_out_camera": CAMERAS[held_out],
            "training_cameras": [CAMERAS[i] for i in range(4) if i != held_out],
            "test_observations_gated": int(hand["keep"][:, held_out].sum()),
            "test_observations_ungated": int(
                (np.isfinite(hand["observations"][:, held_out, :, :2]).all(axis=2)
                 & (hand["observations"][:, held_out, :, 2] > 0.0)).sum()),
            "training_observations_weighted_nonzero": int(np.load(
                CACHE / f"{name}-candidate-{CAMERAS[held_out]}.npz")["observations_used"]),
            "seconds_to_fit": fit_seconds(name, "candidate", held_out),
            "arms": {},
        }
        for arm, result in scored.items():
            row = {k: v for k, v in result.items() if not k.startswith("_")}
            if arm != "ORACLE_mamma_hand_joints":
                arm_local = wrist_local_frame(
                    arms[arm], hand["wrist"], knuckles, index1, pinky1)
                figures = articulation(arm_local[:, tips], float(np.median(
                    np.linalg.norm(candidate[:, middle1] - hand["wrist"], axis=1))))
                row["fingertip_amplitude_mm"] = figures["amplitude_mm"]
                row["fingertip_jitter_mm"] = figures["jitter_mm"]
            if arm != "candidate":
                row["P_candidate_beats_this_arm_block_bootstrap"] = \
                    probability_candidate_beats(scored["candidate"], result, index, frames)
            if arm.startswith("CONTROL"):
                # Two independent clauses, and a control fails if EITHER fires.
                # The held-out clause is the reprojection the fit never saw; the
                # amplitude clause is "is the hand moving at all", which no solver
                # here optimises and which is the clause the gate card names for the
                # frozen arms. The amplitude threshold is reference-free -- a
                # millimetre of excursion, not a band read off MAMMA -- because a
                # threshold taken from the reference would be the reference
                # selecting something.
                beaten = row["P_candidate_beats_this_arm_block_bootstrap"] >= 0.95
                collapsed = row["fingertip_amplitude_mm"] < COLLAPSED_AMPLITUDE_MM
                row["rejected_by"] = REJECTED_BY[arm]
                row["rejected_by_held_out_camera"] = bool(beaten)
                row["rejected_by_collapsed_amplitude"] = bool(collapsed)
                row["rejected"] = bool(beaten or collapsed)
            fold["arms"][arm] = row
        fold["lag1_autocorrelation_of_the_candidate_residual_series"] = \
            lag_one_autocorrelation(scored["candidate"]["_pixels"],
                                    scored["candidate"]["_frames"], frames)
        entry["folds"][CAMERAS[held_out]] = fold

    values = [f["arms"]["candidate"]["median_mm_at_the_subject"]
              for f in entry["folds"].values()]
    if values:
        entry["held_out_summary"] = {
            "mean_over_folds_mm": round(float(np.mean(values)), 1),
            "worst_fold_mm": round(float(np.max(values)), 1),
            "best_fold_mm": round(float(np.min(values)), 1),
            "folds_measured": len(values),
            "note": "the spread across folds is wide and it is reported rather than "
                    "averaged away; on one 4-camera rig the mean of four folds is not "
                    "well determined.",
        }

    # ---- articulation, on the four-camera fit
    positions = run_fit(cams, hand, None, "candidate", compute=compute)
    if positions is not None:
        wrist = hand["wrist"]
        hand_length = float(np.median(np.linalg.norm(positions[:, middle1] - wrist, axis=1)))
        local = wrist_local_frame(positions, wrist, knuckles, index1, pinky1)
        ours = articulation(local[:, tips], hand_length)
        ours.update(translation_only_jitter(positions, wrist, tips))
        ours["hand_length_wrist_to_middle1_mm"] = round(hand_length * 1000.0, 1)
        ours["seconds_to_fit"] = fit_seconds(name, "candidate", None)

        mamma, mamma_wrist, mapping = mamma_hand_positions(
            mapping_our_to_mamma[hand["subject"]], hand["side"], layout, joints_names)
        mamma_length = float(np.median(
            np.linalg.norm(mamma[:, middle1] - mamma_wrist, axis=1)))
        mamma_local = wrist_local_frame(mamma, mamma_wrist, knuckles, index1, pinky1)
        theirs = articulation(mamma_local[:, tips], mamma_length)
        theirs.update(translation_only_jitter(mamma, mamma_wrist, tips))
        theirs["hand_length_wrist_to_middle1_mm"] = round(mamma_length * 1000.0, 1)

        # Every control's amplitude, on the same four-camera fit. The gate card asks
        # for "articulation sd 0" on the frozen arms, and a control whose figure is
        # not in the report is not a control: this is where the frozen arms are shown
        # to be zero and the prior-dominated arm is shown to have the BEST jitter on
        # the page while being one of the worst hands on it.
        control_arms = arms_for(hand, chain, positions)
        control_arms.pop("candidate")
        prior_all = run_fit(cams, hand, None, "prior_dominated", compute=False)
        if prior_all is not None:
            control_arms["CONTROL_prior_dominated"] = prior_all
        control_figures = {}
        for arm, arm_positions in control_arms.items():
            arm_local = wrist_local_frame(arm_positions, wrist, knuckles, index1, pinky1)
            row = articulation(arm_local[:, tips], hand_length)
            row.update(translation_only_jitter(arm_positions, wrist, tips))
            row["rejected_by"] = REJECTED_BY[arm]
            control_figures[arm] = row

        entry["articulation"] = {
            "frame": "wrist-local, built from POSITIONS ONLY and identically on both "
                     "sides: origin at the wrist, first axis to the knuckle centroid, "
                     "third the palm normal (axis1 x (index1 - pinky1)), second "
                     "completing the right-handed set. Gauge-invariant: it does not "
                     "care that ours is MHR Euler and MAMMA's is SMPL-X axis-angle, "
                     "which is why the old 18-24 deg vs 5.6 deg comparison was void.",
            "ours": ours,
            "AGREEMENT_mamma": theirs,
            "THE_THRASH_FIGURE_fingertip_jitter_mm": ours["jitter_mm"],
            "thrash_figure_note":
                "wrist-local fingertip jitter, the parameterisation-invariant "
                "replacement for the void angle-sd comparison. THE SOLVER OPTIMISES "
                "THIS QUANTITY (`pose_smooth_weight` penalises wrist-relative joint "
                "acceleration), so it is a knob setting and not evidence: it rejects "
                "no control on this page. It is reported because rung 8 asks for the "
                "thrash figure, and it is read only against MAMMA's on its own "
                "reference string.",
            "CONTROLS": control_figures,
            "controls_note":
                "read the amplitude column: the two frozen arms and the prior-dominated "
                "arm sit at or near zero, which is the clause a constant cannot pass. "
                "Then read the jitter column: the prior-dominated arm is the SMOOTHEST "
                "hand on this page and one of the worst. That is the failure the rung's "
                "band was rewritten to exclude, and it is why jitter rejects nothing.",
            "WRAPPING_ANCHOR_retained_arrays": retained_reproduction(
                hand, knuckles, tips, index1, pinky1, middle1),
            "agreement_fingertips": fingertip_agreement(
                positions, wrist, mamma, mamma_wrist, knuckles, tips, index1, pinky1),
        }
    return entry


def retained_reproduction(hand: dict, knuckles: list[int], tips: list[int],
                          index1: int, pinky1: int, middle1: int) -> dict:
    """The same figures computed on the fit ALREADY IN `artifacts/handfit-arrays/`.

    The standing rule is *wrap the pipeline, never re-implement it*, and the way to
    show that this file wrapped rather than reinvented is to run its articulation
    code over the retained solution the increment-5 and increment-6 documents were
    written from, and check that the documented numbers come back. They do:
    48.8 mm amplitude, ~26 mm jitter, 50.6 mm with only wrist translation removed and
    8.83 mm of wrist-anchor jitter on subj0's left hand
    (`BATTLE1_INCREMENT5_ARTICULATION_VARIANCE.md`). This is not a score; it is the
    receipt.
    """
    positions, wrist = hand["positions"], hand["wrist"]
    length = float(np.median(np.linalg.norm(positions[:, middle1] - wrist, axis=1)))
    local = wrist_local_frame(positions, wrist, knuckles, index1, pinky1)
    row = articulation(local[:, tips], length)
    row.update(translation_only_jitter(positions, wrist, tips))
    row["hand_length_wrist_to_middle1_mm"] = round(length * 1000.0, 1)
    row["is_a_score"] = False
    row["what_it_is"] = (
        "the retained solution in artifacts/handfit-arrays/, scored by THIS file's "
        "articulation code. It reproduces the figures in "
        "BATTLE1_INCREMENT5_ARTICULATION_VARIANCE.md, which is the receipt that this "
        "instrument wrapped the existing protocol rather than reinventing it.")
    return row


def fingertip_agreement(ours: np.ndarray, our_wrist: np.ndarray, theirs: np.ndarray,
                        their_wrist: np.ndarray, knuckles: list[int], tips: list[int],
                        index1: int, pinky1: int) -> dict:
    """How far our fingertips sit from MAMMA's -- AGREEMENT, never accuracy.

    Anchored at the knuckle centroid and normalised by knuckle-to-tip span, because
    MHR's hand root is not SMPL-X's wrist: our wrist-to-knuckle segment is
    7.4-8.6 cm against MAMMA's 10.4-11.9 cm, a joint-definition difference that
    would otherwise be charged to the fingers. The residual after that removal is
    articulation disagreement.
    """
    def anchored(positions, wrist):
        local = wrist_local_frame(positions, wrist, knuckles, index1, pinky1)
        centroid = local[:, knuckles].mean(axis=1, keepdims=True)
        centred = local - centroid
        span = np.median(np.linalg.norm(centred[:, tips], axis=2))
        return centred, float(span)

    a, span_a = anchored(ours, our_wrist)
    b, span_b = anchored(theirs, their_wrist)
    scaled = a * (span_b / span_a)
    distance = np.linalg.norm(scaled[:, tips] - b[:, tips], axis=2) * 1000.0
    return {
        "median_mm": round(float(np.median(distance)), 1),
        "p95_mm": round(float(np.percentile(distance, 95)), 1),
        "our_knuckle_to_tip_span_mm": round(span_a * 1000.0, 1),
        "mamma_knuckle_to_tip_span_mm": round(span_b * 1000.0, 1),
        "is_a_score": False,
        "note": "agreement with another estimator on its own reference string, never "
                "accuracy: MAMMA is an instrument, there is no ground truth on disk "
                "(gt_joints is a byte-copy of pred_joints), and MAMMA's own two-person "
                "hand error on this footage is ~48 mm.",
    }


def convention_offsets(layout: dict, cams, mapping_our_to_mamma: dict) -> dict:
    """How much of the oracle's floor is joint convention rather than 2D error.

    Per MHR slot, the median 3D distance between our fitted joint and the SMPL-X
    joint it is mapped to, on the four-camera fit. An oracle that carries a 20 mm
    convention offset cannot reach zero however good MAMMA's hand is, and the
    reader is entitled to see which part of the floor is which.
    """
    rows: dict = {}
    for name in HANDS:
        cache = CACHE / f"{name}-candidate-all.npz"
        if not cache.exists():
            continue
        hand = load_hand(name)
        ours = np.load(cache)["positions"]
        mamma, _, mapping = mamma_hand_positions(
            mapping_our_to_mamma[hand["subject"]], hand["side"], layout, hand["joints"])
        rows[name] = {
            hand["joints"][slot]: round(float(np.median(
                np.linalg.norm(ours[:, slot] - mamma[:, slot], axis=1))) * 1000.0, 1)
            for slot in sorted(mapping)
        }
    return rows


def build(hands: list[str], folds: list[int], compute: bool) -> dict:
    cams = cameras()
    body = np.load(ARRAYS / "body-track.npz", allow_pickle=True)["positions"]
    mapping_our_to_mamma = mamma_index_for(body)
    reference = np.load(MA3D / "verts_joints_body_id-00.npz",
                        allow_pickle=True)["pred_joints"].astype(np.float64)
    layout = derive_smplx_hand_layout(reference)

    subjects = {name: measure_hand(name, cams, layout, mapping_our_to_mamma, folds, compute)
                for name in hands}
    subjects = {k: v for k, v in subjects.items() if v}

    means = [v["held_out_summary"]["mean_over_folds_mm"] for v in subjects.values()
             if "held_out_summary" in v]
    controls_rejected = sum(
        1 for v in subjects.values() for f in v["folds"].values()
        for arm, row in f["arms"].items() if arm.startswith("CONTROL") and row["rejected"])
    controls_total = sum(
        1 for v in subjects.values() for f in v["folds"].values()
        for arm in f["arms"] if arm.startswith("CONTROL"))
    not_rejected = sorted({
        f"{arm} ({name}, {camera})"
        for name, v in subjects.items() for camera, f in v["folds"].items()
        for arm, row in f["arms"].items()
        if arm.startswith("CONTROL") and not row["rejected"]})

    oracle = [f["arms"]["ORACLE_mamma_hand_joints"]["median_mm_at_the_subject"]
              for v in subjects.values() for f in v["folds"].values()]
    by_clause = {"held_out_camera": 0, "collapsed_amplitude": 0, "both": 0}
    for v in subjects.values():
        for f in v["folds"].values():
            for arm, row in f["arms"].items():
                if not arm.startswith("CONTROL") or not row["rejected"]:
                    continue
                a, b = row["rejected_by_held_out_camera"], row["rejected_by_collapsed_amplitude"]
                by_clause["both" if a and b else
                          "held_out_camera" if a else "collapsed_amplitude"] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "tools/hands/hand_fit_report.py",
        "regenerate": ".venv/bin/python tools/hands/hand_fit_report.py",
        "git_commit": git_commit(),
        "selection_rule": {
            "selector": "held-out camera. Fit on N-1 cameras, score reprojection on "
                        "the camera the fit never saw, rotate over all four.",
            "never_a_selector": [
                "MAMMA agreement -- MAMMA reports, it never selects (it is a "
                "research-licensed instrument that never ships)",
                "the temporal term -- `pose_smooth_weight` regularises the very "
                "acceleration this report calls jitter, so a jitter band is a knob "
                "setting and not evidence",
                "in-sample reprojection -- that is the data term",
            ],
            "written_before_the_code": True,
        },
        "protocol": {
            "leave_one_camera_out": "cross_view_weights is computed on the THREE "
                "training cameras only and expanded with a zero column for the held-out "
                "one, so no held-out pixel reaches the fit even through a sigma. All "
                "four cameras stay in the fit_hand_sequence call so _pixels_per_metre, "
                "and with it the position prior's strength, is identical across folds.",
            "test_set": "the retained cross-view gate `keep[:, held_out]` -- geometry "
                        "among cameras, not a function of any solver, and the identical "
                        "(frame, joint) set for every arm. Same denominator.",
            "wraps": "src/autoanim_gnm/hand_fit.py::fit_hand_sequence at its shipped "
                     "defaults. Nothing in this file re-implements the fit or the gate.",
            "solver_kwargs": {
                "smooth_weight": 2.0, "pose_smooth_weight": 0.25, "limit_weight": 50.0,
                "robust_scale_px": 8.0, "maximum_evaluations": 120,
                "cross_view_weights_threshold_px": GATE_THRESHOLD_PX,
                "note": "the SHIPPED defaults. Increment 6's arms C and D used "
                        "pose_smooth_weight 1.0 and 4.0 and were never the default; "
                        "this report scores the code as it stands, and choosing among "
                        "them is solver work that belongs behind this gate.",
            },
            "working_resolution_px": [WORKING_WIDTH, WORKING_HEIGHT],
            "pixels_to_millimetres": "per observation, depth / focal at the working "
                                     "resolution -- not a take-wide constant, because "
                                     "the four cameras stand at different ranges.",
            "statistics": {
                "moving_block_bootstrap_frames": BLOCK_FRAMES,
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "identical_draws_across_arms": True,
                "rejection_rule": "a control is rejected when P(candidate beats it) "
                                  ">= 0.95 on the shared draws",
            },
        },
        "inputs": {
            **{f"artifacts/handfit-arrays/{n}.npz": sha256_of(ARRAYS / f"{n}.npz")
               for n in HANDS + ("body-track",)},
            "camera_rig": sha256_of(RIG),
            "src/autoanim_gnm/hand_fit.py": sha256_of(ROOT / "src/autoanim_gnm/hand_fit.py"),
        },
        "subject_map": {"our_subject_to_mamma_body_id": mapping_our_to_mamma,
                        "derived_by": "tools/head/subject_map.py, from pelvis agreement"},
        "smplx_hand_layout_DERIVED": layout,
        "smplx_layout_note":
            "derived from the file on every run, not recalled. The tip permutation is "
            "the part that matters: the naive index reading (66=index, matching the "
            "25..39 chain order) is WRONG; optimal assignment on median distal-to-tip "
            "distance gives 66=thumb, 67=index, 68=middle, 69=ring, 70=pinky, "
            "consistently on both bodies and both sides.",
        "oracle_convention_offset_mm_per_slot": convention_offsets(
            layout, cams, mapping_our_to_mamma),
        "hands": subjects,
        "summary": {
            "held_out_mean_over_hands_mm": (round(float(np.mean(means)), 1) if means else None),
            "hands_measured": len(means),
            "controls_rejected": controls_rejected,
            "controls_scored": controls_total,
            "controls_NOT_rejected": not_rejected,
            "controls_rejected_by_clause": by_clause,
            "ORACLE_floor_mean_mm": (round(float(np.mean(oracle)), 1) if oracle else None),
            "ORACLE_floor_range_mm": ([round(float(np.min(oracle)), 1),
                                       round(float(np.max(oracle)), 1)] if oracle else None),
            "what_the_oracle_floor_means":
                "MAMMA's own hand joints, on the MHR chain's slots, scored against the "
                "same SOMA-77 held-out observations. It is NOT a competitor and the "
                "candidate beating it is not a claim of superiority: our fit minimises "
                "SOMA-77 pixels on three cameras and MAMMA never saw SOMA-77 at all. "
                "What it measures is the floor this protocol imposes -- SOMA-77's own 2D "
                "error plus the MHR-to-SMPL-X joint convention gap, which the per-slot "
                "table puts at 26-73 mm in 3D. Read it as: a held-out figure at or below "
                "this floor cannot be separated from the reference's own answer by this "
                "instrument, so the gate discriminates degenerate motion, not accuracy.",
            "how_each_control_is_rejected": REJECTED_BY,
            "note": "every control above is rejected by the held-out camera, by "
                    "fingertip amplitude, or by both. None is rejected by jitter -- the "
                    "solver regularises that quantity, so a control that only failed on "
                    "it would prove nothing. A control listed under "
                    "`controls_NOT_rejected` is a measured blindness of this protocol, "
                    "not a passing arm: report it as such.",
        },
        "first_candidate_fix_not_attempted_here":
            "`smooth_weight` multiplies a second difference in RADIANS against a data "
            "block in PIXELS; measured at 1.9 % of the objective. Reconciling the units "
            "is solver work and belongs behind this gate, not inside the step that "
            "builds it.",
        "arms_absent_from_this_report": {
            "CONTROL_prior_dominated, held-out folds": {
                "present": "its FOUR-CAMERA fit and therefore its amplitude, jitter and "
                           "roughness, on every hand -- see hands.*.articulation.CONTROLS",
                "absent": "its leave-one-camera-out reprojection, on every hand and fold",
                "why": "a prior-dominated refit takes ~54 min per hand per fold on this "
                       "machine against ~16-23 min for the candidate, and the grid was "
                       "stopped after the four-camera arms landed. Recorded as absent "
                       "rather than filled: a placeholder here would be a fabricated "
                       "figure for the one control the rung most needs.",
                "regenerate": ".venv/bin/python tools/hands/hand_fit_report.py "
                              "--fits-only  (cached, resumable; then --assemble)",
                "what_it_would_decide": "this arm drifts smoothly at LARGER amplitude "
                                        "than the candidate with jitter better than "
                                        "MAMMA's, so it passes both articulation clauses. "
                                        "Its held-out reprojection is the only figure "
                                        "that can reject it, and it is the missing one.",
            },
        },
        "blind_to": [
            "depth along the held-out camera's rays -- reprojection into one camera "
            "cannot score it, and that is the direction fingers are least determined in",
            "accuracy -- there is no ground truth on disk; this is view generalisation "
            "and, separately, agreement with MAMMA",
            "observations the retained gate removed, including in the held-out view "
            "(the ungated denominator is reported beside every gated one)",
            "contact precision",
            "anything beyond this one 150-frame take with two performers",
        ],
    }


def warm_the_cache(hands: list[str]) -> None:
    """Run the solves in the order that leaves the most useful partial report.

    A leave-one-camera-out fit of one hand takes 16-32 minutes on this machine and
    the prior-dominated refit takes longer still -- a strong second-difference prior
    makes the trust region crawl. So the order is chosen so that an interrupted grid
    still answers the rung's question:

      1. the four candidate folds -- the headline;
      2. the four-camera candidate fit -- amplitude, jitter and the thrash figure;
      3. the four-camera prior-dominated fit -- the control that must show that a
         smooth hand can be a wrong one;
      4. the prior-dominated held-out folds, best effort.

    Everything is cached per (hand, arm, fold), so this is resumable and re-running
    it costs nothing for work already done.
    """
    cams = cameras()
    for name in hands:
        hand = load_hand(name)
        hand["name"] = name
        plan = ([("candidate", index) for index in range(len(CAMERAS))]
                + [("candidate", None), ("prior_dominated", None)]
                + [("prior_dominated", index) for index in range(len(CAMERAS))])
        for arm, held_out in plan:
            where = "all" if held_out is None else CAMERAS[held_out]
            if cache_path(name, arm, held_out).exists():
                print(f"{name} {arm} {where}: cached", flush=True)
                continue
            started = time.time()
            run_fit(cams, hand, held_out, arm)
            print(f"{name} {arm} {where}: {time.time() - started:.0f} s", flush=True)


def _plain(value):
    """numpy scalars are not JSON; nothing in this report should reach here as one."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not serialisable: {type(value)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", action="append", choices=list(HANDS))
    parser.add_argument("--fold", action="append", choices=list(CAMERAS))
    parser.add_argument("--assemble", action="store_true",
                        help="write the report from cached fits only; run no solves")
    parser.add_argument("--fits-only", action="store_true",
                        help="run the solve grid in priority order and write nothing")
    args = parser.parse_args()

    hands = args.hand or list(HANDS)
    folds = [CAMERAS.index(c) for c in (args.fold or list(CAMERAS))]
    if args.fits_only:
        warm_the_cache(hands)
        return 0
    report = build(hands, folds, compute=not args.assemble)
    if args.assemble or hands == list(HANDS):
        OUT.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=False, default=_plain) + "\n")
        print(f"wrote {REPORT.relative_to(ROOT)}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
