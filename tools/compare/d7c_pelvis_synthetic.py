#!/usr/bin/env python3
"""D7c's SELECTOR, S: which rig-rest pelvis construction ships, decided on synthetic truth.

THIS FILE SELECTS. It runs BEFORE `src/` moves, and if it stops, `src/` is not touched at all.
The take cannot decide this: the two candidates differ in exactly what a real take is blind to
(see "the trade" below), and every take-level figure in this step measures the same delivered
consequence for both.

THE TRUTH. The D3 gate's six exact-skeleton bodies -- a perturbed per-performer rest, forward
kinematicked with the delivered subject-0 motion. The pelvis's true world rotation on such a
body is known exactly, because forward kinematics wrote it. The SOMA-posed fixture
`d7_pelvis_synthetic` uses cannot referee this step: its pelvis IS the SOMA convention whose
removal is the change, so it would score the candidate worse by construction.

  BLIND SPOT, stated before any number: the truth MOTION is D9b's delivered pelvis motion,
  produced by the very fit being replaced. S therefore ranks estimators under the rig
  convention on a motion that inherits the shipped fit's smoothness, and says nothing about
  how much real pelvic motion a performer has.

THE OBSERVATION. I7/I8's own heavy-tail frame-correlated pixel noise at
`thorax_window_sweep.NOISE_SIGMA_PX` = 3.20 px, injected in PIXELS and recovered through the
REAL `triangulate_point` on the delivery's own four-camera rig -- `d7_pelvis_synthetic.observe`
itself, with its truth array replaced by the D3 bodies' landmarks. Never MAMMA's residual. One
seeded draw per body, frozen, and EVERY ARM READS THE SAME DRAW.

  LIMITATION, stated: `observe` uses every camera with positive depth and does NOT reproduce
  the take's A-C-only support on performer 1's lying stretch. S measures estimator behaviour
  under the detector's noise, not under the take's occlusion pattern.

THE TRADE S DECIDES, and why it is a genuine choice and not a construction argument. On
noiseless input both rig modes are EXACT (the gate measures 1.5-2.4e-7 m of residual), so
neither is "more correct". They differ in what they do with a WRONG observation:

  (b) `D_rig_rest_hipline` makes the observed hip line the exact primary axis. A hip line
      rotated about its own midpoint -- which keeps its LENGTH and therefore passes the
      guard's ceiling by that rule's own docstring -- is followed in full.
  (a) `E_rig_rest_kabsch` lets the 197 mm spine lever pull against the same rotation inside
      one un-centred SVD, so it costs less there (Astra's counterexample: 10 deg / 18 mm at
      the leg roots for (b) against 3.6 deg / 6.5 mm for (a)) -- and pays for it by letting
      a bad Spine1 move the hip line, which (b) never does.

THE CONTROLS, and what each is for:

  * C-on-SOMA -- today's shipped code, mode `C_kabsch_pelvis`, reading the raw Spine1 as the
    delivery reads it. The constant the winner must beat on (i).
  * the FROZEN-PITCH HIP-LINE FOLLOWER -- a control LAW with no constant: the observed hip
    line as the exact primary axis (as (b)), the up axis world +Y orthogonalised against it.
    It never reads Spine1, so its pitch about the hip line is frozen at zero relative to
    gravity on every frame, and it places the leg roots exactly where (b) does. It is the
    control leg-root placement cannot discriminate, and if the fixture cannot separate it
    from the winner the fixture has not shown the discrimination the card claims -- a STOP,
    never a pass.
  * world-vertical (D7's control) and thorax-as-pelvis (the pre-D7 code), reported.
  * (b) UNGUARDED -- the ablation. Exempt from the guard by definition; nothing in the merge
    predicate reads its two report quantities.

THE GUARD'S OWN EXPERIMENT, two SEPARATE trials that are never combined:

  G1 MISSING-ONLY is an EQUIVALENCE, not a superiority trial. With Spine1 absent, guarded and
     unguarded take the same `np.interp` recovery, so the interpolated arrays must be
     bit-identical; the recovery error is reported, not banded.
  G2 FINITE-ONLY is where the guard must earn its place: a finite, DIRECTION-MOVING corruption
     (a donor frame's Spine1 from at least 10 frames away, read from an immutable
     pre-corruption array so replacement cannot cascade). A radial rescaling is not used --
     it is invisible to (b)'s normalised secondary axis. The stop is here, and it requires
     BOTH (i) on the corrupted frames AND (ii) on the transition pairs.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_pelvis_synthetic.py \\
        --out artifacts/compare/d7c-pelvis-rest/selector.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "tools/swap-harness", "scripts",
                  "workers/commercial_multiview"):
    if str(ROOT / _relative) not in sys.path:
        sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

import autoanim_gnm.commercial_multiview as cm  # noqa: E402
from autoanim_gnm.body import (  # noqa: E402
    DETAILED_HUMANOID, forward_kinematics_positions, _quaternion_multiply)
from autoanim_gnm.commercial_multiview import load_camera_rig  # noqa: E402
from autoanim_gnm.soma_motion import SOMASKEL77_NAMES  # noqa: E402
import d3_skeleton_gate as d3  # noqa: E402
import d7_pelvis_synthetic as d7s  # noqa: E402
import d7c_pelvis_estimators as est  # noqa: E402
import retarget_cost as rc  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d7c-pelvis-rest"
RIG = ROOT / "artifacts/commercial-multiview-soma77/camera-rig.json"
N = {name: index for index, name in enumerate(SOMASKEL77_NAMES)}

# ------------------------------------------------------------ every definition, fixed first
BENT_TERCILE_FRAMES = 50            # "the 50 frames with the largest truth trunk tilt"
TIE_DEG = 0.1                       # (i) and (ii)
TIE_MM = 0.1                        # (iii)
FOLLOWER_RATIO = 2.0                # the frozen-pitch follower must read >= 2x the winner's
FOLLOWER_FLOOR_DEG = 2.0            # ... and >= 2 deg on the bent tercile, on EVERY body
WORLD_VERTICAL_LIMITATION_DEG = 2.0
G2_CORRUPTED_FRAMES = 30            # 20 % of 150
G2_MAX_RUN = 9
G2_DONOR_MINIMUM_GAP = 10
TRANSITION_HALO = 4                 # "the pairs whose both endpoints lie in the frames +-4"
# performer 1's own 29-frame demote pattern, frozen in the card and reproduced by
# `d7c_pelvis_rest_gate.py` on the shipped build's own bytes.
PERFORMER_1_MISSING = (24, 25, 28, 29, 32, 33, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                       65, 68, 70, 78, 79, 80, 81, 140, 141, 144, 145, 147, 148, 149)

ARMS = ("b_hipline_guarded", "a_kabsch_guarded", "b_hipline_unguarded",
        "C_on_SOMA", "world_vertical", "thorax_as_pelvis", "frozen_pitch_follower")
RIG_ARMS = {"b_hipline_guarded": ("D_rig_rest_hipline", True),
            "a_kabsch_guarded": ("E_rig_rest_kabsch", True),
            "b_hipline_unguarded": ("D_rig_rest_hipline", False)}


def world_rotations(rotations: np.ndarray, skeleton) -> np.ndarray:
    out = np.zeros_like(rotations)
    for index, joint in enumerate(skeleton.joints):
        if joint.parent == -1:
            out[:, index] = rotations[:, index]
        else:
            out[:, index] = _quaternion_multiply(out[:, joint.parent], rotations[:, index])
    return out


def geodesic_deg(a: Rotation, b: Rotation) -> np.ndarray:
    return np.degrees(np.linalg.norm((a * b.inv()).as_rotvec(), axis=1))


# ----------------------------------------------------------------------------- the fixture
def body(seed: int) -> dict:
    """One D3 exact-skeleton body: its truth pose, its truth pelvis, its landmarks."""
    donor = d3.load_track(d3.DELIVERED, 0)
    rotations = np.asarray(donor.local_rotations_xyzw, np.float64)
    roots = np.asarray(donor.root_translation_m, np.float64)
    rest_array, factors = d3.perturbed_rest(np.random.default_rng(seed))
    skeleton = DETAILED_HUMANOID.with_rest_translations(rest_array)
    index = skeleton.index
    truth = forward_kinematics_positions(roots, rotations,
                                         skeleton=skeleton).astype(np.float64)
    lowest = float(min(truth[:, index("LeftToes"), 1].min(),
                       truth[:, index("RightToes"), 1].min()))
    truth = truth.copy()
    truth[..., 1] -= lowest
    truth_world = world_rotations(rotations, skeleton)
    landmarks = rc.landmarks_from_fk(truth, skeleton)
    rest = {n: rest_array[index(n)] for n in skeleton.names}
    return {"seed": seed, "skeleton": skeleton, "rest": rest, "factors": factors,
            "truth_landmarks": landmarks,
            "truth_spine": truth[:, index("Spine")],
            "truth_pelvis": Rotation.from_quat(truth_world[:, index("Hips")]),
            "frames": len(truth)}


def observe_body(cameras, data: dict, rng, sigma_scale: float = 1.0) -> dict:
    """`d7_pelvis_synthetic.observe`, its truth replaced by this body's landmarks.

    The SOMA-77-shaped array is a CARRIER, not a claim: `observe` iterates `d7s.NOISED` by
    SOMA index, so the body's five relevant points are written into those slots, pushed
    through the real cameras and the real `triangulate_point`, and read back. Nothing about
    SOMA-77's rest or its convention enters -- only the indices its noise model iterates.
    Every arm reads the SAME returned array; the joints noised are every joint ANY arm
    reads, so no arm enjoys a cleaner input than another.
    """
    landmarks = data["truth_landmarks"]
    frames = data["frames"]
    carrier = np.zeros((frames, len(SOMASKEL77_NAMES), 3))
    z_up = rc.Z_UP_FROM_Y_UP(landmarks)
    spine_z_up = rc.Z_UP_FROM_Y_UP(data["truth_spine"])
    carrier[:, N["Hips"]] = z_up[:, cm.JOINT_INDEX["root"]]
    carrier[:, N["Spine1"]] = spine_z_up
    carrier[:, N["Spine2"]] = spine_z_up
    carrier[:, N["Neck2"]] = z_up[:, cm.JOINT_INDEX["neck"]]
    carrier[:, N["LeftLeg"]] = z_up[:, cm.JOINT_INDEX["left_hip"]]
    carrier[:, N["RightLeg"]] = z_up[:, cm.JOINT_INDEX["right_hip"]]
    noisy = d7s.observe(cameras, carrier, rng, smooth_spine=False,
                        sigma_scale=sigma_scale)
    points = np.array(landmarks, copy=True)
    points[:, cm.JOINT_INDEX["root"]] = rc.Y_UP_FROM_Z_UP(noisy[:, N["Hips"]])
    points[:, cm.JOINT_INDEX["neck"]] = rc.Y_UP_FROM_Z_UP(noisy[:, N["Neck2"]])
    points[:, cm.JOINT_INDEX["left_hip"]] = rc.Y_UP_FROM_Z_UP(noisy[:, N["LeftLeg"]])
    points[:, cm.JOINT_INDEX["right_hip"]] = rc.Y_UP_FROM_Z_UP(noisy[:, N["RightLeg"]])
    spine = rc.Y_UP_FROM_Z_UP(noisy[:, N["Spine1"]])
    return {"points": points, "spine": spine}


# -------------------------------------------------------------------------------- the arms
def thorax_as_pelvis_frames(points: np.ndarray) -> np.ndarray:
    """The converter's pre-D7 line, verbatim: the TRUNK line is the pelvis's up axis."""
    points = np.asarray(points, dtype=np.float64)
    left = points[:, cm.JOINT_INDEX["left_hip"]]
    right = points[:, cm.JOINT_INDEX["right_hip"]]
    neck = points[:, cm.JOINT_INDEX["neck"]]
    hip_mid = 0.5 * (left + right)
    quaternions = np.zeros((len(points), 4))
    for frame in range(len(points)):
        quaternions[frame] = cm._frame_alignment(
            (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
            neck[frame] - hip_mid[frame], left[frame] - right[frame])
    for frame in range(1, len(points)):
        if float(np.dot(quaternions[frame], quaternions[frame - 1])) < 0.0:
            quaternions[frame] *= -1.0
    return quaternions


def run_arm(arm: str, points: np.ndarray, spine: np.ndarray, rest: dict) -> np.ndarray:
    if arm in RIG_ARMS:
        mode, guard = RIG_ARMS[arm]
        quaternions, _report = est.rig_rest_pelvis_frames(
            points, spine, rest, mode=mode, guard=guard)
        if quaternions is None:
            raise SystemExit(f"{arm} fell back to the torso frame on this body")
        return quaternions
    if arm == "C_on_SOMA":
        # the SHIPPED function, on the raw Spine1, exactly as today's delivery reads it
        quaternions, _report = cm._pelvis_world_frames(points, spine,
                                                       mode="C_kabsch_pelvis")
        return quaternions
    if arm == "world_vertical":
        return est.world_vertical_frames(points)
    if arm == "thorax_as_pelvis":
        return thorax_as_pelvis_frames(points)
    if arm == "frozen_pitch_follower":
        return est.frozen_pitch_hipline_frames(points)
    raise SystemExit(f"unknown arm {arm!r}")


def root_before_projection(quaternions: np.ndarray, hip_mid: np.ndarray,
                           rest: dict) -> np.ndarray:
    """The converter's own root line, `pelvis - rest[Hips] - R . mid` (:2882), and nothing
    else. No projection, no hoist: metric (iii) is defined before the ground contact."""
    mid = 0.5 * (np.asarray(rest["LeftUpperLeg"], np.float64)
                 + np.asarray(rest["RightUpperLeg"], np.float64))
    rotated = Rotation.from_quat(quaternions).apply(
        np.broadcast_to(mid, (len(quaternions), 3)))
    return hip_mid - np.asarray(rest["Hips"], np.float64) - rotated


# -------------------------------------------------------------------------- the statistics
def metrics(estimated: np.ndarray, truth: Rotation, hip_mid_obs: np.ndarray,
            hip_mid_truth: np.ndarray, rest: dict, frame_mask: np.ndarray,
            pair_mask: np.ndarray) -> dict:
    """(i) orientation, (ii) step, (iii) root step -- on ONE frame and ONE pair population.

    The pair population is built on the FULL sequence and masked by BOTH endpoints; a
    filtered array is never differenced.
    """
    rotation = Rotation.from_quat(estimated)
    orientation = geodesic_deg(rotation, truth)
    step_est = rotation[1:] * rotation[:-1].inv()
    step_truth = truth[1:] * truth[:-1].inv()
    step = geodesic_deg(step_est, step_truth)
    root_est = root_before_projection(estimated, hip_mid_obs, rest)
    root_truth = root_before_projection(truth.as_quat(), hip_mid_truth, rest)
    root_step = 1e3 * np.linalg.norm(np.diff(root_est, axis=0)
                                     - np.diff(root_truth, axis=0), axis=1)
    # the arm-specific part of (iii): the observation's own hip-midpoint noise is common to
    # every arm, so this companion row attributes what the ROTATION contributes.
    mid = 0.5 * (np.asarray(rest["LeftUpperLeg"], np.float64)
                 + np.asarray(rest["RightUpperLeg"], np.float64))
    compensation = (Rotation.from_quat(estimated).apply(np.broadcast_to(mid, (len(estimated), 3)))
                    - truth.apply(np.broadcast_to(mid, (len(estimated), 3))))
    compensation_step = 1e3 * np.linalg.norm(np.diff(compensation, axis=0), axis=1)
    return {
        "i_orientation_deg": float(np.median(orientation[frame_mask])),
        "ii_step_deg": float(np.median(step[pair_mask])),
        "iii_root_step_mm": float(np.median(root_step[pair_mask])),
        "iii_rotational_compensation_step_mm": float(np.median(compensation_step[pair_mask])),
        "n_frames": int(frame_mask.sum()), "n_pairs": int(pair_mask.sum()),
    }


def populations(data: dict) -> dict:
    """Whole take, and the bent tercile: the 50 frames with the largest TRUTH trunk tilt.

    Pairs `(t-1, t)` are formed on the FULL frame sequence and a pair belongs to a
    population iff BOTH endpoints do.
    """
    landmarks = data["truth_landmarks"]
    hip_mid = 0.5 * (landmarks[:, cm.JOINT_INDEX["left_hip"]]
                     + landmarks[:, cm.JOINT_INDEX["right_hip"]])
    trunk = landmarks[:, cm.JOINT_INDEX["neck"]] - hip_mid
    trunk = trunk / np.linalg.norm(trunk, axis=1)[:, None]
    tilt = np.degrees(np.arccos(np.clip(trunk[:, 1], -1.0, 1.0)))
    order = np.argsort(-tilt)[:BENT_TERCILE_FRAMES]
    bent = np.zeros(len(tilt), bool)
    bent[order] = True
    whole = np.ones(len(tilt), bool)
    return {
        "whole_take": {"frames": whole, "pairs": whole[1:] & whole[:-1]},
        "bent_tercile": {"frames": bent, "pairs": bent[1:] & bent[:-1]},
        "truth_trunk_tilt_deg": {
            "whole_median": round(float(np.median(tilt)), 3),
            "whole_max": round(float(tilt.max()), 3),
            "bent_median": round(float(np.median(tilt[bent])), 3),
            "bent_min": round(float(tilt[bent].min()), 3),
            "bent_max": round(float(tilt[bent].max()), 3)},
    }


def score_body(data: dict, observation: dict, arms: tuple[str, ...]) -> dict:
    rest = data["rest"]
    points, spine = observation["points"], observation["spine"]
    hip_mid_obs = 0.5 * (points[:, cm.JOINT_INDEX["left_hip"]]
                         + points[:, cm.JOINT_INDEX["right_hip"]])
    landmarks = data["truth_landmarks"]
    hip_mid_truth = 0.5 * (landmarks[:, cm.JOINT_INDEX["left_hip"]]
                           + landmarks[:, cm.JOINT_INDEX["right_hip"]])
    pop = populations(data)
    row: dict = {"truth_trunk_tilt_deg": pop["truth_trunk_tilt_deg"], "arms": {}}
    for arm in arms:
        quaternions = run_arm(arm, points, spine, rest)
        row["arms"][arm] = {
            name: metrics(quaternions, data["truth_pelvis"], hip_mid_obs, hip_mid_truth,
                          rest, pop[name]["frames"], pop[name]["pairs"])
            for name in ("whole_take", "bent_tercile")}
    return row


# ------------------------------------------------------------------------- the G2 placement
def g2_runs(rng: np.random.Generator, frames: int) -> list[tuple[int, int]]:
    """Astra's own executable placement law, adopted verbatim in the card.

    Draw a length uniformly from `1..min(9, remaining)` AMONG the lengths that have at least
    one legal placement (no overlap, no adjacency with a placed run), then a start uniformly
    among that length's legal starts. Repeat until exactly `G2_CORRUPTED_FRAMES` are covered.
    """
    covered = np.zeros(frames, bool)
    blocked = np.zeros(frames, bool)        # covered or adjacent to a covered frame
    placed: list[tuple[int, int]] = []
    while int(covered.sum()) < G2_CORRUPTED_FRAMES:
        remaining = G2_CORRUPTED_FRAMES - int(covered.sum())
        legal: dict[int, list[int]] = {}
        for length in range(1, min(G2_MAX_RUN, remaining) + 1):
            starts = [s for s in range(frames - length + 1)
                      if not blocked[s:s + length].any()]
            if starts:
                legal[length] = starts
        if not legal:
            raise SystemExit("G2: no legal placement remains; the law cannot reach 30 frames")
        lengths = sorted(legal)
        length = int(lengths[rng.integers(len(lengths))])
        starts = legal[length]
        start = int(starts[rng.integers(len(starts))])
        covered[start:start + length] = True
        blocked[max(0, start - 1):min(frames, start + length + 1)] = True
        placed.append((start, length))
    return placed


def g2_corrupt(rng: np.random.Generator, spine: np.ndarray,
               runs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Replace Spine1 on each corrupted frame by a DONOR frame's value.

    Every donor is read from `spine`, the IMMUTABLE pre-corruption array (the frozen
    `observe` output of this draw), so a replacement can never cascade into a later donor.
    The donor is at least `G2_DONOR_MINIMUM_GAP` frames away: a finite, DIRECTION-MOVING
    corruption. A radial rescaling is NOT used -- it is invisible to (b)'s normalised
    secondary axis and would test nothing.
    """
    frames = len(spine)
    corrupted = np.array(spine, copy=True)
    mask = np.zeros(frames, bool)
    for start, length in runs:
        for frame in range(start, start + length):
            choices = np.flatnonzero(np.abs(np.arange(frames) - frame)
                                     >= G2_DONOR_MINIMUM_GAP)
            donor = int(choices[rng.integers(len(choices))])
            corrupted[frame] = spine[donor]          # from the IMMUTABLE array
            mask[frame] = True
    return corrupted, mask


def halo_pairs(mask: np.ndarray) -> np.ndarray:
    """Transition pairs: `(t-1, t)` with BOTH endpoints inside the corrupted frames +- 4."""
    frames = len(mask)
    halo = np.zeros(frames, bool)
    for frame in np.flatnonzero(mask):
        halo[max(0, frame - TRANSITION_HALO):min(frames, frame + TRANSITION_HALO + 1)] = True
    return halo[1:] & halo[:-1]


# ---------------------------------------------------------------------------- the verdicts
def better(a: float, b: float, tie: float) -> str:
    """`a` against `b`: 'better', 'tied' or 'worse'. Lower is better on every metric here."""
    if abs(a - b) <= tie:
        return "tied"
    return "better" if a < b else "worse"


def aggregate(per_body: dict, arm: str, population: str, key: str) -> float:
    return float(np.median([per_body[seed]["arms"][arm][population][key]
                            for seed in per_body]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=OUT_DIR / "selector.json")
    parser.add_argument("--sigma-scale", type=float, default=1.0,
                        help="REPORT ONLY, and never a selection. 1.0 is the card's own "
                             "pre-registration (`d7_pelvis_synthetic.observe` at I7/I8's "
                             "NOISE_SIGMA_PX = 3.20 px) and is the only value whose verdict "
                             "is S's verdict. Any other value writes a SENSITIVITY file "
                             "and refuses to overwrite `selector.json`: the fixture's noise "
                             "amplitude was pre-registered by the card and reviewed four "
                             "times against the 2x follower band, so re-choosing it after "
                             "seeing the verdict would be selecting on a knob. The sweep "
                             "exists so the coordinator and Astra can see how the a/b "
                             "selection and the follower ratio move with it.")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if args.sigma_scale != 1.0:
        out = out.parent / f"selector-sensitivity-sigma{args.sigma_scale:g}.json"
    rig = load_camera_rig(RIG)
    cameras = tuple(c.scaled(d7s.WORKING_WIDTH, d7s.WORKING_HEIGHT) for c in rig)

    report: dict = {
        "title": "D7c selector S -- which rig-rest pelvis construction ships",
        "truth": ("the D3 gate's six exact-skeleton bodies; the pelvis's true world "
                  "rotation is the one forward kinematics wrote"),
        "truth_motion_blind_spot": (
            "the truth MOTION is D9b's delivered pelvis motion, produced by the fit being "
            "replaced: S ranks estimators under the rig convention and says nothing about "
            "how much real pelvic motion a performer has"),
        "noise": {"model": "I7/I8 heavy-tail frame-correlated, in PIXELS through the real "
                           "`triangulate_point` on the delivery's own four-camera rig",
                  "sigma_px": d7s.sweep.NOISE_SIGMA_PX,
                  "limitation": ("`observe` uses every positive-depth camera and does NOT "
                                 "reproduce the take's A-C-only support on performer 1's "
                                 "lying stretch")},
        "definitions": {
            "i_orientation_deg": "geodesic angle of R_est . R_truth^-1, pooled median",
            "ii_step_deg": ("geodesic angle of (R_t R_{t-1}^-1)_est . "
                            "((R_t R_{t-1}^-1)_truth)^-1, pooled median over frame PAIRS"),
            "iii_root_step_mm": ("|d root_est - d root_truth| in mm BEFORE the projection, "
                                 "from the converter's own root line; the observation's "
                                 "hip-midpoint noise is common to every arm and the "
                                 "rotational-compensation companion row attributes it"),
            "populations": ("whole take, and the bent tercile = the 50 frames with the "
                            "largest TRUTH trunk tilt from world vertical, per body"),
            "pairs": ("pairs are formed on the FULL frame sequence and belong to a "
                      "population iff BOTH endpoints do"),
            "aggregation": "the median over the six per-body medians, all six reported",
            "ties": f"within {TIE_DEG} deg (i, ii) or {TIE_MM} mm (iii)",
        },
        "arms": list(ARMS),
        "sigma_scale": args.sigma_scale,
        "is_the_pre_registered_fixture": args.sigma_scale == 1.0,
        "sensitivity_note": (None if args.sigma_scale == 1.0 else
                             "SENSITIVITY ONLY. The card pre-registers the fixture at "
                             "I7/I8's own amplitude; this file selects nothing and its "
                             "verdict is not S's verdict."),
        "bodies": {},
    }

    per_body: dict = {}
    frozen: dict = {}
    for seed in d3.SEEDS:
        data = body(seed)
        rng = np.random.default_rng(seed)          # one frozen draw per body
        observation = observe_body(cameras, data, rng, args.sigma_scale)
        frozen[seed] = {"data": data, "observation": observation}
        per_body[str(seed)] = score_body(data, observation, ARMS)
        row = per_body[str(seed)]
        print(f"seed {seed} tilt {row['truth_trunk_tilt_deg']}")
        for arm in ARMS:
            whole = row["arms"][arm]["whole_take"]
            bent = row["arms"][arm]["bent_tercile"]
            print(f"  {arm:24s} whole i {whole['i_orientation_deg']:7.4f} ii "
                  f"{whole['ii_step_deg']:7.4f} iii {whole['iii_root_step_mm']:7.4f} | "
                  f"bent i {bent['i_orientation_deg']:7.4f} ii {bent['ii_step_deg']:7.4f} "
                  f"iii {bent['iii_root_step_mm']:7.4f}")
    report["bodies"] = per_body
    report["fixture_attribution"] = fixture_attribution(frozen)
    print("\nfixture attribution:",
          json.dumps(report["fixture_attribution"]["summary"], indent=1))

    # ---------------------------------------------------------------- the selection verdict
    aggregated = {
        arm: {population: {key: round(aggregate(per_body, arm, population, key), 5)
                           for key in ("i_orientation_deg", "ii_step_deg",
                                       "iii_root_step_mm",
                                       "iii_rotational_compensation_step_mm")}
              for population in ("whole_take", "bent_tercile")}
        for arm in ARMS}
    report["aggregated_median_of_six"] = aggregated

    comparison: dict = {}
    for population in ("whole_take", "bent_tercile"):
        for key, tie in (("i_orientation_deg", TIE_DEG), ("ii_step_deg", TIE_DEG),
                         ("iii_root_step_mm", TIE_MM)):
            comparison[f"b_vs_a__{population}__{key}"] = better(
                aggregated["b_hipline_guarded"][population][key],
                aggregated["a_kabsch_guarded"][population][key], tie)
    report["b_vs_a"] = comparison
    verdicts = list(comparison.values())
    if "worse" not in verdicts and "better" in verdicts:
        winner, loser = "b_hipline_guarded", "a_kabsch_guarded"
        winner_mode = "D_rig_rest_hipline"
    elif all(v in ("worse", "tied") for v in verdicts) and "worse" in verdicts:
        winner, loser = "a_kabsch_guarded", "b_hipline_guarded"
        winner_mode = "E_rig_rest_kabsch"
    else:
        report["S_verdict"] = "SPLIT"
        report["stop"] = ("neither rig mode is better-or-tied on all three metrics on both "
                          "populations with a strict win somewhere; the card says the step "
                          "STOPS for the coordinator")
        return finish(report, out, stop=True)
    report["winner"] = {"arm": winner, "mode": winner_mode, "loser": loser}

    # ... strictly better than C-on-SOMA on (i) on both populations, better-or-tied on (ii), (iii)
    against_c: dict = {}
    for population in ("whole_take", "bent_tercile"):
        for key, tie in (("i_orientation_deg", TIE_DEG), ("ii_step_deg", TIE_DEG),
                         ("iii_root_step_mm", TIE_MM)):
            against_c[f"{population}__{key}"] = better(
                aggregated[winner][population][key],
                aggregated["C_on_SOMA"][population][key], tie)
    report["winner_vs_C_on_SOMA"] = against_c
    c_ok = (all(against_c[f"{p}__i_orientation_deg"] == "better"
                for p in ("whole_take", "bent_tercile"))
            and all(against_c[f"{p}__{k}"] in ("better", "tied")
                    for p in ("whole_take", "bent_tercile")
                    for k in ("ii_step_deg", "iii_root_step_mm")))
    report["winner_beats_the_constant_it_removes"] = c_ok
    if not c_ok:
        report["S_verdict"] = "STOP"
        report["stop"] = ("the winner does not beat C-on-SOMA on (i) on both populations "
                          "while holding (ii) and (iii)")
        return finish(report, out, stop=True)

    # ... the frozen-pitch follower must read (i) >= 2x the winner's AND >= 2 deg on the
    # bent tercile, on EVERY body. Otherwise the fixture has not shown the discrimination.
    follower: dict = {}
    for seed in per_body:
        winner_i = per_body[seed]["arms"][winner]["bent_tercile"]["i_orientation_deg"]
        control_i = per_body[seed]["arms"]["frozen_pitch_follower"]["bent_tercile"][
            "i_orientation_deg"]
        follower[seed] = {
            "winner_i_deg": round(winner_i, 5), "follower_i_deg": round(control_i, 5),
            "ratio": round(control_i / winner_i, 3) if winner_i > 0 else None,
            "ratio_at_least_2x": bool(control_i >= FOLLOWER_RATIO * winner_i),
            "at_least_2_deg": bool(control_i >= FOLLOWER_FLOOR_DEG)}
    report["frozen_pitch_follower_bent_tercile"] = follower
    follower_ok = all(r["ratio_at_least_2x"] and r["at_least_2_deg"]
                      for r in follower.values())
    report["follower_discriminated_on_every_body"] = follower_ok
    if not follower_ok:
        report["S_verdict"] = "STOP"
        report["stop"] = ("the frozen-pitch hip-line follower is not separated from the "
                          "winner on every body; the fixture has not shown the "
                          "discrimination the card claims -- a STOP, never a pass")
        return finish(report, out, stop=True)

    # ... the world-vertical control against the truth's own tilt range
    report["world_vertical_vs_truth_tilt"] = {
        "truth_bent_tilt_median_deg": round(float(np.median(
            [per_body[s]["truth_trunk_tilt_deg"]["bent_median"] for s in per_body])), 3),
        "world_vertical_i_bent_deg": aggregated["world_vertical"]["bent_tercile"][
            "i_orientation_deg"],
        "winner_i_bent_deg": aggregated[winner]["bent_tercile"]["i_orientation_deg"],
        "stated_limitation_if_within_2_deg_of_the_tilt": bool(abs(
            aggregated["world_vertical"]["bent_tercile"]["i_orientation_deg"]
            - float(np.median([per_body[s]["truth_trunk_tilt_deg"]["bent_median"]
                               for s in per_body]))) < WORLD_VERTICAL_LIMITATION_DEG),
    }

    # ------------------------------------------------------------------ G1 and G2
    report["G1_missing_only"] = g1(frozen, winner)
    report["G2_finite_only"] = g2(frozen, winner)
    if not report["G2_finite_only"]["guard_wins_both"]:
        report["S_verdict"] = "STOP"
        report["stop"] = ("G2: the guarded winner does not read BOTH (i) on the corrupted "
                          "frames AND (ii) on the transition pairs below the unguarded "
                          "winner. A failure of either does not authorise shipping the "
                          "unguarded winner.")
        return finish(report, out, stop=True)

    report["S_verdict"] = "PROCEED"
    report["ships"] = winner_mode
    return finish(report, out, stop=False)


# ----------------------------------------------------- what the fixture itself contributes
# D7's own report records the real take's measured spread of the pelvis lever beside the
# synthetic's (`artifacts/compare/d7-pelvis-frame/synthetic.json`,
# `noise_calibration_vs_the_real_take`). It is the only measured figure available and it is
# quoted here verbatim, frozen, so this block can be read without re-deriving it.
D7_REAL_TAKE_MIDHIPS_TO_SPINE1_SD_MM = {"subject_00": 6.61, "subject_01": 11.10}


def fixture_attribution(frozen: dict) -> dict:
    """How much of every arm's error is the FIXTURE's noise and how much is the arm.

    Two rows, and neither selects anything:

      * the NOISELESS deficit -- every arm on the truth landmarks with no observation noise
        at all. It is each arm's own structural error, and for the frozen-pitch follower it
        is the quantity the follower clause is about: the true pelvis pitch about the hip
        line, relative to gravity.
      * the fixture's own lever spread against the real take's measured one. THE COMPARISON
        IS CONFOUNDED AND THE CONFOUND IS STATED: the D3 rig is RIGID, so the synthetic
        lever's spread is pure observation noise, while the take's mixes observation noise
        with the performer's real lever variation. Matching them would therefore UNDER-noise
        the fixture in the candidate's favour, which is one reason this file does not pick a
        calibration.
    """
    noiseless: dict = {}
    for seed, held in frozen.items():
        data = held["data"]
        pop = populations(data)
        bent = pop["bent_tercile"]["frames"]
        row = {}
        for arm in ARMS:
            quaternions = run_arm(arm, data["truth_landmarks"], data["truth_spine"],
                                  data["rest"])
            error = geodesic_deg(Rotation.from_quat(quaternions), data["truth_pelvis"])
            row[arm] = {"whole_deg": round(float(np.median(error)), 5),
                        "bent_deg": round(float(np.median(error[bent])), 5)}
        noiseless[str(seed)] = row
    spread: dict = {}
    for seed, held in frozen.items():
        observation = held["observation"]
        points = observation["points"]
        hip_mid = 0.5 * (points[:, cm.JOINT_INDEX["left_hip"]]
                         + points[:, cm.JOINT_INDEX["right_hip"]])
        lever = 1e3 * np.linalg.norm(observation["spine"] - hip_mid, axis=1)
        truth_lm = held["data"]["truth_landmarks"]
        truth_mid = 0.5 * (truth_lm[:, cm.JOINT_INDEX["left_hip"]]
                           + truth_lm[:, cm.JOINT_INDEX["right_hip"]])
        per_point = {
            name: round(float(np.median(1e3 * np.linalg.norm(
                points[:, cm.JOINT_INDEX[name]] - truth_lm[:, cm.JOINT_INDEX[name]],
                axis=1))), 3)
            for name in ("root", "left_hip", "right_hip", "neck")}
        per_point["Spine1"] = round(float(np.median(1e3 * np.linalg.norm(
            observation["spine"] - held["data"]["truth_spine"], axis=1))), 3)
        spread[str(seed)] = {
            "midhips_to_spine1_sd_mm": round(float(np.std(lever)), 3),
            "truth_lever_sd_mm": round(float(np.std(
                1e3 * np.linalg.norm(held["data"]["truth_spine"] - truth_mid, axis=1))), 5),
            "per_landmark_3d_noise_median_mm": per_point}
    return {
        "noiseless_deficit_per_body": noiseless,
        "observation_spread_per_body": spread,
        "real_take_measured_midhips_to_spine1_sd_mm": D7_REAL_TAKE_MIDHIPS_TO_SPINE1_SD_MM,
        "confound": ("the D3 rig is RIGID so the synthetic lever spread is pure observation "
                     "noise, while the take's mixes noise with real lever variation; "
                     "matching them would under-noise the fixture in the candidate's favour"),
        "summary": {
            "noiseless_rig_modes_bent_deg": round(float(np.median(
                [noiseless[s]["a_kabsch_guarded"]["bent_deg"] for s in noiseless])), 5),
            "noiseless_follower_bent_deg": round(float(np.median(
                [noiseless[s]["frozen_pitch_follower"]["bent_deg"] for s in noiseless])), 5),
            "noiseless_C_on_SOMA_bent_deg": round(float(np.median(
                [noiseless[s]["C_on_SOMA"]["bent_deg"] for s in noiseless])), 5),
            "synthetic_lever_sd_mm_median_of_six": round(float(np.median(
                [spread[s]["midhips_to_spine1_sd_mm"] for s in spread])), 3),
            "real_take_lever_sd_mm": list(
                D7_REAL_TAKE_MIDHIPS_TO_SPINE1_SD_MM.values()),
        },
    }


def g1(frozen: dict, winner: str) -> dict:
    """MISSING-ONLY. An EQUIVALENCE and an error measurement; no superiority claim."""
    mode, _ = RIG_ARMS[winner]
    block: dict = {
        "what_it_is": ("Spine1 set to NaN on performer 1's own 29-frame pattern on every "
                       "body. Guarded and unguarded take the SAME `np.interp` recovery, so "
                       "the interpolated arrays must be bit-identical; the recovery error "
                       "is REPORTED, never banded."),
        "pattern": list(PERFORMER_1_MISSING), "bodies": {}}
    for seed, held in frozen.items():
        data, observation = held["data"], held["observation"]
        spine = np.array(observation["spine"], copy=True)
        spine[list(PERFORMER_1_MISSING)] = np.nan
        points, rest = observation["points"], data["rest"]
        guarded, report_guarded = est.rig_rest_pelvis_frames(
            points, spine, rest, mode=mode, guard=True)
        unguarded, _ = est.rig_rest_pelvis_frames(
            points, spine, rest, mode=mode, guard=False)
        missing = np.zeros(len(spine), bool)
        missing[list(PERFORMER_1_MISSING)] = True
        pairs = halo_pairs(missing)
        truth = data["truth_pelvis"]
        error = geodesic_deg(Rotation.from_quat(guarded), truth)
        step_est = Rotation.from_quat(guarded[1:]) * Rotation.from_quat(guarded[:-1]).inv()
        step_truth = truth[1:] * truth[:-1].inv()
        step = geodesic_deg(step_est, step_truth)
        block["bodies"][str(seed)] = {
            "quaternions_bit_identical": bool(np.array_equal(guarded, unguarded)),
            "guard_additionally_demoted": [
                f for f in report_guarded["lever_guard"]["demoted_frames"]
                if not missing[f]],
            "guard_demoted_count": report_guarded["lever_guard"]["demoted_count"],
            "recovery_error_on_missing_frames_i_deg": round(
                float(np.median(error[missing])), 5),
            "recovery_error_on_transition_pairs_ii_deg": round(
                float(np.median(step[pairs])), 5),
            "i_deg_elsewhere": round(float(np.median(error[~missing])), 5),
        }
    block["equivalence_holds_on_every_body"] = all(
        r["quaternions_bit_identical"] for r in block["bodies"].values())
    return block


def g2(frozen: dict, winner: str) -> dict:
    """FINITE-ONLY. Where the guard must earn its place, and where the stop is located."""
    mode, _ = RIG_ARMS[winner]
    block: dict = {
        "what_it_is": ("30 frames per body (20 %) in runs placed by the card's law, each "
                       "corrupted frame's Spine1 replaced by a DONOR frame's value from at "
                       "least 10 frames away, every donor read from the IMMUTABLE "
                       "pre-corruption array so replacement cannot cascade. A finite, "
                       "DIRECTION-MOVING corruption; a radial rescaling is invisible to "
                       "(b)'s normalised secondary axis and is NOT used."),
        "rule": ("the guarded arm must read BOTH (i) on the corrupted frames AND (ii) on "
                 "the transition pairs below the unguarded arm, per body and on the median "
                 "over the six. A failure of either STOPS the step and does not authorise "
                 "shipping the unguarded winner."),
        "bodies": {}}
    guarded_i, unguarded_i, guarded_ii, unguarded_ii = [], [], [], []
    for seed, held in frozen.items():
        data, observation = held["data"], held["observation"]
        rng = np.random.default_rng(int(seed) + 77)      # frozen before scoring
        spine = observation["spine"]
        runs = g2_runs(rng, len(spine))
        corrupted, mask = g2_corrupt(rng, spine, runs)
        pairs = halo_pairs(mask)
        points, rest = observation["points"], data["rest"]
        truth = data["truth_pelvis"]
        row: dict = {"runs": [[int(s), int(l)] for s, l in runs],
                     "corrupted_frames": int(mask.sum()),
                     "transition_pairs": int(pairs.sum())}
        for label, guard in (("guarded", True), ("unguarded", False)):
            quaternions, report_arm = est.rig_rest_pelvis_frames(
                points, corrupted, rest, mode=mode, guard=guard)
            error = geodesic_deg(Rotation.from_quat(quaternions), truth)
            step_est = (Rotation.from_quat(quaternions[1:])
                        * Rotation.from_quat(quaternions[:-1]).inv())
            step = geodesic_deg(step_est, truth[1:] * truth[:-1].inv())
            row[label] = {
                "i_on_corrupted_frames_deg": round(float(np.median(error[mask])), 5),
                "ii_on_transition_pairs_deg": round(float(np.median(step[pairs])), 5),
                "i_elsewhere_deg": round(float(np.median(error[~mask])), 5)}
            if guard:
                demoted = np.zeros(len(spine), bool)
                demoted[report_arm["lever_guard"]["demoted_frames"]] = True
                row["guard_demoted_count"] = int(demoted.sum())
                row["guard_missed_corrupted_frames"] = [
                    int(f) for f in np.flatnonzero(mask & ~demoted)]
                row["guard_miss_rate"] = round(
                    float((mask & ~demoted).sum() / max(1, mask.sum())), 4)
                row["guard_demoted_uncorrupted_frames"] = [
                    int(f) for f in np.flatnonzero(demoted & ~mask)]
        row["guard_better_on_i"] = bool(
            row["guarded"]["i_on_corrupted_frames_deg"]
            < row["unguarded"]["i_on_corrupted_frames_deg"])
        row["guard_better_on_ii"] = bool(
            row["guarded"]["ii_on_transition_pairs_deg"]
            < row["unguarded"]["ii_on_transition_pairs_deg"])
        guarded_i.append(row["guarded"]["i_on_corrupted_frames_deg"])
        unguarded_i.append(row["unguarded"]["i_on_corrupted_frames_deg"])
        guarded_ii.append(row["guarded"]["ii_on_transition_pairs_deg"])
        unguarded_ii.append(row["unguarded"]["ii_on_transition_pairs_deg"])
        block["bodies"][str(seed)] = row
        print(f"  G2 seed {seed}: guarded i {row['guarded']['i_on_corrupted_frames_deg']:.4f} "
              f"vs unguarded {row['unguarded']['i_on_corrupted_frames_deg']:.4f}; "
              f"guarded ii {row['guarded']['ii_on_transition_pairs_deg']:.4f} vs "
              f"{row['unguarded']['ii_on_transition_pairs_deg']:.4f}; miss rate "
              f"{row['guard_miss_rate']}")
    block["median_of_six"] = {
        "guarded_i_deg": round(float(np.median(guarded_i)), 5),
        "unguarded_i_deg": round(float(np.median(unguarded_i)), 5),
        "guarded_ii_deg": round(float(np.median(guarded_ii)), 5),
        "unguarded_ii_deg": round(float(np.median(unguarded_ii)), 5)}
    block["guard_wins_i_on_the_median"] = bool(
        block["median_of_six"]["guarded_i_deg"] < block["median_of_six"]["unguarded_i_deg"])
    block["guard_wins_ii_on_the_median"] = bool(
        block["median_of_six"]["guarded_ii_deg"] < block["median_of_six"]["unguarded_ii_deg"])
    block["guard_wins_per_body_i"] = {s: r["guard_better_on_i"]
                                      for s, r in block["bodies"].items()}
    block["guard_wins_per_body_ii"] = {s: r["guard_better_on_ii"]
                                       for s, r in block["bodies"].items()}
    block["guard_wins_both"] = bool(
        block["guard_wins_i_on_the_median"] and block["guard_wins_ii_on_the_median"]
        and all(block["guard_wins_per_body_i"].values())
        and all(block["guard_wins_per_body_ii"].values()))
    return block


def finish(report: dict, out: Path, stop: bool) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nS_verdict: {report.get('S_verdict')}")
    if stop:
        print(f"STOP: {report.get('stop')}")
    else:
        print(f"ships: {report.get('ships')}")
    print(f"wrote {out}")
    return 1 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
