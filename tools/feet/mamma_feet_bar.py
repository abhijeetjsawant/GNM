#!/usr/bin/env python3
"""I4 -- MAMMA's feet on THIS footage as the bar, and ours scored against it.

`docs/HEAD_FEET_HANDS_PLAN.md` §7: *"Score MAMMA's own hands and feet on this footage
first. The bar is measured, not assumed."* It never was. `tools/feet/toe_gate.py` scored
SOMA-77's toe landmarks against our own body controls, and
`tools/feet/delivered_foot_is_fiction.py` scored the delivered foot against the
triangulated `ToeBase` direction -- which, since commit f6a4973, is the very direction the
delivered foot is SOLVED FROM. That instrument scores a solver against its own input and
is retired as a gate. What remains is MAMMA's foot, and this measures it.

MAMMA is parity, never truth. Neither side of anything below is ground truth.

===============================================================================
DEFINITIONS -- written down before the code, per the execution plan's I4 row
===============================================================================

1. THE SHIN FRAME.  Per leg, per frame, a right-handed orthonormal frame with its
   origin at the ANKLE:

       d = unit(ankle - knee)              distal shin axis; points knee -> ankle
                                           ("down the shin" in a standing pose)
       m = unit(d x r_hip)                 anterior ("forward")
       l = d x m                           the subject's RIGHT
       F = [m | l | d]   as COLUMNS,       det(F) = +1

   with the documented REFERENCE DIRECTION

       r_hip = unit(left_hip - right_hip)  the pelvic medio-lateral axis, pointing
                                           to the subject's LEFT.

   *Why the pelvis and not the thigh.*  The obvious alternative second reference is the
   knee flexion axis, cross(thigh, shin) -- anatomically the better one, and unusable
   here: it is singular whenever the leg is straight, which is most of a standing frame.
   The pelvic axis is available identically on all three joint sources (MAMMA
   `pred_joints` 1/2, our rig `LeftUpperLeg`/`RightUpperLeg`, SOMA-77 `LeftLeg`/`RightLeg`),
   is far from the shin axis for most of a human pose, and needs no extra landmark.

   *What it costs.*  The pelvis does not rotate with the femur, so hip internal/external
   rotation appears in this frame as apparent foot ab/adduction. It is a shared gauge, not
   an anatomical ankle frame. This is exactly why the ORACLE arm below is not optional:
   MAMMA's own foot direction expressed in OUR shin frames isolates how much of a
   candidate's gap is shin-frame disagreement rather than foot.

   *Degeneracy.*  When d is near-parallel to r_hip, m is ill-conditioned. Frames with
   |sin angle(d, r_hip)| < 0.20 (within 11.5 deg of parallel) are DROPPED, on every arm at
   once, and counted in the report. Physically this is a shin pointing along the pelvic
   left-right axis -- a leg abducted sideways.

2. THE FOOT DIRECTION.  A unit 3-vector expressed in the shin frame:

       f = F^T * unit(foot_point - ankle_point)

   REFERENCE (the bar):  MAMMA `pred_joints`, ankle 7 / 8 and foot 10 / 11, knee 4 / 5,
       hips 1 / 2.  ankle->foot measures 141.0 / 139.1 mm (subject 1) and 150.8 / 148.9 mm
       (subject 0) with standard deviation 0.00 mm -- it is a rigid body model, so no
       length figure on the MAMMA arm is evidence about anything.
   OURS, arm A "delivered":  the shipped `subject-*.body-track.npz`, forward-kinematic
       `LeftFoot -> LeftToes` (and right), with the shin frame from the same FK
       (`LeftLowerLeg`, `LeftFoot`, `Left/RightUpperLeg`). Rig basis is Y-up; converted to
       the capture's Z-up by (x, -z, y) and the conversion is ASSERTED numerically, never
       assumed -- the retired instrument hid the basis change inside its Kabsch alignment.
   OURS, arm B "triangulated ToeBase":  SOMA-77 `Foot`(69/74) -> `ToeBase`(70/75),
       triangulated by `triangulate_point` at production settings on the pipeline's own
       association, with the shin frame from `Shin`(68/73), `Foot`(69/74) and
       `Leg`(67/72).
   ORACLE:  MAMMA's world ankle->foot direction expressed in OUR shin frames (one per our
       arm). It measures the floor our own frame definitions impose.

3. ANGLES, PERIODICITY AND SIGN.

   3D agreement:  theta = acos(clip(<f_a, f_b>, -1, 1)) in [0, 180] deg, on UNIT vectors --
       CLAUDE.md: normalise before acos or an unnormalised matrix reads "0.00 deg apart".
   Constant offset:  the two conventions need not share a zero, so a single rotation R
       (Kabsch over the whole valid set of directions, det-corrected) is fitted ONCE per
       arm and reported as `alignment_offset_deg` = its geodesic magnitude, SEPARATELY
       from the residual spread. It is never summed with the spread and never hidden.
   Decomposition in the shin frame, both periodic quantities declared:
       phi   = atan2(f.m, f.d),  wrapped to (-180, 180] deg
               DORSI/PLANTARFLEXION.  90 deg is a foot square to the shin. phi > 90 deg is
               dorsiflexion (toes toward the shin), phi < 90 deg plantarflexion. Circular:
               medians and percentiles are taken after unwrapping about the circular mean.
       psi   = asin(f.l) in [-90, 90] deg, then signed MEDIALLY:
               psi_med = +asin(f.l) for the LEFT foot, -asin(f.l) for the RIGHT foot,
               so positive is "toward the midline" on both sides. Non-periodic.
               This is AB/ADDUCTION -- foot swing in the transverse plane. It is NOT
               inversion/eversion: inversion is roll ABOUT the foot's long axis, a third
               degree of freedom that a single ankle->foot DIRECTION (2 DoF) cannot carry
               at all. The execution plan's wording is corrected here deliberately.

4. THREE THINGS, NEVER SUMMED.
   (a) FOOT ORIENTATION -- everything above.
   (b) TOE ARTICULATION -- reported and NOT SCOREABLE against this reference. SMPL-X's
       foot chain ends at the ball joint (10 / 11); there is no toe-end joint in
       `pred_joints`, so MAMMA has no toe articulation on this footage to be a bar. Our
       delivered `LeftToes`/`RightToes` are the identity quaternion on every frame of both
       subjects. Both facts are reported as counts, in their own block, with no angle.
   (c) GROUND CONTACT -- a separate block, from MAMMA's `smplx_floor_contact` restricted to
       its OWN `left_feet` / `right_feet` landmark sets
       (`.cache/mamma/.../smplx_512_body_parts.npz`, membership self-checked against joints
       10/11), against the delivered `foot_contacts` flag. Reported as the distribution of
       MAMMA's foot floor-contact probability on frames where our flag is True vs False,
       plus a shuffled control. NO probability threshold is chosen: a threshold picked to
       maximise agreement with MAMMA would be a MAMMA-selected constant.

===============================================================================
GATE -- pre-registered, from anatomy, never from MAMMA agreement
===============================================================================
G1 ORIENTATION ZERO.  `mean_direction_offset_deg` <= 90 -- the angle between the two
   take-MEAN foot directions.  An ankle->ball direction cannot need a
   posterior-pointing correction relative to its own shin. This band exists because Kabsch
   offset removal is a proper rotation and would otherwise ABSORB the mirrored-anterior
   control; the raw, un-removed median is reported beside it for every arm.
G2 TRACKING.  P(candidate's offset-removed median < the WELDED control's) >= 0.95 on a
   moving-block bootstrap, candidate and control on identical draws.
G3 The remaining controls (constant-in-world axis, time-shuffled MAMMA, L/R swap,
   mirrored anterior) each reported with the same paired P(beat).

PRE-REGISTERED POSSIBLE OUTCOME, so that it is not written up as a pass if it happens:
if MAMMA's own range of motion in the shin frame is small at this 5 m working distance,
NO candidate whose noise exceeds that range can beat the welded control. That result reads
"the foot signal on this footage is below our noise floor" -- it is a finding, not an
instrument defect, and nothing is to be changed to chase it.

STATISTICS.  One take, 150 frames, lag-1 autocorrelation ~0.99 in this lane. Moving-block
bootstrap only (blocks 10/20/30, 2000 draws, fixed seed); every arm resampled on the
IDENTICAL draw so a difference between arms is not itself a random variable. The alignment
rotation is fitted once on the full valid set, not per draw.

===============================================================================
blind_to
===============================================================================
  * ACCURACY, entirely. MAMMA is a reference, not truth; `gt_joints`/`gt_vertices` on disk
    are byte-copies of `pred_*` and nothing here touches them. Both sides can be wrong
    together and this would not see it.
  * INVERSION/EVERSION. Roll about the foot's long axis is a degree of freedom no
    ankle->foot direction carries, on either arm.
  * TOE ARTICULATION, for want of a reference joint (see 4b).
  * A CONSTANT OFFSET is measured but not adjudicated beyond G1's 90 deg: the spread
    figures are a TRACKING comparison and are blind to where the foot actually points.
  * HIP ROTATION leaks into ab/adduction through the pelvic reference direction; the
    oracle arm bounds that leak, it does not remove it.
  * MAMMA's foot is a fitted body model with 0.00 mm segment variation, so its smoothness
    is partly its own prior. A smooth foot can be smoothly wrong.
  * ONE TAKE, TWO PERFORMERS. No claim generalises past this fixture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))
from autoanim_gnm.body import forward_kinematics_positions, skeleton_for_joint_names  # noqa: E402
from autoanim_gnm.commercial_multiview import JOINT_INDEX  # noqa: E402
from subject_map import mamma_index_for  # noqa: E402
from triangulate_soma import triangulate  # noqa: E402

MA3D = ROOT / "artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/pushing_and_lifting_from_ground"
TRACKS = ROOT / "artifacts/commercial-multiview-soma77"
BODY_PARTS = ROOT / ".cache/mamma/data/body_models/downsampled_verts/smplx_512_body_parts.npz"
OUT = ROOT / "artifacts/feet-lane/mamma-feet-bar.json"

# --- MAMMA / SMPL-X joint indices (verified against SMPLX_NEUTRAL kintree elsewhere in
#     this lane; ankle->foot lengths 139-151 mm confirm the semantics).
M = {
    "L": {"hip": 1, "knee": 4, "ankle": 7, "foot": 10},
    "R": {"hip": 2, "knee": 5, "ankle": 8, "foot": 11},
}
M_HIP_L, M_HIP_R = 1, 2

# --- SOMA-77 indices, from src/autoanim_gnm/data/somaskel77-v1.json
SOMA = {
    "LeftLeg": 67, "LeftShin": 68, "LeftFoot": 69, "LeftToeBase": 70,
    "RightLeg": 72, "RightShin": 73, "RightFoot": 74, "RightToeBase": 75,
}
SOMA_ORDER = list(SOMA)
SLOT = {n: i for i, n in enumerate(SOMA_ORDER)}

# --- delivered rig joint names
RIG = {
    "L": {"hip": "LeftUpperLeg", "knee": "LeftLowerLeg", "ankle": "LeftFoot", "foot": "LeftToes"},
    "R": {"hip": "RightUpperLeg", "knee": "RightLowerLeg", "ankle": "RightFoot", "foot": "RightToes"},
}

CONDITION_MIN_SIN = 0.20          # shin vs pelvic axis; 11.5 deg of parallel
OFFSET_BAND_DEG = 90.0            # G1
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_BLOCKS = (10, 20, 30)
SEED = 20260902


# --------------------------------------------------------------------------- primitives
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.full_like(v, np.nan), where=n > 1e-12)


def shin_frames(hip_l, hip_r, knee, ankle) -> tuple[np.ndarray, np.ndarray]:
    """Return (F[n,3,3] with columns [m|l|d], |sin| conditioning[n])."""
    d = unit(ankle - knee)
    r_hip = unit(hip_l - hip_r)
    cross = np.cross(d, r_hip)
    conditioning = np.linalg.norm(cross, axis=-1)      # = |sin angle(d, r_hip)|
    m = unit(cross)
    l = np.cross(d, m)
    return np.stack([m, l, d], axis=-1), conditioning


def in_frame(F: np.ndarray, ankle: np.ndarray, foot: np.ndarray) -> np.ndarray:
    """Foot direction expressed in the shin frame: f = F^T (foot - ankle), unit."""
    return unit(np.einsum("nji,nj->ni", F, unit(foot - ankle)))


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", a, b), -1.0, 1.0)))


def kabsch_directions(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Proper rotation R minimising |source @ R - target|, and its CONDITIONING.

    Conditioning is the ratio of the smallest to the largest singular value of the
    cross-covariance. It matters here and was not a footnote: a set of unit directions
    clustered about a mean (which every arm on this footage is -- the spreads are 3-25
    deg) makes the cross-covariance near rank-1, so the component of R about that mean
    direction is barely determined; for the WELDED control, whose source is a single
    constant direction, it is not determined at all and the returned rotation magnitude is
    arbitrary. The first run of this instrument reported 130-173 deg offsets for exactly
    that reason. The gated offset is therefore the angle between the two take-MEAN
    directions, which is always identified; the Kabsch magnitude is reported beside it
    with this conditioning number so the reader can see when it means nothing.
    """
    u, s, vt = np.linalg.svd(source.T @ target)
    R = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
    return R, float(s[-1] / s[0])


def rotation_magnitude_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def circular_stats(deg: np.ndarray) -> dict:
    """Median / percentiles of a (-180, 180] periodic quantity, unwrapped about its mean."""
    rad = np.radians(deg)
    mean = np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))
    centred = (deg - mean + 180.0) % 360.0 - 180.0
    return {
        "circular_mean_deg": float(mean),
        "median_deg": float(mean + np.median(centred)),
        "p05_deg": float(mean + np.percentile(centred, 5)),
        "p95_deg": float(mean + np.percentile(centred, 95)),
        "range_deg": float(centred.max() - centred.min()),
        "sd_deg": float(centred.std()),
    }


def linear_stats(x: np.ndarray) -> dict:
    return {
        "median_deg": float(np.median(x)), "p05_deg": float(np.percentile(x, 5)),
        "p95_deg": float(np.percentile(x, 95)),
        "range_deg": float(x.max() - x.min()), "sd_deg": float(x.std()),
    }


def spread_stats(x: np.ndarray) -> dict:
    return {"median_deg": float(np.median(x)), "p95_deg": float(np.percentile(x, 95)),
            "max_deg": float(x.max())}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


# --------------------------------------------------------------- range of motion, the bar
def decompose(f: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(phi, psi_medial, ankle_angle) in degrees, per the docstring's definitions."""
    phi = (np.degrees(np.arctan2(f[:, 0], f[:, 2])) + 180.0) % 360.0 - 180.0
    psi = np.degrees(np.arcsin(np.clip(f[:, 1], -1.0, 1.0))) * (1.0 if side == "L" else -1.0)
    ankle_angle = np.degrees(np.arccos(np.clip(f[:, 2], -1.0, 1.0)))
    return phi, psi, ankle_angle


def component_agreement(candidate: np.ndarray, reference: np.ndarray, side: str) -> dict:
    """Where the disagreement LIVES: dorsi/plantarflexion or ab/adduction.

    This is what tells a reader whether an 18-26 deg constant offset is a ball-joint
    PLACEMENT convention between SOMA-77 and SMPL-X (flexion -- benign, the two skeletons
    put the ball in different places along the foot) or toes pointing outward by 20 deg
    (ab/adduction -- a delivery concern that would be visible on screen).
    """
    phi_c, psi_c, _ = decompose(candidate, side)
    phi_r, psi_r, _ = decompose(reference, side)
    dphi = (phi_c - phi_r + 180.0) % 360.0 - 180.0
    dpsi = psi_c - psi_r
    rad = np.radians(dphi)
    phi_offset = float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))
    phi_centred = (dphi - phi_offset + 180.0) % 360.0 - 180.0
    psi_offset = float(dpsi.mean())
    return {
        "dorsi_plantarflexion_dphi": {
            "offset_circular_mean_deg": phi_offset,
            "spread_sd_deg": float(phi_centred.std()),
            "spread_median_abs_deviation_deg": float(np.median(np.abs(phi_centred))),
            "p95_abs_deviation_deg": float(np.percentile(np.abs(phi_centred), 95)),
        },
        "ab_adduction_dpsi": {
            "offset_mean_deg": psi_offset,
            "spread_sd_deg": float((dpsi - psi_offset).std()),
            "spread_median_abs_deviation_deg": float(np.median(np.abs(dpsi - psi_offset))),
            "p95_abs_deviation_deg": float(np.percentile(np.abs(dpsi - psi_offset), 95)),
        },
        "note": "offset and spread are reported separately per component and are never "
                "summed with each other or with the 3D figures",
    }


def lag1_autocorrelation(x: np.ndarray) -> float:
    centred = x - x.mean()
    denominator = float((centred * centred).sum())
    return float((centred[1:] * centred[:-1]).sum() / denominator) if denominator > 0 else float("nan")


def range_of_motion(f: np.ndarray, side: str) -> dict:
    """Foot direction statistics in the shin frame. On MAMMA's arm this IS the bar."""
    phi, psi, ankle_angle = decompose(f, side)
    mean_dir = unit(f.mean(axis=0)[None, :])[0]
    about_mean = angle_between(f, np.broadcast_to(mean_dir, f.shape))
    travel = angle_between(f[1:], f[:-1])
    return {
        "frames": int(len(f)),
        "dorsi_plantarflexion_phi": circular_stats(phi),
        "ab_adduction_psi_medial_positive": linear_stats(psi),
        "ankle_angle_shin_to_foot_deg": linear_stats(ankle_angle),
        "spread_about_take_mean_direction_deg": spread_stats(about_mean),
        "frame_to_frame_travel_deg": spread_stats(travel),
        "total_angular_range_deg": float(
            angle_between(
                np.repeat(f, len(f), axis=0), np.tile(f, (len(f), 1))
            ).max()
        ),
    }


def where_it_lives(blocks: dict) -> dict:
    """Split the constant offset into flexion and ab/adduction, and say which dominates.

    A flexion offset is a ball-joint PLACEMENT convention -- SOMA-77 and SMPL-X put the
    ball at different points along the foot -- and is benign. An ab/adduction offset is
    the toes pointing inward or outward by that much, which is visible on screen. The
    oracle's own component offsets bound how much of it is the shin frame rather than the
    foot, and are quoted beside it so the attribution is not assumed.
    """
    out = {}
    for arm in ("ours_delivered", "ours_triangulated_toebase",
                "ORACLE_mamma_foot_in_our_triangulated_shin_frame",
                "ORACLE_mamma_foot_in_our_delivered_shin_frame"):
        c = blocks[arm]["component_agreement_in_the_shin_frame"]
        out[arm] = {
            "dorsi_plantarflexion_offset_deg": c["dorsi_plantarflexion_dphi"]["offset_circular_mean_deg"],
            "ab_adduction_offset_deg": c["ab_adduction_dpsi"]["offset_mean_deg"],
        }
    a = out["ours_triangulated_toebase"]
    oracle = out["ORACLE_mamma_foot_in_our_triangulated_shin_frame"]
    dominant = ("ab/adduction" if abs(a["ab_adduction_offset_deg"]) > abs(a["dorsi_plantarflexion_offset_deg"])
                else "dorsi/plantarflexion")
    out["reading"] = (
        f"the constant offset is dominated by {dominant}: our triangulated foot sits "
        f"{a['ab_adduction_offset_deg']:+.1f} deg in ab/adduction (positive = MORE MEDIAL, "
        f"toes turned inward) and {a['dorsi_plantarflexion_offset_deg']:+.1f} deg in "
        f"flexion, against MAMMA's foot. The oracle -- MAMMA's OWN foot through our shin "
        f"frame -- carries {oracle['ab_adduction_offset_deg']:+.1f} deg and "
        f"{oracle['dorsi_plantarflexion_offset_deg']:+.1f} deg of the same, so the shin "
        f"frame accounts for only part of it and the rest is the foot itself.")
    return out


def adjudicate(blocks: dict, boot: dict) -> dict:
    """Pass/fail per candidate, on the WORST block size -- never the friendliest one.

    A control is "rejected" if the candidate beats it with P >= 0.95 on every block size,
    OR if the control itself fails G1 (its foot points more than 90 deg away from the
    reference's, which no ankle can). Both are ways of failing; a gate that only counted
    the first would be defeated by the mirrored-anterior control, whose offset-removed
    spread is competitive precisely because the Kabsch alignment absorbs the mirror.
    """
    controls = [name for name in blocks if name.startswith("CONTROL_")]
    out: dict = {}
    for cand in ("ours_delivered", "ours_triangulated_toebase"):
        rows = {}
        for control in controls:
            p_min = min(float(boot[b][cand]["P_beats"][control]) for b in boot)
            control_g1 = blocks[control]["G1_offset_within_90deg"]
            rows[control] = {
                "P_candidate_beats_min_over_blocks": p_min,
                "control_fails_G1_orientation_zero": not control_g1,
                "control_rejected": bool(p_min >= 0.95 or not control_g1),
            }
        welded = rows["CONTROL_welded_to_shin_zero_articulation"]
        out[cand] = {
            "G1_pass": blocks[cand]["G1_offset_within_90deg"],
            "G1_mean_direction_offset_deg": blocks[cand]["mean_direction_offset_deg"],
            "G2_beats_welded_P_min_over_blocks": welded["P_candidate_beats_min_over_blocks"],
            "G2_pass": bool(welded["P_candidate_beats_min_over_blocks"] >= 0.95),
            "G3_controls": rows,
            "all_controls_rejected": bool(all(r["control_rejected"] for r in rows.values())),
        }
        unrejected = [name for name, r in rows.items() if not r["control_rejected"]]
        entry = out[cand]
        if not entry["G1_pass"]:
            entry["verdict"] = (
                f"FAILS G1: its take-mean foot direction sits "
                f"{entry['G1_mean_direction_offset_deg']:.1f} deg from the reference's, "
                "which no ankle can produce")
        elif not entry["G2_pass"]:
            entry["verdict"] = ("NOT SHOWN to beat a foot welded to the shin on this take "
                                f"(P = {entry['G2_beats_welded_P_min_over_blocks']:.3f})")
        elif unrejected:
            entry["verdict"] = ("beats the welded control, but does not clear "
                                + ", ".join(unrejected) + " at P >= 0.95")
        else:
            entry["verdict"] = "carries foot orientation beyond a welded foot"
    return out


# ------------------------------------------------------------------------- arm comparison
def compare(candidate: np.ndarray, reference: np.ndarray, side: str = "L") -> tuple[dict, np.ndarray]:
    """Score one arm against the reference. Returns (block, per-frame residual series)."""
    raw = angle_between(candidate, reference)
    R, conditioning = kabsch_directions(candidate, reference)
    residual = angle_between(candidate @ R, reference)
    mean_cand = unit(candidate.mean(axis=0)[None, :])[0]
    mean_ref = unit(reference.mean(axis=0)[None, :])[0]
    offset = float(angle_between(mean_cand[None, :], mean_ref[None, :])[0])
    block = {
        "frames": int(len(candidate)),
        "raw_angle_no_offset_removed_deg": spread_stats(raw),
        "mean_direction_offset_deg": offset,
        "kabsch_rotation_magnitude_deg": rotation_magnitude_deg(R),
        "kabsch_conditioning_smallest_over_largest_singular_value": conditioning,
        "spread_after_offset_removed_deg": spread_stats(residual),
        "component_agreement_in_the_shin_frame": component_agreement(candidate, reference, side),
        "residual_series_lag1_autocorrelation": lag1_autocorrelation(residual),
        "G1_offset_within_90deg": bool(offset <= OFFSET_BAND_DEG),
    }
    return block, residual


def bootstrap(series: dict[str, np.ndarray], candidates: list[str]) -> dict:
    """P(candidate median < other median) on identical moving-block draws."""
    names = list(series)
    n = len(next(iter(series.values())))
    out: dict = {}
    for block in BOOTSTRAP_BLOCKS:
        rng = np.random.default_rng(SEED)
        medians = {name: np.empty(BOOTSTRAP_DRAWS) for name in names}
        for draw in range(BOOTSTRAP_DRAWS):
            idx = moving_block_indices(n, block, rng)
            for name in names:
                medians[name][draw] = np.median(series[name][idx])
        entry = {
            name: {
                "point_median_deg": float(np.median(series[name])),
                "ci95_deg": [float(np.percentile(medians[name], 2.5)),
                             float(np.percentile(medians[name], 97.5))],
            }
            for name in names
        }
        for cand in candidates:
            entry[cand]["P_beats"] = {
                other: float((medians[cand] < medians[other]).mean())
                for other in names if other != cand
            }
        out[str(block)] = entry
    return out


# ------------------------------------------------------------------------------- contact
def contact_block(subject: int, mamma_id: int, our_flags: np.ndarray, n: int) -> dict:
    parts = np.load(BODY_PARTS, allow_pickle=True)
    params = np.load(MA3D / f"smplx_params_body_id-{mamma_id:02d}.npz", allow_pickle=True)
    joints = np.load(MA3D / f"verts_joints_body_id-{mamma_id:02d}.npz",
                     allow_pickle=True)["pred_joints"].astype(np.float64)
    points = params["triangulated_3d_pts"].astype(np.float64)
    floor = params["smplx_floor_contact"].astype(np.float64)
    rng = np.random.default_rng(SEED)
    block: dict = {
        "verdict": "NO USABLE AGREEMENT MEASURED. The mean over 44 foot landmarks includes "
                   "the dorsum and saturates near 0.25-0.35 whether the foot is planted or "
                   "not, so the true/false medians sit within 0.01-0.09 of each other and "
                   "the shuffled control lands on top of them on subject 0. The max "
                   "reducer is reported beside it without a threshold. Our own flag fires "
                   "on 3-28 % of frames of a take spent lifting from the ground. Nothing "
                   "here supports a claim in either direction, and nothing here may select "
                   "a contact constant.",
        "note": "MAMMA's contact is a per-landmark PROBABILITY over its own 512 points; "
                "ours is a boolean flag from a different definition. No threshold is "
                "chosen here -- a threshold fitted to maximise agreement would be a "
                "MAMMA-selected constant. Never combined with the orientation figures.",
    }
    for side, key, joint in (("L", "left_feet", 10), ("R", "right_feet", 11)):
        idx = parts[key]
        centre = np.nanmedian(points[:n, idx], axis=1)
        own = np.nanmedian(np.linalg.norm(centre - joints[:n, joint], axis=1)) * 1000.0
        other = np.nanmedian(
            np.linalg.norm(centre - joints[:n, 11 if joint == 10 else 10], axis=1)) * 1000.0
        flag = our_flags[:n, 0 if side == "L" else 1].astype(bool)
        reducers = {"mean_over_the_44_foot_landmarks": floor[:n, idx].mean(axis=1),
                    "max_over_the_44_foot_landmarks": floor[:n, idx].max(axis=1)}
        by_reducer = {}
        for reducer, probability in reducers.items():
            shuffled = probability[rng.permutation(n)]
            by_reducer[reducer] = {
                "median_when_our_flag_true": (
                    float(np.median(probability[flag])) if flag.any() else None),
                "median_when_our_flag_false": (
                    float(np.median(probability[~flag])) if (~flag).any() else None),
                "shuffled_control_median_when_our_flag_true": (
                    float(np.median(shuffled[flag])) if flag.any() else None),
            }
        probability = reducers["mean_over_the_44_foot_landmarks"]
        shuffled = probability[rng.permutation(n)]
        block[side] = {
            "by_reducer": by_reducer,
            "selfcheck_landmark_centroid_to_own_foot_joint_mm": float(own),
            "selfcheck_landmark_centroid_to_other_foot_joint_mm": float(other),
            "selfcheck_passes": bool(own < other),
            "mamma_floor_contact_probability_when_our_flag_true": {
                "frames": int(flag.sum()),
                "median": float(np.median(probability[flag])) if flag.any() else None,
            },
            "mamma_floor_contact_probability_when_our_flag_false": {
                "frames": int((~flag).sum()),
                "median": float(np.median(probability[~flag])) if (~flag).any() else None,
            },
            "shuffled_control_when_our_flag_true_median": (
                float(np.median(shuffled[flag])) if flag.any() else None),
            "our_flag_true_frames_of": [int(flag.sum()), int(n)],
        }
    return block


# ---------------------------------------------------------------------------------- main
def assert_rig_basis(world_rig: np.ndarray, names: list[str], captured_z_up: np.ndarray) -> dict:
    """The rig is Y-up and the capture Z-up. Prove the conversion instead of assuming it."""
    ankle = world_rig[:, names.index("LeftFoot")]
    target = captured_z_up[:, JOINT_INDEX["left_ankle"]]
    ok = np.isfinite(target).all(axis=1)
    bases = {
        "(x,-z,y) CHOSEN": np.stack([ankle[:, 0], -ankle[:, 2], ankle[:, 1]], axis=-1),
        "(x,y,z) identity": ankle,
        "(x,z,-y)": np.stack([ankle[:, 0], ankle[:, 2], -ankle[:, 1]], axis=-1),
        "(-z,x,y)": np.stack([-ankle[:, 2], ankle[:, 0], ankle[:, 1]], axis=-1),
        "(y,-z,x)": np.stack([ankle[:, 1], -ankle[:, 2], ankle[:, 0]], axis=-1),
    }
    scores = {k: float(np.median(np.linalg.norm(v[ok] - target[ok], axis=1)) * 1000.0)
              for k, v in bases.items()}
    best = min(scores, key=scores.get)
    if best != "(x,-z,y) CHOSEN":
        raise SystemExit(f"rig->capture basis assertion FAILED: {scores}")
    return scores


def rig_to_z_up(w: np.ndarray) -> np.ndarray:
    return np.stack([w[..., 0], -w[..., 2], w[..., 1]], axis=-1)


def main() -> None:
    soma_positions, _, used = triangulate([SOMA[n] for n in SOMA_ORDER])

    ours_for_map = np.stack([
        np.load(TRACKS / f"subject-{s:02d}.body-track.npz")[
            "triangulated_world_positions_z_up_m"] for s in (0, 1)])
    mapping = mamma_index_for(ours_for_map)

    report: dict = {
        "instrument": "tools/feet/mamma_feet_bar.py",
        "step": "I4",
        "reference": "MAMMA pred_joints (ankle 7/8, foot 10/11, knee 4/5, hip 1/2) -- "
                     "parity, never truth; nothing scored against any gt_ variable",
        "subject_correspondence": {
            f"our_subject_{ours:02d}": f"mamma_body_id-{theirs:02d}"
            for ours, theirs in sorted(mapping.items())},
        "definitions": {
            "shin_frame": "origin ankle; d = unit(ankle-knee); m = unit(d x r_hip) "
                          "(anterior); l = d x m (subject's right); F = [m|l|d] columns, "
                          "right-handed. r_hip = unit(left_hip - right_hip), the pelvic "
                          "medio-lateral axis, chosen over the knee flexion axis because "
                          "cross(thigh, shin) is singular on a straight leg. Cost: hip "
                          "internal/external rotation reads as foot ab/adduction, which "
                          "is what the ORACLE arm bounds.",
            "degeneracy": f"|sin angle(d, r_hip)| < {CONDITION_MIN_SIN} dropped on every "
                          "arm at once (shin near-parallel to the pelvic axis)",
            "foot_direction": "f = F^T unit(foot - ankle); MAMMA ankle 7/8 -> foot 10/11; "
                              "ours arm A = delivered FK LeftFoot->LeftToes; arm B = "
                              "triangulated SOMA-77 Foot(69/74)->ToeBase(70/75)",
            "phi_dorsi_plantarflexion": "atan2(f.m, f.d), wrapped to (-180,180], PERIODIC; "
                                        "90 deg = foot square to shin; >90 dorsiflexion, "
                                        "<90 plantarflexion; circular median/percentiles",
            "psi_ab_adduction": "asin(f.l) in [-90,90], sign flipped on the RIGHT foot so "
                                "positive is medial on both sides; NOT periodic. This is "
                                "AB/ADDUCTION. Inversion/eversion is roll about the foot's "
                                "long axis and NO ankle->foot direction can carry it.",
            "agreement_angle": "acos(clip(<f_a,f_b>,-1,1)) on unit vectors; the constant "
                               "convention offset is a single det-corrected Kabsch rotation "
                               "fitted ONCE per arm over the whole valid set and reported "
                               "separately from the spread, never summed with it",
        },
        "gate": {
            "G1_orientation_zero": f"mean_direction_offset_deg <= {OFFSET_BAND_DEG}, the "
                                   "angle between the two take-MEAN foot directions -- an "
                                   "ankle->ball direction cannot need a posterior "
                                   "correction relative to its own shin. Exists because "
                                   "the Kabsch alignment would otherwise ABSORB the "
                                   "mirrored-anterior control (it does: that control's "
                                   "offset-removed spread beats nothing). NOT the Kabsch "
                                   "rotation magnitude: for a direction cloud concentrated "
                                   "about a mean the rotation about that mean is barely "
                                   "identified, and for the constant WELDED control not at "
                                   "all -- see kabsch_conditioning_* beside each arm.",
            "G2_tracking": "P(candidate median < WELDED control median) >= 0.95 on the "
                           "moving-block bootstrap, identical draws",
            "G3": "every other control reported with the same paired P(beat). The "
                  "mirrored-anterior control is built from the DELIVERED arm (its "
                  "anterior component negated in the shin frame); the L/R swap control "
                  "exists in both a delivered and a triangulated form so each candidate "
                  "is scored against a swap made by its own estimator.",
            "preregistered_possible_outcome":
                "If MAMMA's own shin-frame range of motion is small at this working "
                "distance, no candidate whose noise exceeds it can beat the welded "
                "control. That reads 'the foot signal on this footage is below our noise "
                "floor' -- a finding, not a defect. Nothing is changed to chase it.",
        },
        "statistics": {"bootstrap": "moving block, identical draws for every arm",
                       "draws": BOOTSTRAP_DRAWS, "blocks": list(BOOTSTRAP_BLOCKS),
                       "seed": SEED,
                       "note": "one take, ~150 correlated frames; the lag-1 "
                               "autocorrelation of each arm's own residual series is "
                               "MEASURED here and reported per arm as "
                               "residual_series_lag1_autocorrelation, rather than taken "
                               "from the head lane. The alignment rotation is fitted once "
                               "on the full valid set, not per draw."},
        "input_sha256": {},
        "subjects": {},
        "blind_to":
            "ACCURACY entirely -- MAMMA is a reference, not truth, and both sides can be "
            "wrong together. INVERSION/EVERSION -- roll about the foot's long axis is a "
            "degree of freedom no ankle->foot direction carries on either arm. TOE "
            "ARTICULATION -- SMPL-X's foot chain ends at the ball joint, so there is no "
            "reference toe to score against. ABSOLUTE ORIENTATION beyond G1's 90 deg band "
            "-- the spread figures are a tracking comparison. HIP ROTATION leaks into "
            "ab/adduction through the pelvic reference direction; the oracle arm bounds "
            "that leak but does not remove it. MAMMA's foot is a fitted body model with "
            "0.00 mm segment variation, so its smoothness is partly its own prior. ONE "
            "TAKE, TWO PERFORMERS -- nothing here generalises past this fixture.",
    }
    for path in (BODY_PARTS, TRACKS / "camera-rig.json",
                 ROOT / "artifacts/head-lane/association.npz",
                 *[MA3D / f"verts_joints_body_id-{i:02d}.npz" for i in (0, 1)],
                 *[MA3D / f"smplx_params_body_id-{i:02d}.npz" for i in (0, 1)],
                 *[TRACKS / f"subject-{s:02d}.body-track.npz" for s in (0, 1)]):
        report["input_sha256"][str(path.relative_to(ROOT))] = sha256(path)

    print(f"subject map: " + ", ".join(f"ours {k} -> body_id-{v:02d}"
                                       for k, v in sorted(mapping.items())))

    for subject in (0, 1):
        mid = mapping[subject]
        m_joints = np.load(MA3D / f"verts_joints_body_id-{mid:02d}.npz",
                           allow_pickle=True)["pred_joints"].astype(np.float64)
        track = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")
        names = json.loads((TRACKS / f"subject-{subject:02d}.body-track.json").read_text())[
            "joint_names"]
        world_rig = forward_kinematics_positions(
            track["root_translation_m"], track["local_rotations_xyzw"],
            skeleton=skeleton_for_joint_names(names))
        basis_scores = assert_rig_basis(
            world_rig, names, track["raw_triangulated_world_positions_z_up_m"])
        world = rig_to_z_up(world_rig)
        n = min(len(world), len(m_joints), soma_positions.shape[1])
        soma = soma_positions[subject]

        segment_sd = {
            f"mamma_ankle_to_foot_{side}_mm": {
                "mean": float(np.linalg.norm(
                    m_joints[:, M[side]["foot"]] - m_joints[:, M[side]["ankle"]], axis=1).mean() * 1000.0),
                "sd": float(np.linalg.norm(
                    m_joints[:, M[side]["foot"]] - m_joints[:, M[side]["ankle"]], axis=1).std() * 1000.0),
            } for side in ("L", "R")}

        subject_block: dict = {
            "mamma_body_id": mid,
            "plan_corrections": {
                "toe_articulation_is_not_scoreable_against_mamma":
                    "The I4 gate card asks for toe articulation 'if scoreable'. It is not: "
                    "SMPL-X's foot chain ends at the ball joint (pred_joints 10/11) and "
                    "there is no toe-end joint in the reference at all. Our delivered "
                    "Toes channels are the identity quaternion on every frame, so there "
                    "is nothing on either side of that comparison.",
                "inversion_eversion_is_not_observable_from_a_direction":
                    "The plan asks for a decomposition into plantar/dorsiflexion and "
                    "inversion/eversion. Inversion/eversion is ROLL ABOUT the foot's long "
                    "axis -- a third degree of freedom that a single ankle->foot direction "
                    "(2 DoF) cannot carry on either arm. What is reported in its place is "
                    "AB/ADDUCTION, and it is named that.",
                "no_length_figure_on_the_mamma_arm_means_anything":
                    "MAMMA's pred_joints come from a fitted body model whose segment "
                    "lengths are constant to 0.00 mm over the take (measured below), so a "
                    "length invariant on that arm is a property of the model, not of the "
                    "footage.",
                "mamma_segment_lengths": segment_sd,
                "our_two_arms_are_not_independent":
                    "Since commit f6a4973 the delivered foot is SOLVED FROM the "
                    "triangulated ToeBase direction, so the two candidate arms are two "
                    "views of one estimator, not two estimators. Only the MAMMA "
                    "comparison is independent; delivered-vs-ToeBase is not. The measured "
                    "gap between the two arms is filled in below once both are scored.",
                "delivered_minus_triangulated_median_deg": None,
            },
            "mamma_segment_length_mm": segment_sd,
            "rig_to_capture_basis_assertion_median_mm": basis_scores,
            "toe_articulation": {
                "scoreable_against_mamma": False,
                "why": "SMPL-X's foot chain ends at the ball joint (pred_joints 10/11); "
                       "there is no toe-end joint in the reference at all",
                "our_delivered_LeftToes_distinct_rotations": int(len(np.unique(
                    np.round(track["local_rotations_xyzw"][:, names.index("LeftToes")], 6),
                    axis=0))),
                "our_delivered_RightToes_distinct_rotations": int(len(np.unique(
                    np.round(track["local_rotations_xyzw"][:, names.index("RightToes")], 6),
                    axis=0))),
                "our_delivered_toes_are_identity_on_every_frame": bool(np.allclose(
                    track["local_rotations_xyzw"][:, [names.index("LeftToes"),
                                                      names.index("RightToes")]],
                    np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6)),
            },
            "ground_contact": contact_block(subject, mid, track["foot_contacts"], n),
            "feet": {},
        }

        for side in ("L", "R"):
            long = "Left" if side == "L" else "Right"
            other = "Right" if side == "L" else "Left"

            # ---------------- the three joint sources, each with its own shin frame
            F_m, cond_m = shin_frames(m_joints[:n, M_HIP_L], m_joints[:n, M_HIP_R],
                                      m_joints[:n, M[side]["knee"]],
                                      m_joints[:n, M[side]["ankle"]])
            f_ref = in_frame(F_m, m_joints[:n, M[side]["ankle"]], m_joints[:n, M[side]["foot"]])

            r = RIG[side]
            F_d, cond_d = shin_frames(world[:n, names.index("LeftUpperLeg")],
                                      world[:n, names.index("RightUpperLeg")],
                                      world[:n, names.index(r["knee"])],
                                      world[:n, names.index(r["ankle"])])
            f_del = in_frame(F_d, world[:n, names.index(r["ankle"])],
                             world[:n, names.index(r["foot"])])

            F_t, cond_t = shin_frames(soma[:n, SLOT["LeftLeg"]], soma[:n, SLOT["RightLeg"]],
                                      soma[:n, SLOT[f"{long}Shin"]],
                                      soma[:n, SLOT[f"{long}Foot"]])
            f_tri = in_frame(F_t, soma[:n, SLOT[f"{long}Foot"]],
                             soma[:n, SLOT[f"{long}ToeBase"]])

            # the other side, for the L/R swap control (its own shin frame, as delivered)
            ro = RIG["R" if side == "L" else "L"]
            F_o, cond_o = shin_frames(world[:n, names.index("LeftUpperLeg")],
                                      world[:n, names.index("RightUpperLeg")],
                                      world[:n, names.index(ro["knee"])],
                                      world[:n, names.index(ro["ankle"])])
            f_swap = in_frame(F_o, world[:n, names.index(ro["ankle"])],
                              world[:n, names.index(ro["foot"])])
            F_ot, cond_ot = shin_frames(soma[:n, SLOT["LeftLeg"]], soma[:n, SLOT["RightLeg"]],
                                        soma[:n, SLOT[f"{other}Shin"]],
                                        soma[:n, SLOT[f"{other}Foot"]])
            f_swap_tri = in_frame(F_ot, soma[:n, SLOT[f"{other}Foot"]],
                                  soma[:n, SLOT[f"{other}ToeBase"]])

            # ---------------- ORACLE: MAMMA's WORLD foot direction in OUR shin frames
            m_world = unit(m_joints[:n, M[side]["foot"]] - m_joints[:n, M[side]["ankle"]])
            f_or_del = unit(np.einsum("nji,nj->ni", F_d, m_world))
            f_or_tri = unit(np.einsum("nji,nj->ni", F_t, m_world))

            # ---------------- the frame mask: same denominator on every arm
            reasons = {
                "pipeline_rejected_subject_frame": int((~used[:n, subject]).sum()),
                "soma_triangulation_missing": int(
                    (~np.isfinite(soma[:n][:, [SLOT["LeftLeg"], SLOT["RightLeg"],
                                               SLOT[f"{long}Shin"], SLOT[f"{long}Foot"],
                                               SLOT[f"{long}ToeBase"]]]).all(axis=(1, 2))).sum()),
                "soma_triangulation_missing_other_side_for_the_swap_control": int(
                    (~np.isfinite(soma[:n][:, [SLOT[f"{other}Shin"], SLOT[f"{other}Foot"],
                                               SLOT[f"{other}ToeBase"]]]).all(axis=(1, 2))).sum()),
                "shin_parallel_to_pelvic_axis_mamma": int((cond_m < CONDITION_MIN_SIN).sum()),
                "shin_parallel_to_pelvic_axis_delivered": int((cond_d < CONDITION_MIN_SIN).sum()),
                "shin_parallel_to_pelvic_axis_triangulated": int(
                    (np.isfinite(cond_t) & (cond_t < CONDITION_MIN_SIN)).sum()),
                "note": "the reasons OVERLAP; a frame missing from the triangulation is "
                        "not also counted as near-parallel",
            }
            mask = (
                used[:n, subject]
                & np.isfinite(soma[:n][:, [SLOT["LeftLeg"], SLOT["RightLeg"],
                                           SLOT[f"{long}Shin"], SLOT[f"{long}Foot"],
                                           SLOT[f"{long}ToeBase"]]]).all(axis=(1, 2))
                & (cond_m >= CONDITION_MIN_SIN)
                & (cond_d >= CONDITION_MIN_SIN)
                & (np.nan_to_num(cond_t) >= CONDITION_MIN_SIN)
                & np.isfinite(f_ref).all(axis=1) & np.isfinite(f_del).all(axis=1)
                & np.isfinite(f_tri).all(axis=1) & np.isfinite(f_swap).all(axis=1)
                & (np.nan_to_num(cond_o) >= CONDITION_MIN_SIN)
                & (np.nan_to_num(cond_ot) >= CONDITION_MIN_SIN)
                & np.isfinite(f_swap_tri).all(axis=1)
            )
            if mask.sum() < 20:
                subject_block["feet"][side] = {"valid_frames": int(mask.sum()),
                                               "note": "too few valid frames to score"}
                continue

            ref = f_ref[mask]
            rng = np.random.default_rng(SEED + subject)

            # ---------------- controls
            const_world = unit(m_world[mask].mean(axis=0)[None, :])          # constant in WORLD
            c_const_world = unit(np.einsum("nji,nj->ni", F_m[mask],
                                           np.broadcast_to(const_world, (int(mask.sum()), 3))))
            c_welded = np.broadcast_to(unit(ref.mean(axis=0)[None, :]), ref.shape)
            c_shuffled = ref[rng.permutation(int(mask.sum()))]
            c_swap = f_swap[mask]
            c_mirror = f_del[mask] * np.array([-1.0, 1.0, 1.0])              # negate anterior

            arms = {
                "ours_delivered": f_del[mask],
                "ours_triangulated_toebase": f_tri[mask],
                "ORACLE_mamma_foot_in_our_delivered_shin_frame": f_or_del[mask],
                "ORACLE_mamma_foot_in_our_triangulated_shin_frame": f_or_tri[mask],
                "CONTROL_constant_axis_in_world": c_const_world,
                "CONTROL_welded_to_shin_zero_articulation": c_welded,
                "CONTROL_time_shuffled_mamma": c_shuffled,
                "CONTROL_left_right_swap_delivered": c_swap,
                "CONTROL_left_right_swap_triangulated": f_swap_tri[mask],
                "CONTROL_mirrored_anterior_axis": c_mirror,
            }
            blocks, series = {}, {}
            for name, arm in arms.items():
                blocks[name], series[name] = compare(arm, ref, side)

            boot = bootstrap(series, ["ours_delivered", "ours_triangulated_toebase"])
            verdicts = adjudicate(blocks, boot)
            bar_spread = range_of_motion(ref, side)[
                "spread_about_take_mean_direction_deg"]["median_deg"]

            subject_block["feet"][side] = {
                "valid_frames": int(mask.sum()),
                "frames_considered": int(n),
                "frames_dropped_by_reason_not_disjoint": reasons,
                "THE_BAR_mamma_foot_range_of_motion_in_its_own_shin_frame":
                    range_of_motion(ref, side),
                "ours_delivered_range_of_motion": range_of_motion(f_del[mask], side),
                "ours_triangulated_range_of_motion": range_of_motion(f_tri[mask], side),
                "arms": blocks,
                "residual_over_bar_ratio": {
                    arm: (blocks[arm]["spread_after_offset_removed_deg"]["median_deg"]
                          / bar_spread)
                    for arm in ("ours_delivered", "ours_triangulated_toebase")
                } | {"what_it_is": "each arm's offset-removed median residual divided by "
                                   "MAMMA's OWN range of motion about its take mean. "
                                   "Descriptive, never a gate: it is a ratio of two "
                                   "numbers that answer different questions, and a small "
                                   "value is not accuracy."},
                "where_the_constant_offset_lives": where_it_lives(blocks),
                "verdicts": verdicts,
                "bootstrap": boot,
            }

        gaps = [abs(e["arms"]["ours_delivered"]["spread_after_offset_removed_deg"]["median_deg"]
                    - e["arms"]["ours_triangulated_toebase"]["spread_after_offset_removed_deg"]["median_deg"])
                for e in subject_block["feet"].values() if "arms" in e]
        subject_block["plan_corrections"]["delivered_minus_triangulated_median_deg"] = {
            "min": float(min(gaps)), "max": float(max(gaps)),
            "note": "absolute difference between the two arms' offset-removed medians, "
                    "over this subject's two feet"} if gaps else None
        report["subjects"][f"subject_{subject:02d}"] = subject_block

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    summarise(report)
    print(f"\nwrote {OUT}")


def summarise(report: dict) -> None:
    for subject, block in report["subjects"].items():
        for side, entry in block["feet"].items():
            if "note" in entry:
                print(f"\n=== {subject} foot {side}: {entry['note']} ==="); continue
            bar = entry["THE_BAR_mamma_foot_range_of_motion_in_its_own_shin_frame"]
            print(f"\n=== {subject} foot {side}  ({entry['valid_frames']} valid of "
                  f"{entry['frames_considered']}) ===")
            print(f"  THE BAR -- MAMMA's foot in its own shin frame:")
            print(f"    dorsi/plantarflexion phi  median {bar['dorsi_plantarflexion_phi']['median_deg']:7.2f}"
                  f"  p05-p95 {bar['dorsi_plantarflexion_phi']['p05_deg']:7.2f} .. "
                  f"{bar['dorsi_plantarflexion_phi']['p95_deg']:7.2f}  range "
                  f"{bar['dorsi_plantarflexion_phi']['range_deg']:6.2f}")
            print(f"    ab/adduction psi (medial+) median {bar['ab_adduction_psi_medial_positive']['median_deg']:7.2f}"
                  f"  p05-p95 {bar['ab_adduction_psi_medial_positive']['p05_deg']:7.2f} .. "
                  f"{bar['ab_adduction_psi_medial_positive']['p95_deg']:7.2f}  range "
                  f"{bar['ab_adduction_psi_medial_positive']['range_deg']:6.2f}")
            print(f"    spread about take mean    median "
                  f"{bar['spread_about_take_mean_direction_deg']['median_deg']:7.2f}  p95 "
                  f"{bar['spread_about_take_mean_direction_deg']['p95_deg']:7.2f}   "
                  f"frame-to-frame median "
                  f"{bar['frame_to_frame_travel_deg']['median_deg']:.2f}")
            for cand, v in entry["verdicts"].items():
                print(f"  VERDICT {cand:34s} G1 {'pass' if v['G1_pass'] else 'FAIL'} "
                      f"(offset {v['G1_mean_direction_offset_deg']:5.1f} deg)  "
                      f"G2 vs welded P={v['G2_beats_welded_P_min_over_blocks']:.3f} "
                      f"{'pass' if v['G2_pass'] else 'FAIL'}  "
                      f"controls rejected: {v['all_controls_rejected']}  -- {v['verdict']}")
            print("  " + entry["where_the_constant_offset_lives"]["reading"].replace(
                ", ", ",\n    "))
            boot = entry["bootstrap"]["20"]
            print(f"  {'arm':52s} {'raw':>8s} {'offset':>8s} {'spread':>8s} {'p95':>8s} "
                  f"{'P(del beats)':>13s}")
            for name, arm in entry["arms"].items():
                p = boot["ours_delivered"]["P_beats"].get(name)
                flag = "" if arm["G1_offset_within_90deg"] else "  G1-FAIL"
                print(f"  {name:52s} {arm['raw_angle_no_offset_removed_deg']['median_deg']:8.2f} "
                      f"{arm['mean_direction_offset_deg']:8.2f} "
                      f"{arm['spread_after_offset_removed_deg']['median_deg']:8.2f} "
                      f"{arm['spread_after_offset_removed_deg']['p95_deg']:8.2f} "
                      f"{('' if p is None else f'{p:13.3f}')}{flag}")


if __name__ == "__main__":
    main()
