#!/usr/bin/env python3
"""What does stage 6 cost -- the positions-to-rotations retarget -- and how much of
that is the BODY, how much the CONVERTER?

`positions_to_body_track` consumes only bone DIRECTIONS and rebuilds the body on
DETAILED_HUMANOID's **canonical bone lengths**. The performer's own limb lengths
are discarded. So every delivered joint inherits the difference between this rig's
proportions and the subject's -- permanently, in every frame, however good the
detector becomes. This sizes that, isolating the stage by feeding it the
pipeline's own triangulated positions and reading what comes back out.

Two things had to be got right, and both were found by a control that failed:

1. The stage is NOT just a retarget. It ends in `project_generated_foot_contacts`
   (ground_band 0.08 m, maximum_root_correction_m 0.08), which plants the character
   on the floor. That is a deliberate global vertical shift -- ~90 mm here -- and
   it is not error. Everything below is therefore scored ROOT-RELATIVE.

2. "Hips must return the pelvis" is not a usable control, because root-relative it
   is identically zero. The control that CAN fail is a ROUND TRIP: take the rig's
   own output, forward-kinematic it into landmark positions, and feed those back
   in. That body has canonical proportions BY CONSTRUCTION, so the converter must
   reproduce it. Whatever it costs there is the converter's own approximation; the
   excess on a real performer is the proportion mismatch. It is the ORACLE arm:
   0.00 mm on the legs, and 67-79 mm on the arms since D2 (2026-09-02).

   THE ARM FIGURE IS NOT THE CLAVICLE'S COST, and before D2 it was not either.
   It read 36-47 mm while the clavicle direction was measured from `pelvis + 0.72 *
   torso_up`; D2 measures it from the rig's own Shoulder origin, and this arm went
   UP to 67-79 while the delivered configuration went DOWN, 181/218 -> 124/90 mm.
   The reason is in this construction, not in the converter: `landmarks_from_fk`
   below writes the rig's UpperLeg origins into the `left_hip`/`right_hip` slots --
   anatomically right, they are the femoral joint centres -- and the converter puts
   `Hips` on that midpoint, 80 mm higher. So the SECOND solve sees the rig 80 mm
   lower against the same landmark cloud. Every direction measured
   landmark-to-landmark is blind to a translation, and this score is root-relative,
   which is why the legs read 0.00 here and always did; that was never evidence the
   root placement is right. The clavicle is the only direction measured in the RIG's
   frame, so it is the only one that can see it. Take the rig's own hip drop out of
   the re-solve's origin and this arm reads 0.60/0.22 mm.
   tools/compare/d2_clavicle_gate.py, docs/reviews/clavicle-origin-2026-09-02.md.

THE SPLIT (ladder step I1). Three questions, three families of arm:

  * how much is the CONVERTER?      the round-trip oracle, per skeleton
  * how much is PROPORTIONS?        canonical minus the RE-SOLVED sized arm
  * is the converter's cost a fact about OUR CAPTURE, or about the converter?
                                    arm B: MAMMA's own `pred_joints` in, through
                                    the scoreboard's PAIRS, scored against
                                    themselves

**Arm B and the own-capture arms never share an axis.** They score the converter
against two different inputs -- different bodies, different poses, a different
joint convention (SMPL-X regressor vs SOMA-77 adapter). "MAMMA's arms cost more
than ours" would be a claim about the input population, not about the converter.
Each figure carries its reference; there is no combined number.

Run:  python3 tools/swap-harness/retarget_cost.py      (SYSTEM python3, not .venv)
Writes: artifacts/compare/retarget-cost.json
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))
from autoanim_gnm import commercial_multiview as cm  # noqa: E402
from autoanim_gnm.body import DETAILED_HUMANOID, forward_kinematics_positions  # noqa: E402
import subject_map  # noqa: E402

# `subject_map.MA3D` is written relative to the process cwd. Anchor it to the repo
# so this runs from anywhere; a stale relative path would silently resolve elsewhere.
MA3D = ROOT / "artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/pushing_and_lifting_from_ground"
subject_map.MA3D = MA3D
from sized_skeleton import sized_skeleton  # noqa: E402

TRACKS = ROOT / "artifacts/commercial-multiview-soma77"
OUT = ROOT / "artifacts/compare/retarget-cost.json"
# D1 (fix): `--tracks`/`--out` point this at a rebuild without touching the delivery.
if "--tracks" in sys.argv:
    TRACKS = ROOT / sys.argv[sys.argv.index("--tracks") + 1]
if "--out" in sys.argv:
    OUT = ROOT / sys.argv[sys.argv.index("--out") + 1]

# Which rig joint carries which captured landmark, read off the chain table in
# positions_to_body_track: each chain rotates the parent so the child's REST
# offset points along the measured direction, so the CHILD's origin is the landmark.
RIG_FOR = {
    "neck": "Neck",
    "left_shoulder": "LeftUpperArm",  "right_shoulder": "RightUpperArm",
    "left_elbow": "LeftLowerArm",     "right_elbow": "RightLowerArm",
    "left_wrist": "LeftHand",         "right_wrist": "RightHand",
    "left_hip": "LeftUpperLeg",       "right_hip": "RightUpperLeg",
    "left_knee": "LeftLowerLeg",      "right_knee": "RightLowerLeg",
    "left_ankle": "LeftFoot",         "right_ankle": "RightFoot",
}
GROUPS = {
    "arms": ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist"),
    "legs": ("left_hip", "right_hip", "left_knee", "right_knee",
             "left_ankle", "right_ankle"),
    "torso": ("neck",),
}
# rig parent -> rig child for the eight limb bones, used by the INTEGRITY check
# (a converter output has rest-constant bone lengths; copied positions do not).
RIG_BONES = (("LeftUpperArm", "LeftLowerArm"), ("LeftLowerArm", "LeftHand"),
             ("RightUpperArm", "RightLowerArm"), ("RightLowerArm", "RightHand"),
             ("LeftUpperLeg", "LeftLowerLeg"), ("LeftLowerLeg", "LeftFoot"),
             ("RightUpperLeg", "RightLowerLeg"), ("RightLowerLeg", "RightFoot"))
BONES = [("left_shoulder", "left_elbow", "LeftLowerArm"), ("left_elbow", "left_wrist", "LeftHand"),
         ("right_shoulder", "right_elbow", "RightLowerArm"), ("right_elbow", "right_wrist", "RightHand"),
         ("left_hip", "left_knee", "LeftLowerLeg"), ("left_knee", "left_ankle", "LeftFoot"),
         ("right_hip", "right_knee", "RightLowerLeg"), ("right_knee", "right_ankle", "RightFoot")]
# Rig child whose rest offset IS that bone, paired with the limb the pipeline
# already measures per subject per take in `estimate_limb_lengths_m`.
SCALE_FROM = {
    "LeftLowerArm": ("left_shoulder", "left_elbow"), "LeftHand": ("left_elbow", "left_wrist"),
    "RightLowerArm": ("right_shoulder", "right_elbow"), "RightHand": ("right_elbow", "right_wrist"),
    "LeftLowerLeg": ("left_hip", "left_knee"), "LeftFoot": ("left_knee", "left_ankle"),
    "RightLowerLeg": ("right_hip", "right_knee"), "RightFoot": ("right_knee", "right_ankle"),
}
TORSO_CHAIN = ("Spine", "Chest", "UpperChest", "Neck")
SHA = "0" * 64

# our 19-joint name -> SMPL-X joint index in MAMMA's `pred_joints`. Copied from
# tools/compare/mamma_scoreboard.py so the two instruments cannot drift apart; the
# scoreboard is the owner of this table.
PAIRS = {
    "root": 0, "neck": 12, "nose": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
    "left_hip": 1, "right_hip": 2,
    "left_knee": 4, "right_knee": 5,
    "left_ankle": 7, "right_ankle": 8,
}
UNMAPPED = tuple(n for n in cm.JOINT_NAMES if n not in PAIRS)   # eyes and ears

BLOCK = 15          # moving-block bootstrap: 10 blocks over 150 frames
DRAWS = 2000
RNG_SEED = 20260902


# --------------------------------------------------------------------------- basis
Z_UP_FROM_Y_UP = lambda p: np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)
Y_UP_FROM_Z_UP = lambda p: np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def scaled_skeleton(src_z_up, torso=True):
    """DETAILED_HUMANOID with this performer's OWN limb bone lengths substituted in.

    Retained from the original instrument as a *variant* of the sized arm: it scales
    only the eight limb bones (and optionally the spine chain), leaving the shoulder
    and hip spans canonical, so limb length and span stay separable. The headline
    sized arm below uses `tools/head/sized_skeleton.py` -- the same sizing the
    scoreboard uses -- so the two instruments can be read against each other.
    The clavicle anchor defect is deliberately LEFT IN in both.
    """
    lengths = cm.estimate_limb_lengths_m(src_z_up)
    joints = list(DETAILED_HUMANOID.joints)
    for name, key in SCALE_FROM.items():
        if key not in lengths:
            continue
        i = DETAILED_HUMANOID.index(name)
        rest = np.asarray(joints[i].rest_translation_m, dtype=np.float64)
        n = np.linalg.norm(rest)
        if n < 1e-9:
            continue
        joints[i] = dataclasses.replace(
            joints[i], rest_translation_m=tuple(rest / n * lengths[key]))
    pelvis = 0.5 * (src_z_up[:, cm.JOINT_INDEX["left_hip"]] + src_z_up[:, cm.JOINT_INDEX["right_hip"]])
    span = float(np.median(np.linalg.norm(src_z_up[:, cm.JOINT_INDEX["neck"]] - pelvis, axis=1)))
    canon = sum(float(np.linalg.norm(DETAILED_HUMANOID.joints[DETAILED_HUMANOID.index(n)].rest_translation_m))
                for n in TORSO_CHAIN)
    # The scoring origin is the HIP MIDPOINT, which hangs below Hips by the upper-leg
    # rest offset, while the chain sum measures Hips->Neck. Scaling by span/canon
    # therefore forces hipmid->Neck = span + drop, a guaranteed overshoot -- it showed
    # up as the +torso arm's neck median sitting at exactly 80.00 mm on both subjects,
    # and it made torso scaling look harmful when it is not. Caught by Fable.
    drop = -0.5 * (np.asarray(DETAILED_HUMANOID.joints[DETAILED_HUMANOID.index("LeftUpperLeg")].rest_translation_m)[1]
                   + np.asarray(DETAILED_HUMANOID.joints[DETAILED_HUMANOID.index("RightUpperLeg")].rest_translation_m)[1])
    if canon > 1e-9 and torso:
        k = (span - drop) / canon
        for n in TORSO_CHAIN:
            i = DETAILED_HUMANOID.index(n)
            joints[i] = dataclasses.replace(
                joints[i], rest_translation_m=tuple(np.asarray(joints[i].rest_translation_m) * k))
    return dataclasses.replace(DETAILED_HUMANOID, joints=tuple(joints))


# --------------------------------------------------------------------------- the stage
def solve(src_z_up, skeleton=None):
    """Run the REAL converter. Returns the BodyTrack it produced.

    D3, 2026-09-03. The skeleton is now passed EXPLICITLY, which is the supported way
    since `positions_to_body_track` takes one; the module rebind is kept beside it
    because `_joint_origin` and `_set_world` still read `cm.DETAILED_HUMANOID` for joint
    names and parents, and because an instrument that substitutes those functions relies
    on the rebind being in force.

    TWO CONSEQUENCES FOR THE SIZED ARM, both reported and neither chased. The returned
    track now CARRIES the skeleton it was solved on, so (a) `project_generated_foot_contacts`
    inside the converter hoists on the sized rest instead of the canonical one -- before
    D3 the rebind never reached it, because `body_projection` reads `DETAILED_HUMANOID`
    from its own namespace -- and (b) the sized track can no longer be validated against
    the canonical body, which is the point.
    """
    skel = skeleton or DETAILED_HUMANOID
    saved = cm.DETAILED_HUMANOID
    cm.DETAILED_HUMANOID = skel                    # the retarget reads it from its own namespace
    try:
        return cm.positions_to_body_track(
            src_z_up, sample_rate_hz=30, provenance_sha256=SHA, skeleton=skel)
    finally:
        cm.DETAILED_HUMANOID = saved


def fk_of(track, skeleton=None):
    skel = skeleton or DETAILED_HUMANOID
    return forward_kinematics_positions(
        np.asarray(track.root_translation_m, dtype=np.float64),
        np.asarray(track.local_rotations_xyzw, dtype=np.float64),
        skeleton=skel).astype(np.float64)


def retarget_then_fk(src_z_up, skeleton=None):
    return fk_of(solve(src_z_up, skeleton), skeleton)


def score(fk, ref_y_up, skeleton=None):
    """Root-relative per-landmark error, mm. Removes the ground-projection shift.

    The alignment removes TRANSLATION ONLY -- deliberately. A rotational alignment
    would hide a facing error; the yaw-180 control below demonstrates both halves.
    """
    skel = skeleton or DETAILED_HUMANOID
    fk_o = 0.5 * (fk[:, skel.index("LeftUpperLeg")] + fk[:, skel.index("RightUpperLeg")])
    rf_o = 0.5 * (ref_y_up[:, cm.JOINT_INDEX["left_hip"]] + ref_y_up[:, cm.JOINT_INDEX["right_hip"]])
    out = {}
    for name, rig in RIG_FOR.items():
        a = fk[:, skel.index(rig)] - fk_o
        b = ref_y_up[:, cm.JOINT_INDEX[name]] - rf_o
        out[name] = np.linalg.norm(a - b, axis=1) * 1000
    return out


def score_procrustes(fk, ref_y_up, skeleton=None):
    """Same, but each frame is additionally rotated onto the reference (Kabsch).

    Not the instrument -- the demonstration that an alignment which absorbs rotation
    CANNOT see a facing error. Used only by the yaw-180 control.
    """
    skel = skeleton or DETAILED_HUMANOID
    names = list(RIG_FOR)
    A = np.stack([fk[:, skel.index(RIG_FOR[n])] for n in names], axis=1)
    B = np.stack([ref_y_up[:, cm.JOINT_INDEX[n]] for n in names], axis=1)
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    out = {n: np.zeros(len(A)) for n in names}
    for f in range(len(A)):
        u, _, vt = np.linalg.svd(A[f].T @ B[f])
        d = np.sign(np.linalg.det(u @ vt))
        r = u @ np.diag([1.0, 1.0, d]) @ vt
        err = np.linalg.norm(A[f] @ r - B[f], axis=1) * 1000
        for i, n in enumerate(names):
            out[n][f] = err[i]
    return out


def landmarks_from_fk(fk, skeleton=None):
    """Rebuild a 19-joint observation set from a rig FK pose (the round-trip input)."""
    skel = skeleton or DETAILED_HUMANOID
    out = np.zeros((len(fk), len(cm.JOINT_NAMES), 3))
    for name, rig in RIG_FOR.items():
        out[:, cm.JOINT_INDEX[name]] = fk[:, skel.index(rig)]
    head = fk[:, skel.index("Head")]
    for name in cm.JOINT_NAMES:                      # unused by the retarget, must be finite
        i = cm.JOINT_INDEX[name]
        if not out[:, i].any():
            out[:, i] = head
    out[:, cm.JOINT_INDEX["root"]] = 0.5 * (out[:, cm.JOINT_INDEX["left_hip"]] + out[:, cm.JOINT_INDEX["right_hip"]])
    return out


# --------------------------------------------------------------------------- arm B adapter
def adapt_mamma(pred_joints, filler="nose"):
    """MAMMA's `pred_joints` -> our 19-joint observation contract, in capture Z-up.

    `positions_to_body_track` REJECTS a non-finite input outright
    (`not np.isfinite(source).all()` -> CommercialMultiviewError), so the four joints
    PAIRS does not cover (left/right eye, left/right ear) cannot be left NaN. They are
    filled with a finite placeholder instead. The converter never reads them -- proved
    empirically, not by inspection: `filler_is_inert` below re-runs the whole solve
    with a different placeholder and requires bit-identical rotations.

    MAMMA's world frame IS the camera-rig world frame (tools/head/subject_map.py), so
    no change of basis is applied here; the converter does its own Z-up -> Y-up.
    """
    out = np.zeros((len(pred_joints), len(cm.JOINT_NAMES), 3), dtype=np.float64)
    for name, mi in PAIRS.items():
        out[:, cm.JOINT_INDEX[name]] = pred_joints[:, mi]
    fill = out[:, cm.JOINT_INDEX["nose"]] if filler == "nose" else out[:, cm.JOINT_INDEX["root"]]
    for name in UNMAPPED:
        out[:, cm.JOINT_INDEX[name]] = fill
    return out


def nan_tolerance_probe(src_z_up):
    """Does the converter accept NaN in the four unmapped slots? Recorded, not assumed."""
    probe = src_z_up.copy()
    for name in UNMAPPED:
        probe[:, cm.JOINT_INDEX[name]] = np.nan
    try:
        solve(probe)
    except Exception as exc:                                    # noqa: BLE001 -- recording it
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def filler_is_inert(pred_joints):
    """Two different placeholders in the four unmapped slots -> identical solve?"""
    a = solve(adapt_mamma(pred_joints, filler="nose"))
    b = solve(adapt_mamma(pred_joints, filler="root"))
    return (np.array_equal(np.asarray(a.local_rotations_xyzw), np.asarray(b.local_rotations_xyzw))
            and np.array_equal(np.asarray(a.root_translation_m), np.asarray(b.root_translation_m)))


# --------------------------------------------------------------------------- integrity
def integrity(fk, track, skeleton=None):
    """Is this a RIG, or a bag of points wearing one?

    A converter output stands on the skeleton's rest lengths: every bone is exactly
    its rest length in every frame, so the std across frames is 0 and the deviation
    from rest is 0. Copied input positions match neither -- and on a SIZED skeleton
    they nearly match the rest lengths (the rest lengths ARE the performer's medians),
    so **constancy, not the rest length, is the discriminant**. Both are reported.
    """
    skel = skeleton or DETAILED_HUMANOID
    std_mm, dev_mm = [], []
    for parent, child in RIG_BONES:
        a, b = fk[:, skel.index(parent)], fk[:, skel.index(child)]
        length = np.linalg.norm(b - a, axis=1) * 1000.0
        ok = np.isfinite(length)
        if not ok.any():
            continue
        rest = np.linalg.norm(skel.joints[skel.index(child)].rest_translation_m) * 1000.0
        std_mm.append(float(np.std(length[ok])))
        dev_mm.append(float(np.max(np.abs(length[ok] - rest))))
    out = {"bone_length_std_mm_max": round(max(std_mm), 3) if std_mm else None,
           "bone_length_dev_from_rest_mm_max": round(max(dev_mm), 3) if dev_mm else None}
    if track is None:
        out["rotations_present"] = False
        out["unit_quaternion_max_error"] = None
    else:
        q = np.asarray(track.local_rotations_xyzw, dtype=np.float64)
        out["rotations_present"] = True
        out["unit_quaternion_max_error"] = round(
            float(np.max(np.abs(np.linalg.norm(q, axis=-1) - 1.0))), 12)
    return out


# --------------------------------------------------------------------------- statistics
def summarise(err, valid=None):
    """median / p95 per landmark, per group and overall, in mm."""
    sel = slice(None) if valid is None else valid
    per_joint = {n: {"median_mm": round(float(np.median(v[sel])), 2),
                     "p95_mm": round(float(np.percentile(v[sel], 95)), 2)}
                 for n, v in err.items()}
    out = {"per_landmark": per_joint, "per_group": {}}
    for g, names in GROUPS.items():
        pooled = np.concatenate([err[n][sel] for n in names])
        out["per_group"][g] = {"median_mm": round(float(np.median(pooled)), 2),
                               "p95_mm": round(float(np.percentile(pooled, 95)), 2)}
    pooled = np.concatenate([err[n][sel] for n in RIG_FOR])
    out["overall"] = {"median_mm": round(float(np.median(pooled)), 2),
                      "p95_mm": round(float(np.percentile(pooled, 95)), 2),
                      "mean_mm": round(float(np.mean(pooled)), 2),
                      "note": "the landmark population is bimodal (legs ~25 mm, arms ~180 mm); "
                              "read the per-group figures, not the pooled ones"}
    return out


def group_frame_series(err, group):
    """[frame] mean error over the group's landmarks -- the series the bootstrap resamples."""
    return np.mean(np.stack([err[n] for n in GROUPS[group]], axis=1), axis=1)


def lag1(series):
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    return round(float(np.dot(x[:-1], x[1:]) / denom), 4) if denom > 1e-12 else None


def block_bootstrap_margin(err_a, err_b, group, rng):
    """Median(A) - median(B) per group, moving-block, CANDIDATE AND CONTROL ON THE
    SAME DRAWN FRAMES. One take, 150 frames, lag-1 autocorrelation near 1: an
    ordinary resample would be invalid and a point estimate would be a claim
    the data cannot carry."""
    names = GROUPS[group]
    A = np.stack([err_a[n] for n in names], axis=1)
    B = np.stack([err_b[n] for n in names], axis=1)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    nblocks = max(1, n // BLOCK)
    starts_max = max(1, n - BLOCK + 1)
    diffs = np.empty(DRAWS)
    for d in range(DRAWS):
        s = rng.integers(0, starts_max, size=nblocks)
        idx = np.concatenate([np.arange(t, min(t + BLOCK, n)) for t in s])
        diffs[d] = np.median(A[idx]) - np.median(B[idx])
    return {"median_mm": round(float(np.median(diffs)), 2),
            "ci95_mm": [round(float(np.percentile(diffs, 2.5)), 2),
                        round(float(np.percentile(diffs, 97.5)), 2)],
            "p_wrong_sign": round(float(np.mean(diffs <= 0.0) if np.median(diffs) > 0
                                       else np.mean(diffs >= 0.0)), 4),
            "block_frames": BLOCK, "draws": DRAWS}


# --------------------------------------------------------------------------- controls
def permute_labels(src_z_up):
    """Wrong joint permutation: cyclic shift of the 13 landmarks the converter reads."""
    out = src_z_up.copy()
    names = list(RIG_FOR)
    for i, name in enumerate(names):
        out[:, cm.JOINT_INDEX[name]] = src_z_up[:, cm.JOINT_INDEX[names[(i + 1) % len(names)]]]
    return out


def swap_left_right(src_z_up):
    out = src_z_up.copy()
    for name in cm.JOINT_NAMES:
        if name.startswith("left_"):
            other = "right_" + name[5:]
            out[:, cm.JOINT_INDEX[name]] = src_z_up[:, cm.JOINT_INDEX[other]]
            out[:, cm.JOINT_INDEX[other]] = src_z_up[:, cm.JOINT_INDEX[name]]
    return out


def copy_through_fk(src_z_up, skeleton=None):
    """The degenerate 'converter': put the input positions where the rig joints go,
    emit no rotations. Scores 0.00 mm positionally and is not a rig."""
    skel = skeleton or DETAILED_HUMANOID
    ref = Y_UP_FROM_Z_UP(src_z_up)
    fk = np.full((len(ref), len(skel.joints), 3), np.nan)
    for name, rig in RIG_FOR.items():
        fk[:, skel.index(rig)] = ref[:, cm.JOINT_INDEX[name]]
    return fk


def yaw180(fk, skeleton=None):
    """Rotate the whole output 180 deg about the rig's vertical (+Y) through the hip
    midpoint: a body that faces backwards and is otherwise perfect."""
    skel = skeleton or DETAILED_HUMANOID
    o = 0.5 * (fk[:, skel.index("LeftUpperLeg")] + fk[:, skel.index("RightUpperLeg")])
    out = fk - o[:, None, :]
    out[..., 0] *= -1.0
    out[..., 2] *= -1.0
    return out + o[:, None, :]


# --------------------------------------------------------------------------- provenance
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


REF_OURS = ("our own triangulated capture, root-relative (hip midpoint), "
            "13 landmarks the converter reads -- the converter scored against its OWN input")
REF_MAMMA = ("MAMMA pred_joints through the scoreboard's PAIRS, root-relative (hip midpoint), "
             "13 landmarks -- the converter scored against ITS OWN MAMMA input")

BLIND_TO = (
    "This scores the converter against its OWN input, so it cannot see detector error, "
    "triangulation depth error, or any mislabelling that is consistent between input and "
    "reference -- the L/R control measures the instrument's sensitivity, not the pipeline's "
    "correctness. It scores joint positions BY NAME, so a rig whose 'Left' bones sit on the "
    "mesh's right passes identically and D1's 180-degree facing yaw is invisible (the legs "
    "round-trip at 0.00 mm either way); the yaw-180 control proves only that the ALIGNMENT "
    "does not hide facing. It removes translation, so the deliberate ~90 mm ground projection "
    "and any root-placement defect are outside it. It says nothing about the four inputs the "
    "converter does not consume through these chains (head orientation, foot orientation, toes, "
    "hands). On OUR capture it cannot separate the landmark-to-joint-origin convention offset "
    "from the input's own non-rigidity -- our triangulated bones wander tens of mm across "
    "frames and a constant-length rig cannot follow that, so the residual after sizing is a "
    "sum of the two; only a rigid input (arm B) or physical markers can split them. Arm B's "
    "figures and the own-capture figures are the converter's cost on two different input "
    "populations and are NOT differenceable."
)


def arm(err, track, fk, skeleton, reference, note=""):
    d = summarise(err)
    d["reference"] = reference
    d["integrity"] = integrity(fk, track, skeleton)
    if note:
        d["note"] = note
    return d


def run_subject(subject, mamma_body_id, report, rng):
    src = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
    finite = np.isfinite(src).all(axis=(1, 2))
    src = src[finite]
    ref_y = Y_UP_FROM_Z_UP(src)

    arms: dict = {}
    err_cache: dict = {}

    def add(key, err, track, fk, skel, reference, note=""):
        arms[key] = arm(err, track, fk, skel, reference, note)
        err_cache[key] = err

    # ---- own capture, canonical
    t_can = solve(src)
    fk_can = fk_of(t_can)
    add("performer_canonical", score(fk_can, ref_y), t_can, fk_can, DETAILED_HUMANOID, REF_OURS,
        "converter + proportion mismatch, the delivered configuration")

    # ---- ORACLE: canonical round trip
    synth = landmarks_from_fk(fk_can)
    t_rt = solve(Z_UP_FROM_Y_UP(synth))
    fk_rt = fk_of(t_rt)
    add("ORACLE_roundtrip_canonical", score(fk_rt, synth), t_rt, fk_rt, DETAILED_HUMANOID,
        "a body with canonical proportions BY CONSTRUCTION -- the converter's own cost",
        "must reproduce 0.00 mm on the legs. The ARM figure is 67-79 mm since D2 and it "
        "is this CONSTRUCTION's cost, not the converter's: the hip landmarks fed back are "
        "the rig's UpperLeg origins while the converter puts Hips on their midpoint, "
        "80 mm higher, so the re-solve sits 80 mm low and the clavicle -- the one "
        "direction measured in the rig's own frame -- turns. Remove the rig's hip drop "
        "from the re-solve's origin and it reads 0.60/0.22 mm. See the module docstring "
        "and tools/compare/d2_clavicle_gate.py.")

    # ---- own capture, RE-SOLVED on the sized skeleton (scoreboard's sizing)
    skel_sz, limbs = sized_skeleton(DETAILED_HUMANOID, src)
    t_sz = solve(src, skel_sz)
    fk_sz = fk_of(t_sz, skel_sz)
    add("performer_sized_resolved", score(fk_sz, ref_y, skel_sz), t_sz, fk_sz, skel_sz, REF_OURS,
        "positions_to_body_track RE-RUN on the sized skeleton -- not FK of canonical rotations")

    # ---- the scoreboard's method, for comparison: canonical ROTATIONS, sized skeleton
    fk_lb = fk_of(t_can, skel_sz)
    add("performer_sized_fk_only_scoreboard_method", score(fk_lb, ref_y, skel_sz), t_can, fk_lb,
        skel_sz, REF_OURS,
        "the scoreboard's sized arm: canonical rotations replayed on a sized skeleton, "
        "never re-solved -- reported here to test whether it is the lower bound the plan claims")
    # The plan says the scoreboard's sized arm is a LOWER BOUND on a re-solved one. It is
    # not: `positions_to_body_track` builds every rotation by aligning a rest offset's
    # DIRECTION to a measured direction, and sizing scales rest offsets without turning
    # them, so re-solving on a sized skeleton returns the identical rotations. Asserted,
    # not inferred from two medians that happened to match.
    dq = float(np.max(np.abs(np.asarray(t_sz.local_rotations_xyzw, dtype=np.float64)
                             - np.asarray(t_can.local_rotations_xyzw, dtype=np.float64))))
    dr = float(np.max(np.abs(np.asarray(t_sz.root_translation_m, dtype=np.float64)
                             - np.asarray(t_can.root_translation_m, dtype=np.float64))))

    # ---- ORACLE on the sized skeleton (the converter's cost is skeleton-dependent)
    synth_sz = landmarks_from_fk(fk_sz, skel_sz)
    t_rt_sz = solve(Z_UP_FROM_Y_UP(synth_sz), skel_sz)
    fk_rt_sz = fk_of(t_rt_sz, skel_sz)
    add("ORACLE_roundtrip_sized", score(fk_rt_sz, synth_sz, skel_sz), t_rt_sz, fk_rt_sz, skel_sz,
        "a body with the SIZED skeleton's proportions by construction",
        "the converter's own cost on the sized rig. sized_resolved minus this is NOT a "
        "proportion figure: it is the input's own bone-length wander plus the "
        "landmark-to-joint-origin convention offset. See input_bone_length_std_mm_max.")

    # ---- sizing variants: limbs only, limbs + spine (spans left canonical)
    for key, torso in (("performer_sized_limbs_only", False), ("performer_sized_limbs_and_spine", True)):
        sk = scaled_skeleton(src, torso=torso)
        t = solve(src, sk)
        f = fk_of(t, sk)
        add(key, score(f, ref_y, sk), t, f, sk, REF_OURS,
            "variant sizing: shoulder and hip SPANS left canonical")

    # ---- controls, on the own-capture reference
    controls: dict = {}
    t_p = solve(permute_labels(src))
    f_p = fk_of(t_p)
    controls["CONTROL_wrong_joint_permutation"] = arm(
        score(f_p, ref_y), t_p, f_p, DETAILED_HUMANOID, REF_OURS,
        "MUST FAIL: the 13 landmark labels cyclically shifted by one before the converter")
    t_s = solve(swap_left_right(src))
    f_s = fk_of(t_s)
    controls["CONTROL_left_right_swap"] = arm(
        score(f_s, ref_y), t_s, f_s, DETAILED_HUMANOID, REF_OURS,
        "MUST FAIL: left and right landmarks exchanged before the converter, scored unswapped")
    f_c = copy_through_fk(src)
    controls["CONTROL_input_copied_through"] = arm(
        score(f_c, ref_y), None, f_c, DETAILED_HUMANOID, REF_OURS,
        "PASSES the positional score at 0.00 mm and is NOT a rig: no rotations, and bone "
        "lengths vary frame to frame instead of standing at the skeleton's rest lengths. "
        "The positional figure alone cannot reject it; the integrity block is what does.")
    f_c_sz = copy_through_fk(src, skel_sz)
    controls["CONTROL_input_copied_through_sized_rig"] = arm(
        score(f_c_sz, ref_y, skel_sz), None, f_c_sz, skel_sz, REF_OURS,
        "the same degenerate against the SIZED rig, whose rest lengths ARE this performer's "
        "medians: deviation-from-rest nearly passes, and only the across-frame std rejects it")
    f_y = yaw180(fk_can)
    yaw_arm = arm(score(f_y, ref_y), t_can, f_y, DETAILED_HUMANOID, REF_OURS,
                  "MUST FAIL under this instrument's translation-only alignment: a body that "
                  "faces backwards and is otherwise the canonical arm exactly")
    yaw_arm["under_rotational_alignment"] = summarise(score_procrustes(f_y, ref_y))["per_group"]
    yaw_arm["under_rotational_alignment_of_the_UNYAWED_arm"] = summarise(
        score_procrustes(fk_can, ref_y))["per_group"]
    yaw_arm["why"] = ("The two `under_rotational_alignment` blocks are what this control and the "
                      "canonical arm would score if the instrument aligned ROTATION as well as "
                      "translation. They are equal: a per-frame Kabsch fit absorbs the 180-degree "
                      "yaw completely and the facing error vanishes. That is exactly the "
                      "alignment this instrument refuses to use, and why the headline figures "
                      "remove translation only.")
    controls["CONTROL_facing_yaw_180"] = yaw_arm

    # ---- margins that carry a claim
    margins = {}
    for g in GROUPS:
        margins[f"canonical_minus_sized_resolved_{g}"] = block_bootstrap_margin(
            err_cache["performer_canonical"], err_cache["performer_sized_resolved"], g, rng)
        margins[f"sized_resolved_minus_ORACLE_sized_{g}"] = block_bootstrap_margin(
            err_cache["performer_sized_resolved"], err_cache["ORACLE_roundtrip_sized"], g, rng)
    margins["note"] = (
        "moving-block bootstrap, both arms on IDENTICAL drawn frames; positive means the first "
        "arm is worse (lower is better). READ THE SECOND MARGIN CAREFULLY: "
        "`sized_resolved_minus_ORACLE_sized` is what remains after the rig is sized, and it is "
        "not proportions. A constant-length rig cannot follow an input whose own bones change "
        "length frame to frame, and ours do -- see `input_bone_length_std_mm_max`. What remains "
        "is that non-rigidity plus the landmark-to-joint-origin convention offset. Arm B, whose "
        "input is a rigid SMPL-X skeleton, shows the same residual near 3 mm on the legs; that "
        "is the shape of the evidence, NOT a difference to be taken -- the two live on "
        "different references.")
    margins["lag1_autocorrelation"] = {
        g: lag1(group_frame_series(err_cache["performer_canonical"], g)) for g in GROUPS}

    # ---- geometry, for reading the numbers
    span_c = sum(np.linalg.norm(DETAILED_HUMANOID.joints[DETAILED_HUMANOID.index(n)].rest_translation_m)
                 for n in TORSO_CHAIN) * 1000
    span_s = np.median(np.linalg.norm(src[:, cm.JOINT_INDEX["neck"]] - 0.5 * (
        src[:, cm.JOINT_INDEX["left_hip"]] + src[:, cm.JOINT_INDEX["right_hip"]]), axis=1)) * 1000
    bones = {f"{a}->{b}": {
        "canonical_mm": round(float(np.linalg.norm(
            DETAILED_HUMANOID.joints[DETAILED_HUMANOID.index(rc)].rest_translation_m) * 1000), 1),
        "performer_mm": round(float(np.median(np.linalg.norm(
            src[:, cm.JOINT_INDEX[b]] - src[:, cm.JOINT_INDEX[a]], axis=1)) * 1000), 1)}
        for a, b, rc in BONES}

    entry = {
        "frames_scored": int(len(src)),
        "frames_dropped_non_finite": int((~finite).sum()),
        "arms": arms,
        "controls": controls,
        "margins": margins,
        "converter_is_scale_invariant": {
            "max_abs_rotation_difference_sized_vs_canonical_solve": dq,
            "max_abs_root_translation_difference_m": dr,
            "value": bool(dq < 1e-9),
            "finding": (
                "TRUE for isotropic per-bone sizing, which is every sizing in this repo "
                "(an ANISOTROPIC rescale of a chain child would turn its rest offset and "
                "would change them). Re-solving on the sized skeleton returns the SAME "
                "rotations as the canonical solve. The execution plan's premise that the "
                "scoreboard's sized arm (canonical rotations replayed on a sized skeleton) "
                "is a LOWER BOUND on a re-solved sized arm is refuted: the two are the "
                "same arm. Every rotation is built by turning a rest offset's DIRECTION "
                "onto a measured direction, and sizing scales rest offsets without turning "
                "them."
                if dq < 1e-9 else
                "FALSE, and deliberately so since D2 (2026-09-02). It was TRUE while every "
                "rotation aimed a rest offset's DIRECTION, because sizing scales rest "
                "offsets without turning them. D2 measures the clavicle direction from the "
                "rig's own forward-kinematic Shoulder origin, and sizing MOVES that origin, "
                "so the clavicle chain -- and only that chain, six joints -- now re-solves "
                "differently on a sized skeleton. Two consequences the reader must carry: "
                "the scoreboard's `sized` column (canonical rotations REPLAYED on a sized "
                "skeleton) is no longer equivalent to `performer_sized_resolved` below, and "
                "the two are reported separately; and this max is a QUATERNION COMPONENT "
                "difference, which reads ~2 for a mere sign flip -- read the per-joint "
                "angles in artifacts/compare/d2-clavicle/gate.json, not this number."),
        },
        "input_bone_length_std_mm_max": controls["CONTROL_input_copied_through"][
            "integrity"]["bone_length_std_mm_max"],
        "input_bone_length_std_note": "the largest across-frame standard deviation of the eight "
                                      "limb bones AS TRIANGULATED. A rig holds every bone at its "
                                      "rest length exactly, so this is a floor on what any "
                                      "constant-length skeleton can achieve on this input, "
                                      "whatever its proportions.",
        "measured_limbs_mm": limbs,
        "measured_limbs_note": "one median per bone, estimated by estimate_limb_lengths_m on the "
                               "same frames the arms are scored on. Low capacity (8 scalars over "
                               "150 frames), but it is same-frame fitting and is declared as such.",
        "torso_span_pelvis_to_neck_mm": {"canonical": round(float(span_c), 1),
                                         "performer": round(float(span_s), 1)},
        "limb_bones_mm": bones,
    }

    # ---- ARM B: MAMMA's own joints into the converter. SEPARATE REFERENCE.
    pred = np.load(MA3D / f"verts_joints_body_id-{mamma_body_id:02d}.npz",
                   allow_pickle=True)["pred_joints"].astype(np.float64)
    mfinite = np.isfinite(pred).all(axis=(1, 2))
    pred = pred[mfinite]
    m_src = adapt_mamma(pred)
    m_ref_y = Y_UP_FROM_Z_UP(m_src)

    b_arms, b_err = {}, {}
    tb = solve(m_src)
    fb = fk_of(tb)
    b_arms["mamma_joints_canonical"] = arm(score(fb, m_ref_y), tb, fb, DETAILED_HUMANOID, REF_MAMMA,
                                           "the converter priced on MAMMA's joints, canonical rig")
    b_err["mamma_joints_canonical"] = score(fb, m_ref_y)

    m_skel, m_limbs = sized_skeleton(DETAILED_HUMANOID, m_src)
    tbs = solve(m_src, m_skel)
    fbs = fk_of(tbs, m_skel)
    b_arms["mamma_joints_sized_resolved"] = arm(score(fbs, m_ref_y, m_skel), tbs, fbs, m_skel, REF_MAMMA,
                                                "sized to the limb lengths measured from MAMMA's own joints")
    b_err["mamma_joints_sized_resolved"] = score(fbs, m_ref_y, m_skel)

    synth_b = landmarks_from_fk(fb)
    tbr = solve(Z_UP_FROM_Y_UP(synth_b))
    fbr = fk_of(tbr)
    b_arms["ORACLE_roundtrip_from_mamma_solve"] = arm(
        score(fbr, synth_b), tbr, fbr, DETAILED_HUMANOID,
        "a canonical body built from arm B's own output",
        "the converter's cost in MAMMA's pose distribution -- the same construction as the "
        "own-capture oracle, on different poses")
    b_err["ORACLE_roundtrip_from_mamma_solve"] = score(fbr, synth_b)

    b_controls = {}
    tbp = solve(permute_labels(m_src)); fbp = fk_of(tbp)
    b_controls["CONTROL_wrong_joint_permutation"] = arm(
        score(fbp, m_ref_y), tbp, fbp, DETAILED_HUMANOID, REF_MAMMA, "MUST FAIL")
    tbl = solve(swap_left_right(m_src)); fbl = fk_of(tbl)
    b_controls["CONTROL_left_right_swap"] = arm(
        score(fbl, m_ref_y), tbl, fbl, DETAILED_HUMANOID, REF_MAMMA, "MUST FAIL")
    fbc = copy_through_fk(m_src)
    b_controls["CONTROL_input_copied_through"] = arm(
        score(fbc, m_ref_y), None, fbc, DETAILED_HUMANOID, REF_MAMMA,
        "0.00 mm positionally, rejected only by the integrity block")
    fby = yaw180(fb)
    ya = arm(score(fby, m_ref_y), tb, fby, DETAILED_HUMANOID, REF_MAMMA, "MUST FAIL")
    ya["under_rotational_alignment"] = summarise(score_procrustes(fby, m_ref_y))["per_group"]
    b_controls["CONTROL_facing_yaw_180"] = ya

    b_margins = {g: block_bootstrap_margin(b_err["mamma_joints_canonical"],
                                           b_err["mamma_joints_sized_resolved"], g, rng)
                 for g in GROUPS}
    b_margins["note"] = "canonical minus sized-resolved, ON MAMMA'S JOINTS ONLY"

    entry["arm_B_mamma_joints_in"] = {
        "frames_scored": int(len(pred)),
        "mamma_body_id": f"body_id-{mamma_body_id:02d}",
        "arms": b_arms,
        "controls": b_controls,
        "margins": b_margins,
        "adapter": {
            "mapped_joints": len(PAIRS),
            "our_contract_joints": len(cm.JOINT_NAMES),
            "unmapped": list(UNMAPPED),
            "nan_tolerated_by_positions_to_body_track": report["converter_nan_probe"]["tolerated"],
            "nan_rejection": report["converter_nan_probe"]["error"],
            "adapter_used": "the four unmapped joints are filled with the mapped `nose` position "
                            "(a finite placeholder); the converter reads none of them",
            "filler_inertness_verified": report["converter_nan_probe"]["filler_inert"],
        },
        "input_bone_length_std_mm_max": b_controls["CONTROL_input_copied_through"][
            "integrity"]["bone_length_std_mm_max"],
        "input_bone_length_std_note": "MAMMA's pred_joints come off a fitted SMPL-X skeleton and "
                                      "are RIGID; compare with the own-capture figure to see why "
                                      "arm B's sized legs land near 3 mm and ours near 16-20.",
        "measured_limbs_mm": m_limbs,
        "not_comparable_with": "the own-capture arms above -- different input body, different "
                               "poses, different joint convention (SMPL-X regressor vs SOMA-77 "
                               "adapter). Never put them on one axis.",
    }
    return entry


def main():
    if not (TRACKS / "subject-00.body-track.npz").exists():
        raise SystemExit(f"missing tracks under {TRACKS}")
    ours = np.stack([
        np.load(TRACKS / f"subject-{s:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
        for s in (0, 1)])
    mapping = subject_map.mamma_index_for(ours)

    probe_src = ours[0][np.isfinite(ours[0]).all(axis=(1, 2))]
    tolerated, err = nan_tolerance_probe(probe_src)
    pred0 = np.load(MA3D / f"verts_joints_body_id-{mapping[0]:02d}.npz",
                    allow_pickle=True)["pred_joints"].astype(np.float64)

    report = {
        "instrument": "tools/swap-harness/retarget_cost.py",
        "step": "I1 -- split the retarget cost: body proportions vs the converter",
        "regenerate": "python3 tools/swap-harness/retarget_cost.py   (SYSTEM python3)",
        "rig": {"name": "DETAILED_HUMANOID", "joints": len(DETAILED_HUMANOID.joints)},
        "scoring": "root-relative to the hip midpoint, translation only; the stage's deliberate "
                   "~90 mm ground projection is removed and is not error",
        "subject_correspondence": {f"our_{k}": f"body_id-{v:02d}" for k, v in mapping.items()},
        "subject_correspondence_source": "tools/head/subject_map.py, from 3D pelvis agreement "
                                         "(MAMMA's body_id-00 is our subject 1 on this fixture)",
        "converter_nan_probe": {
            "tolerated": tolerated,
            "error": err,
            "claim_in_the_plan": "the execution plan states the function tolerates NaN in the "
                                 "four unmapped joints; it does not -- positions_to_body_track "
                                 "rejects any non-finite input outright",
            "filler_inert": bool(filler_is_inert(pred0)),
        },
        "blind_to": BLIND_TO,
        "input_sha256": {},
        "subjects": {},
    }
    for p in (TRACKS / "subject-00.body-track.npz", TRACKS / "subject-01.body-track.npz",
              MA3D / "verts_joints_body_id-00.npz", MA3D / "verts_joints_body_id-01.npz",
              ROOT / "src/autoanim_gnm/commercial_multiview.py", ROOT / "src/autoanim_gnm/body.py",
              ROOT / "tools/head/sized_skeleton.py", Path(__file__)):
        report["input_sha256"][str(p.relative_to(ROOT))] = sha256(p)

    rng = np.random.default_rng(RNG_SEED)
    for s in (0, 1):
        report["subjects"][f"subject_{s:02d}"] = run_subject(s, mapping[s], report, rng)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    console(report)
    print(f"\nwrote {OUT}")


def console(report):
    print(f"rig DETAILED_HUMANOID, {report['rig']['joints']} joints")
    print(f"{report['scoring']}\n")
    print(f"NaN in the four unmapped joints tolerated by the converter: "
          f"{report['converter_nan_probe']['tolerated']}  "
          f"(filler inert: {report['converter_nan_probe']['filler_inert']})\n")
    for s, d in report["subjects"].items():
        print(f"=== {s}: {d['frames_scored']} frames, reference = OUR OWN CAPTURE ===")
        print(f"{'arm':>46} {'legs':>9} {'arms':>9} {'torso':>9}   (median mm)")
        for key, a in list(d["arms"].items()) + list(d["controls"].items()):
            g = a["per_group"]
            print(f"{key:>46} {g['legs']['median_mm']:>9.2f} {g['arms']['median_mm']:>9.2f} "
                  f"{g['torso']['median_mm']:>9.2f}")
        for key, a in d["controls"].items():
            i = a["integrity"]
            print(f"    integrity {key:>36}: bone std {i['bone_length_std_mm_max']} mm, "
                  f"dev from rest {i['bone_length_dev_from_rest_mm_max']} mm, "
                  f"rotations {i['rotations_present']}")
        print(f"    input bone-length wander (our capture): "
              f"{d['input_bone_length_std_mm_max']} mm std across frames -- a floor on any "
              f"constant-length rig")
        b = d["arm_B_mamma_joints_in"]
        print(f"\n--- arm B: MAMMA {b['mamma_body_id']} joints IN, {b['frames_scored']} frames, "
              f"reference = MAMMA'S OWN JOINTS (separate axis) ---")
        for key, a in list(b["arms"].items()) + list(b["controls"].items()):
            g = a["per_group"]
            print(f"{key:>46} {g['legs']['median_mm']:>9.2f} {g['arms']['median_mm']:>9.2f} "
                  f"{g['torso']['median_mm']:>9.2f}")
        print(f"    input bone-length wander (MAMMA's joints): "
              f"{b['input_bone_length_std_mm_max']} mm std across frames")
        print()


if __name__ == "__main__":
    main()
