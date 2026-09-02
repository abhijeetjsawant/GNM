#!/usr/bin/env python3
"""D2's gate: the clavicle origin, before and after, on one code path.

THE CHANGE. `positions_to_body_track` measured the clavicle DIRECTION from a synthetic
anchor on the torso axis, `pelvis + 0.72 * torso_up`, while `_world_for_bone` turned the
bone about the rig's OWN Shoulder origin, 110 mm out and 110 mm higher. Two origins for
one direction. D2 measures it from `_joint_origin`, the rig's forward-kinematic Shoulder
origin on that frame -- the point the rotation actually pivots about. One constant leaves
and none arrives.

WHAT THIS GATE FOUND, AND IT IS NOT WHAT WAS PRE-REGISTERED. The canonical round-trip
oracle gets WORSE, 41.57/47.05 -> 67.25/79.32 mm, and all of it is a defect the fix
EXPOSES rather than causes:

    `landmarks_from_fk` writes the rig's UpperLeg origins into the `left_hip`/`right_hip`
    slots -- anatomically right, they are the femoral joint centres -- and the converter
    puts `Hips` on that midpoint, 80 mm above where the rig's leg roots are. So the
    round trip's SECOND solve sees the rig sitting 80 mm lower against the same landmark
    cloud than the first did. Every direction measured landmark-to-landmark is blind to
    that (a translation) and the score is root-relative (another translation), which is
    why the legs read 0.00 and always did. The clavicle is the FIRST direction measured
    in the RIG's frame, so it is the first that can see it, and it turns by ~27 deg.

`CONTROL_hip_drop_removed_pass2` swaps the origin helper for the re-solve only and the
round trip collapses to under a millimetre. That control is the attribution: 100 % of the
residual is root placement, which is open D-lane work, and 0 % is the clavicle.

Every band below carries the arm's reference string. Three references appear and never
share an axis: the own-capture round trip ("a canonical-proportioned body BY
CONSTRUCTION"), arm B ("a canonical body built from arm B's own output") and the
scoreboard (MAMMA `pred_joints`). MAMMA REPORTS; IT NEVER SELECTS -- no number here
chooses anything, and this step introduces no constant.

Run:  PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d2_clavicle_gate.py
Writes: artifacts/compare/d2-clavicle/gate.json
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))
sys.path.insert(0, str(ROOT / "tools" / "swap-harness"))
sys.path.insert(0, str(ROOT / "tools" / "compare"))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src."
    )

from autoanim_gnm import commercial_multiview as cm  # noqa: E402
from autoanim_gnm.body import (  # noqa: E402
    DETAILED_HUMANOID,
    _rotate_vector,
    forward_kinematics_positions,
    skeleton_for_joint_names,
)
import retarget_cost as rc  # noqa: E402
import subject_map  # noqa: E402
import mamma_scoreboard as sb  # noqa: E402
from sized_skeleton import sized_skeleton  # noqa: E402

subject_map.MA3D = rc.MA3D
OUT_DIR = ROOT / "artifacts/compare/d2-clavicle"
TRACKS = ROOT / "artifacts/commercial-multiview-soma77"
REBUILD = OUT_DIR / "delivery"
BEFORE_REPORT = OUT_DIR / "retarget-cost-before.json"
AFTER_REPORT = OUT_DIR / "retarget-cost-after.json"

BAND_ARMS_MM = 5.0           # pre-registered, on every one of the six arm landmarks
BAND_LEGS_MM = 0.0           # exact
BAND_ATTRIBUTION_MM = 1.0    # the corrected control's ceiling
SWEEP = [round(0.40 + 0.05 * i, 2) for i in range(17)]   # 0.40 .. 1.20

REF_ROUNDTRIP = "a canonical-proportioned body BY CONSTRUCTION"
REF_ARMB = "a canonical body built from arm B's own output"
REF_MAMMA = "MAMMA pred_joints, per-joint median then median over joints, in capture Z-up metres"
REF_OURS = ("our own triangulated capture, root-relative (hip midpoint) -- the converter "
            "scored against its OWN input")

TRUE_ORIGIN = cm._joint_origin


# ------------------------------------------------------------------ the origin variants
def legacy_anchor(points_y_up: np.ndarray):
    """`pelvis + 0.72 * torso_up`, the pre-D2 anchor, as an origin helper.

    It CANNOT be written against `_joint_origin`'s signature: the anchor needs the LENGTH
    of `torso_up`, and the signature carries the rig, not the landmarks (`root_translation
    + rest["Hips"]` recovers the pelvis and `world` recovers the torso axis's DIRECTION,
    but nothing recovers the span). So the control closes over the converter's own
    Y-up landmark array instead; `frame` indexes it identically because `solve()` hands
    `positions_to_body_track` exactly this array. Deviation from the brief, recorded.
    """

    def helper(world, frame, root_translation, rest, joint_name, scalar=0.72):
        p = points_y_up[frame]
        pelvis = 0.5 * (p[cm.JOINT_INDEX["left_hip"]] + p[cm.JOINT_INDEX["right_hip"]])
        return pelvis + scalar * (p[cm.JOINT_INDEX["neck"]] - pelvis)

    return helper


def make_legacy(points_y_up: np.ndarray):
    return legacy_anchor(points_y_up)


def make_legacy_scalar(scalar: float):
    def factory(points_y_up):
        base = legacy_anchor(points_y_up)
        return lambda w, f, r, rest, n: base(w, f, r, rest, n, scalar=scalar)
    return factory


def make_upperchest(_points_y_up):
    return upperchest_origin


def make_hip_drop_removed(_points_y_up):
    return hip_drop_removed


def upperchest_origin(world, frame, root_translation, rest, joint_name):
    """The clavicle's PARENT's origin -- the plausible off-by-one mistake. Must fail."""
    return TRUE_ORIGIN(world, frame, root_translation, rest, "UpperChest")


def hip_drop_removed(world, frame, root_translation, rest, joint_name):
    """`_joint_origin` with the rig's OWN hip drop taken back out.

    The converter places `Hips` at the captured hip-midpoint, so the rig's leg roots --
    which are what a `left_hip` landmark actually is -- hang `LeftUpperLeg`'s rest
    translation below it. This variant returns the origin the rig WOULD have if the leg
    roots, not `Hips`, sat on that midpoint. It introduces no constant: the offset is the
    skeleton's own UpperLeg rest translation, read from `rest`. Diagnostic only -- it
    measures from a point the bone does not turn about, which is the D2 defect itself in
    miniature, and it is never shipped.
    """
    skeleton = cm.DETAILED_HUMANOID
    drop = 0.5 * (np.asarray(rest["LeftUpperLeg"]) + np.asarray(rest["RightUpperLeg"]))
    return TRUE_ORIGIN(world, frame, root_translation, rest, joint_name) - _rotate_vector(
        world[frame, skeleton.index("Hips")], drop
    )


class origin(object):
    """`with origin(helper):` -- swap the module attribute the converter calls by name."""

    def __init__(self, helper):
        self.helper = helper

    def __enter__(self):
        self.saved = cm._joint_origin
        cm._joint_origin = self.helper

    def __exit__(self, *exc):
        cm._joint_origin = self.saved
        return False


# ------------------------------------------------------------------ round-trip machinery
def _shipped(_points_y_up):
    return TRUE_ORIGIN


def round_trip(src_z_up, skeleton=None, make_pass1=None, make_pass2=None):
    """solve -> FK -> landmarks -> solve. Returns (per-landmark error mm, fk1, fk2, synth).

    Each pass gets its helper from a FACTORY that receives that pass's OWN Y-up landmark
    array, because the legacy anchor is expressed in the landmark frame: pass 2's input is
    the synthetic body, not the capture, and a helper closed over pass 1's landmarks would
    make the "before" arm something the pre-change converter never computed. That bug was
    live here and showed up as B(a) missing the committed 41.57/47.05 by 20 mm.

    The two factories are separate so a control can indict the INSTRUMENT's input
    construction (pass 2 only) while leaving the delivered path (pass 1) untouched.
    """
    make_pass1 = make_pass1 or _shipped
    make_pass2 = make_pass2 or make_pass1
    with origin(make_pass1(rc.Y_UP_FROM_Z_UP(src_z_up))):
        t1 = rc.solve(src_z_up, skeleton)
        fk1 = rc.fk_of(t1, skeleton)
    synth = rc.landmarks_from_fk(fk1, skeleton)
    with origin(make_pass2(synth)):
        t2 = rc.solve(rc.Z_UP_FROM_Y_UP(synth), skeleton)
        fk2 = rc.fk_of(t2, skeleton)
    return rc.score(fk2, synth, skeleton), fk1, fk2, synth


def on_capture(src_z_up, skeleton=None, helper=None):
    with origin(helper or TRUE_ORIGIN):
        t = rc.solve(src_z_up, skeleton)
        fk = rc.fk_of(t, skeleton)
    return rc.score(fk, rc.Y_UP_FROM_Z_UP(src_z_up), skeleton), t, fk


def groups(err):
    s = rc.summarise(err)
    return {g: s["per_group"][g]["median_mm"] for g in rc.GROUPS}


def per_landmark_arms(err):
    s = rc.summarise(err)
    return {n: s["per_landmark"][n]["median_mm"] for n in rc.GROUPS["arms"]}


# ------------------------------------------------------------------ statistics
def block_bootstrap(series_a, series_b, rng, statistic=np.median):
    """Paired moving-block bootstrap of statistic(A) - statistic(B) on IDENTICAL draws.

    Block 15, 2000 draws, seed 20260902 -- `retarget_cost.block_bootstrap_margin`'s
    scheme, reimplemented here ONLY so the statistic is pluggable (the scoreboard's
    headline is a median of per-joint medians, not a pooled median). The pooled-median
    call below is arithmetically the same procedure as the imported one on the same rng
    stream; `rc.block_bootstrap_margin` is imported and used verbatim wherever the
    statistic IS the pooled median.
    """
    A, B = np.asarray(series_a), np.asarray(series_b)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    nblocks = max(1, n // rc.BLOCK)
    starts_max = max(1, n - rc.BLOCK + 1)
    diffs = np.empty(rc.DRAWS)
    for d in range(rc.DRAWS):
        s = rng.integers(0, starts_max, size=nblocks)
        idx = np.concatenate([np.arange(t, min(t + rc.BLOCK, n)) for t in s])
        diffs[d] = statistic(A[idx]) - statistic(B[idx])
    return {"median_mm": round(float(np.median(diffs)), 2),
            "ci95_mm": [round(float(np.percentile(diffs, 2.5)), 2),
                        round(float(np.percentile(diffs, 97.5)), 2)],
            "p_wrong_sign": round(float(np.mean(diffs <= 0.0) if np.median(diffs) > 0
                                        else np.mean(diffs >= 0.0)), 4),
            "block_frames": rc.BLOCK, "draws": rc.DRAWS,
            "note": "positive means the FIRST arm is worse; lower is better"}


# A human's peak joint angular rate is near 500-800 deg/s (CLAUDE.md, the head gate's
# physical reject). At this take's 30 fps that is 17-27 deg per frame; the ceiling below
# takes the generous end. It is a PHYSICAL reference, not a fitted one.
PHYSICAL_CEILING_DEG_PER_FRAME = 800.0 / 30.0


def step_angles_deg(quaternions):
    """Frame-to-frame rotation of one joint's LOCAL quaternion, in degrees.

    Local, not world, and for the same reason the head gate measures at the joint: a bone
    on a turning body travels in world with the joint perfectly still.
    """
    a, b = quaternions[:-1], quaternions[1:]
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), -1.0, 1.0)))


def temporal_block(track_before, track_after) -> dict:
    """What the positional score cannot see: does the clavicle chain still MOVE like a body?

    D2 measures the direction from an origin only 60-170 mm from the landmark, where the
    old anchor sat ~400 mm away. A short lever arm turns the same landmark noise into much
    more direction noise, so a fix that is unambiguously right about WHERE the arm root
    goes can still be worse about HOW it gets there. Both are reported; neither is the
    other's evidence.
    """
    out: dict = {"ceiling_deg_per_frame": round(PHYSICAL_CEILING_DEG_PER_FRAME, 2),
                 "ceiling_reference": "a human joint's peak angular rate, ~800 deg/s, at "
                                      "this take's 30 fps -- physical, not fitted",
                 "joints": {}}
    qa = np.asarray(track_before.local_rotations_xyzw, dtype=np.float64)
    qb = np.asarray(track_after.local_rotations_xyzw, dtype=np.float64)
    worse = 0
    for name in ("LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
                 "LeftLowerArm", "RightLowerArm", "LeftLowerLeg", "RightLowerLeg",
                 "Neck", "Head"):
        index = DETAILED_HUMANOID.index(name)
        a, b = step_angles_deg(qa[:, index]), step_angles_deg(qb[:, index])
        row = {}
        for label, v in (("before", a), ("after", b)):
            row[label] = {
                "median_deg": round(float(np.median(v)), 2),
                "p95_deg": round(float(np.percentile(v, 95)), 2),
                "max_deg": round(float(v.max()), 2),
                "frames_over_the_physical_ceiling": int((v > PHYSICAL_CEILING_DEG_PER_FRAME).sum()),
            }
        row["identical"] = bool(np.array_equal(a, b))
        out["joints"][name] = row
        if name in ("LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
                    "LeftLowerArm", "RightLowerArm"):
            worse += row["after"]["frames_over_the_physical_ceiling"] - \
                row["before"]["frames_over_the_physical_ceiling"]
    out["clavicle_chain_extra_frames_over_the_ceiling"] = int(worse)
    out["legs_head_and_neck_are_bit_identical"] = all(
        out["joints"][n]["identical"] for n in
        ("LeftLowerLeg", "RightLowerLeg", "Neck", "Head"))
    return out


def world_quaternions(track):
    q = np.asarray(track.local_rotations_xyzw, dtype=np.float64)
    w = np.zeros_like(q)
    for index, joint in enumerate(DETAILED_HUMANOID.joints):
        w[:, index] = (q[:, index] if joint.parent == -1
                       else cm._quaternion_multiply(w[:, joint.parent], q[:, index]))
    return w


def twist_block(track_before, fk_before, track_after) -> dict:
    """How much the arm TURNED ABOUT ITSELF, and proof that is all it did.

    Every joint-origin score in this lane is blind to a rotation about a bone's own long
    axis: it displaces nothing on that bone's chain. D1 learned this on the finger curl.
    Here the whole arm below the clavicle inherits the clavicle's roll -- `_world_for_bone`
    is the MINIMAL rotation from the parent's frame, so it passes the roll straight down --
    and the number below is what the delivered hand's orientation moved by, which nothing
    else in this gate can see. `axis_offset_from_the_bone_deg` near 0 is the proof that
    the change is a pure twist and not a re-aim.
    """
    wb, wa = world_quaternions(track_before), world_quaternions(track_after)
    out: dict = {}
    for name, child in (("LeftUpperArm", "LeftLowerArm"), ("RightUpperArm", "RightLowerArm"),
                        ("LeftLowerArm", "LeftHand"), ("RightLowerArm", "RightHand")):
        index = DETAILED_HUMANOID.index(name)
        rel = cm._quaternion_multiply(wa[:, index], cm._quat_inverse(wb[:, index]))
        vec = rel[:, :3]
        norm = np.linalg.norm(vec, axis=1)
        angle = np.degrees(2.0 * np.arctan2(norm, np.abs(rel[:, 3])))
        ok = norm > 1e-9
        bone = fk_before[:, DETAILED_HUMANOID.index(child)] - fk_before[:, index]
        bone = bone / np.linalg.norm(bone, axis=1, keepdims=True)
        axis = np.where(rel[:, 3:4] >= 0.0, vec, -vec)[ok] / norm[ok, None]
        off = np.degrees(np.arccos(np.clip(np.abs(np.sum(axis * bone[ok], axis=1)), -1.0, 1.0)))
        out[name] = {
            "reorientation_median_deg": round(float(np.median(angle)), 2),
            "reorientation_p95_deg": round(float(np.percentile(angle, 95)), 2),
            "axis_offset_from_the_bone_median_deg": round(float(np.median(off)), 4),
            "axis_offset_from_the_bone_p95_deg": round(float(np.percentile(off, 95)), 4),
        }
    out["reading"] = (
        "the axis offset is 0.00 deg on every bone, so the ENTIRE change below the "
        "clavicle is a rotation about each bone's own long axis: a pure twist, carried "
        "rigidly down to the hand. No joint-origin score in this lane can see it, because "
        "a twist about a bone displaces nothing on that bone's chain. The only instrument "
        "is a picture -- artifacts/compare/d2-clavicle/hand-s*-*-{BEFORE,AFTER}.jpg.")
    return out


def quat_angle_deg(a, b):
    """Smallest rotation between two quaternion arrays, in degrees -- SIGN AWARE.

    `max|q1 - q2|` reads ~2.0 for a pure sign flip and says nothing. q and -q are the
    same rotation; the angle is not.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.abs(np.sum(a * b, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


# ------------------------------------------------------------------ A + B, per subject
def subject_block(subject, mamma_body_id, rng, report):
    src = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")[
        "triangulated_world_positions_z_up_m"]
    src = src[np.isfinite(src).all(axis=(1, 2))]
    points_y = rc.Y_UP_FROM_Z_UP(src)
    legacy = legacy_anchor(points_y)
    out: dict = {"frames_scored": int(len(src))}

    # ---- A. the round trip, after (shipped) and before (the legacy anchor, same path)
    err_after, fk1_after, fk2_after, synth_after = round_trip(src)
    err_before, fk1_before, fk2_before, synth_before = round_trip(src, make_pass1=make_legacy)
    out["roundtrip"] = {
        "reference": REF_ROUNDTRIP,
        "after": {"per_group": groups(err_after), "per_landmark_arms": per_landmark_arms(err_after)},
        "before_via_legacy_anchor_swap": {
            "per_group": groups(err_before), "per_landmark_arms": per_landmark_arms(err_before)},
        "band_arms_median_mm": BAND_ARMS_MM,
        "band_legs_and_torso_median_mm": BAND_LEGS_MM,
    }

    # ---- B(a). the swap must reproduce the COMMITTED pre-change report exactly
    committed = json.loads(BEFORE_REPORT.read_text())["subjects"][f"subject_{subject:02d}"]
    committed_arms = committed["arms"]["ORACLE_roundtrip_canonical"]["per_group"]["arms"]["median_mm"]
    committed_legs = committed["arms"]["ORACLE_roundtrip_canonical"]["per_group"]["legs"]["median_mm"]
    out["CONTROL_legacy_anchor_reproduces_the_committed_report"] = {
        "committed_arms_median_mm": committed_arms,
        "swapped_arms_median_mm": groups(err_before)["arms"],
        "abs_difference_mm": round(abs(committed_arms - groups(err_before)["arms"]), 4),
        "band_mm": 0.01,
        "committed_legs_median_mm": committed_legs,
        "swapped_legs_median_mm": groups(err_before)["legs"],
        "why": "proves the helper swap is FAITHFUL -- the control runs the identical code "
               "path as the candidate -- and that the pre-change converter fails the band.",
    }

    # ---- B(b). no on-axis constant reaches the band. The 'tune 0.72' degenerate.
    sweep = {}
    for scalar in SWEEP:
        e, _, _, _ = round_trip(src, make_pass1=make_legacy_scalar(scalar))
        g = groups(e)
        sweep[f"{scalar:.2f}"] = {"arms_median_mm": g["arms"], "legs_median_mm": g["legs"],
                                  "torso_median_mm": g["torso"]}
    best = min(sweep, key=lambda k: sweep[k]["arms_median_mm"])
    out["CONTROL_legacy_scalar_sweep"] = {
        "swept": "0.40 .. 1.20 step 0.05, the legacy on-torso-axis anchor",
        "per_scalar": sweep,
        "best_scalar": best,
        "best_arms_median_mm": sweep[best]["arms_median_mm"],
        "band_arms_median_mm": BAND_ARMS_MM,
        "why": "NO GATE A CONSTANT CAN PASS. The true origin is 110 mm off the torso axis; "
               "no point ON that axis can land it. This is also the 'tune 0.72 until one "
               "subject improves' degenerate named in the plan.",
    }

    # ---- B(c). the parent's origin -- the plausible off-by-one
    e_uc, _, _, _ = round_trip(src, make_pass1=make_upperchest)
    out["CONTROL_upperchest_origin"] = {
        "per_group": groups(e_uc), "band_arms_median_mm": BAND_ARMS_MM,
        "why": "MUST FAIL: the clavicle's PARENT origin, 110 mm inboard of the right one.",
    }

    # ---- the attribution. Pass 2 only, then both passes.
    e_p2, _, fk2_p2, synth_p2 = round_trip(src, make_pass2=make_hip_drop_removed)
    e_both, _, _, _ = round_trip(src, make_pass1=make_hip_drop_removed)
    out["CONTROL_hip_drop_removed_pass2"] = {
        "per_group": groups(e_p2), "band_arms_median_mm": BAND_ATTRIBUTION_MM,
        "why": "THE ATTRIBUTION, and it indicts the INSTRUMENT, not the shipped converter: "
               "the delivered path (pass 1) is untouched and only the round trip's RE-SOLVE "
               "measures its origin from a rig whose leg roots, rather than Hips, sit on the "
               "hip landmarks. If the arms collapse here, the whole D2 round-trip residual "
               "is the root-placement convention and none of it is the clavicle.",
    }
    out["CONTROL_hip_drop_removed_both_passes"] = {
        "per_group": groups(e_both),
        "why": "confirmation of the same attribution with the variant applied throughout.",
    }

    # ---- expectation 2: is the remainder the shoulder-origin displacement?
    out["expectation_2_residual_attribution"] = residual_attribution(
        fk1_after, fk2_after, synth_after, "as shipped (root placement contaminates it)")
    out["expectation_2_under_the_corrected_control"] = residual_attribution(
        fk1_after, fk2_p2, synth_p2, "pass-2 hip drop removed -- the roll term alone")

    # ---- the delivered configuration, on our own capture
    e_cap_after, t_cap_after, fk_cap_after = on_capture(src)
    e_cap_before, t_cap_before, _ = on_capture(src, helper=legacy)
    out["on_our_capture_canonical"] = {
        "reference": REF_OURS,
        "before": groups(e_cap_before), "after": groups(e_cap_after),
        "note": "the delivered configuration: ONE solve, real captured landmarks. This is "
                "the arm the round-trip contamination does not touch.",
    }

    # ---- sized rig: re-solved vs the scoreboard's replay. P5, scale invariance.
    skel_sz, limbs = sized_skeleton(DETAILED_HUMANOID, src)
    e_sz_after, t_sz_after, _ = on_capture(src, skel_sz)
    e_sz_before, t_sz_before, _ = on_capture(src, skel_sz, helper=legacy)
    fk_replay_after = rc.fk_of(t_cap_after, skel_sz)
    fk_replay_before = rc.fk_of(t_cap_before, skel_sz)
    e_replay_after = rc.score(fk_replay_after, points_y, skel_sz)
    e_replay_before = rc.score(fk_replay_before, points_y, skel_sz)
    ang_after = quat_angle_deg(t_sz_after.local_rotations_xyzw, t_cap_after.local_rotations_xyzw)
    ang_before = quat_angle_deg(t_sz_before.local_rotations_xyzw, t_cap_before.local_rotations_xyzw)
    worst = np.argsort(-ang_after.max(axis=0))[:6]
    out["temporal"] = temporal_block(t_cap_before, t_cap_after)
    out["twist"] = twist_block(t_cap_before, rc.fk_of(t_cap_before), t_cap_after)
    out["sized_rig"] = {
        "reference": REF_OURS,
        "resolved_before": groups(e_sz_before), "resolved_after": groups(e_sz_after),
        "replayed_scoreboard_method_before": groups(e_replay_before),
        "replayed_scoreboard_method_after": groups(e_replay_after),
        "scale_invariance_max_joint_rotation_deg_before": round(float(np.max(ang_before)), 6),
        "scale_invariance_max_joint_rotation_deg_after": round(float(np.max(ang_after)), 6),
        "finding": "P5, expected and by design: before D2 a re-solve on a sized skeleton "
                   "returned bit-identical rotations, because every rotation aimed a rest "
                   "DIRECTION and sizing does not turn a rest offset. D2 measures the "
                   "clavicle from the SIZED rig's own origin, which sizing DOES move, so "
                   "the two stop being the same arm and the scoreboard's replayed `sized` "
                   "column is no longer equivalent to a re-solve. The scoreboard's method "
                   "is unchanged; both are reported.",
        "scale_invariance_worst_joints_after": {
            DETAILED_HUMANOID.joints[int(i)].name: {
                "max_deg": round(float(ang_after[:, int(i)].max()), 2),
                "median_deg": round(float(np.median(ang_after[:, int(i)])), 2)}
            for i in worst},
        "scale_invariance_note": "the break is confined to the six clavicle-chain joints. "
                                 "It is large because the shoulder landmark sits only "
                                 "60-170 mm from the rig's shoulder origin, so moving the "
                                 "origin (which sizing does) swings the direction hard. "
                                 "The same short lever arm is why the temporal block below "
                                 "gets worse.",
        "measured_limbs_mm": limbs,
    }

    # ---- margins on the round trip and on the capture, identical draws
    for label, before, after in (
        ("roundtrip_arms_before_minus_after", err_before, err_after),
        ("on_capture_arms_before_minus_after", e_cap_before, e_cap_after),
    ):
        out.setdefault("margins", {})[label] = rc.block_bootstrap_margin(
            before, after, "arms", rng)
        out["margins"][label.replace("arms", "legs")] = rc.block_bootstrap_margin(
            before, after, "legs", rng)
    out["margins"]["lag1_autocorrelation_on_capture_arms"] = rc.lag1(
        rc.group_frame_series(e_cap_after, "arms"))
    out["margins"]["note"] = (
        "moving-block bootstrap, block 15, 2000 draws, seed 20260902, BOTH ARMS ON "
        "IDENTICAL DRAWN FRAMES via `retarget_cost.block_bootstrap_margin` imported "
        "verbatim. The BEFORE per-frame series is the legacy-anchor SWAP's, not the "
        "committed JSON's (which carries only medians); the swap reproducing the "
        "committed medians to 0.01 mm is what licenses that substitution.")

    # ---- arm B: MAMMA's own joints in. Separate reference, reports and never selects.
    pred = np.load(rc.MA3D / f"verts_joints_body_id-{mamma_body_id:02d}.npz",
                   allow_pickle=True)["pred_joints"].astype(np.float64)
    pred = pred[np.isfinite(pred).all(axis=(1, 2))]
    m_src = rc.adapt_mamma(pred)
    m_points_y = rc.Y_UP_FROM_Z_UP(m_src)
    m_legacy = legacy_anchor(m_points_y)
    b_after, _, _, _ = round_trip(m_src)
    b_before, _, _, _ = round_trip(m_src, make_pass1=make_legacy)
    b_cap_after, _, _ = on_capture(m_src)
    b_cap_before, _, _ = on_capture(m_src, helper=m_legacy)
    m_skel, _ = sized_skeleton(DETAILED_HUMANOID, m_src)
    b_sz_after, _, _ = on_capture(m_src, m_skel)
    b_sz_before, _, _ = on_capture(m_src, m_skel, helper=m_legacy)
    b_p2, _, _, _ = round_trip(m_src, make_pass2=make_hip_drop_removed)
    out["arm_B_mamma_joints_in"] = {
        "reference": REF_ARMB,
        "mamma_body_id": f"body_id-{mamma_body_id:02d}",
        "frames_scored": int(len(pred)),
        "roundtrip_before": groups(b_before), "roundtrip_after": groups(b_after),
        "roundtrip_after_hip_drop_removed_pass2": groups(b_p2),
        "canonical_before": groups(b_cap_before), "canonical_after": groups(b_cap_after),
        "sized_resolved_before": groups(b_sz_before), "sized_resolved_after": groups(b_sz_after),
        "margins": {
            "roundtrip_arms_before_minus_after": rc.block_bootstrap_margin(
                b_before, b_after, "arms", rng),
            "canonical_arms_before_minus_after": rc.block_bootstrap_margin(
                b_cap_before, b_cap_after, "arms", rng),
            "sized_arms_before_minus_after": rc.block_bootstrap_margin(
                b_sz_before, b_sz_after, "arms", rng),
        },
        "not_comparable_with": "the own-capture arms -- a different body, different poses "
                              "and a different joint convention. Never one axis.",
    }
    return out


def residual_attribution(fk1, fk2, synth, label):
    """Expectation 2, and its correction, measured.

    Pre-registered form: residual = (d.e)e, the component of the FK Shoulder-origin
    displacement ALONG the upper arm -- not the displacement itself, because the
    transverse part is absorbed by re-aiming the clavicle.
    """
    D = DETAILED_HUMANOID
    o1 = 0.5 * (fk1[:, D.index("LeftUpperLeg")] + fk1[:, D.index("RightUpperLeg")])
    o2 = 0.5 * (fk2[:, D.index("LeftUpperLeg")] + fk2[:, D.index("RightUpperLeg")])
    rows = {}
    for side, landmark in (("Left", "left_shoulder"), ("Right", "right_shoulder")):
        iS, iA = D.index(f"{side}Shoulder"), D.index(f"{side}UpperArm")
        d = (fk2[:, iS] - o2) - (fk1[:, iS] - o1)
        e = fk1[:, iA] - fk1[:, iS]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        proj = np.sum(d * e, axis=1)[:, None] * e
        resid = (fk2[:, iA] - o2) - (synth[:, cm.JOINT_INDEX[landmark]] - o1)
        mm = lambda v: round(float(np.median(np.linalg.norm(v, axis=1))) * 1000.0, 3)
        rows[side] = {
            "shoulder_origin_displacement_mm": mm(d),
            "its_component_along_the_upper_arm_mm": mm(proj),
            "measured_arm_residual_mm": mm(resid),
            "residual_minus_projection_mm": mm(resid - proj),
        }
    rows["reading"] = (
        f"{label}. The pre-registered prediction is residual == (d.e)e. It holds only when "
        "the two solves place the rig identically against the landmark cloud; the shipped "
        "round trip does not, because of the 80 mm root-placement offset, and there the "
        "unexplained term is the whole residual. Under the corrected control the "
        "prediction is the only term left.")
    return rows


# ------------------------------------------------------------------ C. bit identity
CLAVICLE_CHAIN = {"LeftShoulder", "LeftUpperArm", "LeftLowerArm",
                  "RightShoulder", "RightUpperArm", "RightLowerArm"}
HANDS = {"LeftHand", "RightHand"}


def bit_identity() -> dict:
    """Theorems, asserted rather than measured, between the delivery and the rebuild."""
    out: dict = {"held": True, "checks": {}}

    def check(name, ok, detail=""):
        out["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            out["held"] = False

    for s in (0, 1):
        a = np.load(TRACKS / f"subject-{s:02d}.body-track.npz")
        b = np.load(REBUILD / f"subject-{s:02d}.body-track.npz")
        names = json.loads((TRACKS / f"subject-{s:02d}.body-track.json").read_text())["joint_names"]
        for key in ("root_translation_m", "triangulated_world_positions_z_up_m",
                    "raw_triangulated_world_positions_z_up_m", "ticks", "foot_contacts"):
            if key not in a.files:
                check(f"subject_{s:02d}.{key}", False, "absent from the delivered track")
                continue
            # BYTE equality, not `np.array_equal`: the raw triangulated array carries
            # NaN on unresolved joints and NaN != NaN, so array_equal reports a
            # difference where the bytes are identical. That fired here.
            same = (a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
                    and a[key].tobytes() == b[key].tobytes())
            check(f"subject_{s:02d}.{key}", same,
                  f"{a[key].shape} {a[key].dtype}, "
                  f"{int(np.isnan(a[key]).sum()) if a[key].dtype.kind == 'f' else 0} NaN")
        qa, qb = a["local_rotations_xyzw"], b["local_rotations_xyzw"]
        differ = {names[i] for i in range(len(names)) if not np.array_equal(qa[:, i], qb[:, i])}
        check("differing_local_rotations_contain_the_clavicle_chain",
              CLAVICLE_CHAIN <= differ, f"missing: {sorted(CLAVICLE_CHAIN - differ)}")
        check("differing_local_rotations_are_within_the_chain_plus_the_two_hands",
              differ <= (CLAVICLE_CHAIN | HANDS), f"unexpected: {sorted(differ - CLAVICLE_CHAIN - HANDS)}")
        out["checks"][f"subject_{s:02d}.differing_local_rotations"] = {
            "ok": True, "detail": sorted(differ)}
        fingers = sorted(n for n in differ if any(
            n.startswith(p) for p in ("LeftThumb", "LeftIndex", "LeftMiddle", "LeftRing",
                                      "LeftLittle", "RightThumb", "RightIndex",
                                      "RightMiddle", "RightRing", "RightLittle")))
        check(f"subject_{s:02d}.no_finger_local_rotation_moved", not fingers, str(fingers))

    for name in sorted(p.name for p in (TRACKS / "work").glob("*observations.jsonl")):
        a = (TRACKS / "work" / name).read_bytes()
        b = (REBUILD / "work" / name).read_bytes()
        check(f"work/{name}_byte_identical", a == b, f"{len(a)} vs {len(b)} bytes")
    out["means"] = (
        "The head solve and the facing instrument read the triangulated positions and the "
        "torso/Neck/Head rotations. Every one of those is bit-identical, so rung 9 and the "
        "facing figures CANNOT move under this change -- by construction, not by "
        "re-measurement. Nothing was re-extracted or re-detected: the observation files "
        "are the same bytes.")
    return out


# ------------------------------------------------------------------ E. the scoreboard
def scoreboard_errors(tracks: Path, mapping) -> dict:
    """The scoreboard's own method, per subject, per joint, per frame, in mm.

    Imports PAIRS / RIG / the sizing and the subject map from `mamma_scoreboard` so the
    two instruments cannot drift. `body_id-00` is our subject 1 on this fixture and the
    pairing is resolved from 3D pelvis agreement, never by index.
    """
    out: dict = {}
    for s in (0, 1):
        names = json.loads((tracks / f"subject-{s:02d}.body-track.json").read_text())["joint_names"]
        base = skeleton_for_joint_names(names)
        track = np.load(tracks / f"subject-{s:02d}.body-track.npz")
        cap_z = track["triangulated_world_positions_z_up_m"]
        m_joints = np.load(rc.MA3D / f"verts_joints_body_id-{mapping[s]:02d}.npz",
                           allow_pickle=True)["pred_joints"].astype(np.float64)
        n = min(len(cap_z), len(m_joints))
        fitted, _ = sized_skeleton(base, cap_z)
        w_canon = forward_kinematics_positions(
            track["root_translation_m"], track["local_rotations_xyzw"], skeleton=base)
        w_sized = forward_kinematics_positions(
            track["root_translation_m"], track["local_rotations_xyzw"], skeleton=fitted)
        rig_to_z = lambda w: np.stack([w[..., 0], -w[..., 2], w[..., 1]], axis=-1)
        arms = {"capture": cap_z[:n], "canon": rig_to_z(w_canon)[:n],
                "sized": rig_to_z(w_sized)[:n]}
        rows: dict = {}
        for name, mi in sb.PAIRS.items():
            ref = m_joints[:n, mi]
            for arm, data in arms.items():
                idx = cm.JOINT_INDEX[name] if arm == "capture" else names.index(sb.RIG[name])
                p = data[:, idx]
                ok = np.isfinite(p).all(axis=1) & np.isfinite(ref).all(axis=1)
                series = np.full(n, np.nan)
                series[ok] = np.linalg.norm(p[ok] - ref[ok], axis=1) * 1000.0
                rows.setdefault(arm, {})[name] = series
        out[f"subject_{s:02d}"] = rows
    return out


def median_of_joint_medians(names):
    """The scoreboard's HEADLINE statistic, as a function of drawn frames."""
    def statistic(stack):
        return float(np.median([np.nanmedian(stack[:, j]) for j in range(stack.shape[1])]))
    return statistic, names


ARMS_SIX = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist")


def scoreboard_block(before, after, rng) -> dict:
    out: dict = {"reference": REF_MAMMA,
                 "statistic": "the scoreboard's own headline: per-joint median over frames, "
                              "then the median over joints. The bootstrap resamples THAT "
                              "function, not a pooled median.",
                 "subjects": {}}
    for s in ("subject_00", "subject_01"):
        entry: dict = {}
        for arm in ("capture", "canon", "sized"):
            for scope, names in (("all_joints", list(sb.PAIRS)), ("arms_six", list(ARMS_SIX))):
                A = np.stack([before[s][arm][n] for n in names], axis=1)
                B = np.stack([after[s][arm][n] for n in names], axis=1)
                stat, _ = median_of_joint_medians(names)
                entry[f"{arm}_{scope}"] = {
                    "before_mm": round(stat(A), 2), "after_mm": round(stat(B), 2),
                    "margin_before_minus_after": block_bootstrap(A, B, rng, statistic=stat),
                }
        entry["capture_must_be_identical"] = {
            "max_abs_difference_mm": round(float(np.nanmax(np.abs(
                np.stack([before[s]["capture"][n] for n in sb.PAIRS], axis=1)
                - np.stack([after[s]["capture"][n] for n in sb.PAIRS], axis=1)))), 9),
            "why": "the capture arm reads the triangulated positions, which this change "
                   "does not touch. Anything but 0 means the rebuild is not the same data.",
        }
        out["subjects"][s] = entry
    return out



# =========================================================================== ROOT VARIANT
# Measured on the branch, SHIPPING NOTHING. Everything here is an instrument-side swap;
# `src/` is untouched by it.
#
# THE DERIVATION, and it introduces no constant. Forward kinematics puts the leg roots at
#     UpperLegMid = root_translation + rest[Root] + rest[Hips] + R_hips . mid
# with mid = 0.5 * (rest[LeftUpperLeg] + rest[RightUpperLeg]) and rest[Root] = 0. The
# captured `left_hip`/`right_hip` landmarks ARE the femoral joint centres, so the rig's leg
# roots -- not `Hips` -- are what belongs on their midpoint. Setting UpperLegMid = pelvis,
#     root_translation = pelvis - rest[Hips] - R_hips . mid
# against the shipped `root_translation = pelvis - rest[Hips]`. Every term is the
# skeleton's own rest geometry.
#
# TWO PLACES IT HAS TO ACT, and only using both is faithful:
#   * the ROTATIONS -- `_joint_origin` reads `root_translation`, so the clavicle direction
#     moves. `hip_drop_removed` above produces exactly this, because a change in
#     `root_translation` shifts every FK origin by exactly that vector.
#   * the ROOT ITSELF -- and it must be shifted BEFORE `project_generated_foot_contacts`
#     runs, or the question in item 2 is not being asked. So the projection is WRAPPED,
#     never re-implemented: the wrapper shifts the incoming track and hands it to the real
#     function, whose diagnostics it also captures (`positions_to_body_track` discards them).
ROOT_VARIANT_PREDICTION = (
    "Set by the coordinator BEFORE this ran, and recorded here verbatim: on the CANONICAL "
    "rig the root fix raises the pivot to landmark height and SHORTENS the direction lever "
    "further, so canonical placement may get WORSE than D2 alone and jitter may GROW; on "
    "the SIZED rig both should IMPROVE. Those two cells decide whether 'D2 + root "
    "placement' is a scoping option or whether D2 waits for the per-performer skeleton "
    "(D3/D5)."
)


class root_on_the_leg_roots(object):
    """`with root_on_the_leg_roots() as diag:` -- wrap the foot projection, do not re-write it.

    CLAUDE.md: wrap the pipeline to instrument it. A hand-rolled projection would be a
    different function with the same name, and the whole point of item 2 is what THIS one
    does when it is handed a root 80 mm higher than it expects.
    """

    def __init__(self):
        self.diagnostics = []

    def __enter__(self):
        self.saved = cm.project_generated_foot_contacts
        real = self.saved
        store = self.diagnostics

        def wrapper(track, **kwargs):
            rotations = np.asarray(track.local_rotations_xyzw, dtype=np.float64)
            roots = np.asarray(track.root_translation_m, dtype=np.float64)
            names = list(track.joint_names)
            hips_world = cm._quaternion_multiply(
                rotations[:, names.index("Root")], rotations[:, names.index("Hips")])
            rest = {n: np.asarray(j.rest_translation_m, dtype=np.float64)
                    for n, j in zip(names, DETAILED_HUMANOID.joints)}
            mid = 0.5 * (rest["LeftUpperLeg"] + rest["RightUpperLeg"])
            shift = _rotate_vector(hips_world, np.broadcast_to(mid, roots.shape))
            lifted = dataclasses.replace(
                track, root_translation_m=(roots - shift).astype(np.float32))
            projected, diag = real(lifted, **kwargs)
            store.append({
                "requested_root_lift_mm": round(float(np.median(
                    np.linalg.norm(shift, axis=1))) * 1000.0, 2),
                "diagnostics": diag.as_dict(),
                "root_before_projection": np.asarray(lifted.root_translation_m, np.float64),
                "root_after_projection": np.asarray(projected.root_translation_m, np.float64),
                "feet": foot_heights(projected),
                "kwargs": {k: v for k, v in kwargs.items()},
            })
            return projected, diag

        cm.project_generated_foot_contacts = wrapper
        return self.diagnostics

    def __exit__(self, *exc):
        cm.project_generated_foot_contacts = self.saved
        return False


class record_projection(object):
    """The same capture with NO shift -- the delivered path, so the two are comparable."""

    def __init__(self):
        self.diagnostics = []

    def __enter__(self):
        self.saved = cm.project_generated_foot_contacts
        real = self.saved
        store = self.diagnostics

        def wrapper(track, **kwargs):
            projected, diag = real(track, **kwargs)
            store.append({
                "requested_root_lift_mm": 0.0,
                "diagnostics": diag.as_dict(),
                "root_before_projection": np.asarray(track.root_translation_m, np.float64),
                "root_after_projection": np.asarray(projected.root_translation_m, np.float64),
                "feet": foot_heights(projected),
                "kwargs": {k: v for k, v in kwargs.items()},
            })
            return projected, diag

        cm.project_generated_foot_contacts = wrapper
        return self.diagnostics

    def __exit__(self, *exc):
        cm.project_generated_foot_contacts = self.saved
        return False


def foot_heights(track) -> dict:
    """The lowest point of either foot, per frame, on the CANONICAL rig.

    `project_generated_foot_contacts` hardcodes `DETAILED_HUMANOID` (body_projection.py
    :1127 and its FK calls), so it measures a sized solve's feet on canonical rest lengths.
    That is pre-existing and is not changed here; it is recorded because it is exactly the
    sort of thing that makes a sized-rig ground figure mean less than it looks.
    """
    fk = forward_kinematics_positions(
        np.asarray(track.root_translation_m, dtype=np.float64),
        np.asarray(track.local_rotations_xyzw, dtype=np.float64),
        skeleton=DETAILED_HUMANOID).astype(np.float64)
    idx = [DETAILED_HUMANOID.index(n) for n in
           ("LeftFoot", "LeftToes", "RightFoot", "RightToes")]
    low = np.min(fk[:, idx, 1], axis=1)
    return {"lowest_foot_point_min_m": round(float(low.min()), 5),
            "lowest_foot_point_median_m": round(float(np.median(low)), 5),
            "floor_estimate_m": round(float(np.percentile(
                np.stack([np.min(fk[:, [idx[0], idx[1]], 1], axis=1),
                          np.min(fk[:, [idx[2], idx[3]], 1], axis=1)], axis=1), 2.0)), 5)}


def solve_variant(src_z_up, skeleton=None, helper=None, root_fix=False):
    """One solve, with the origin helper and (optionally) the root fix both installed."""
    capture = root_on_the_leg_roots() if root_fix else record_projection()
    with origin(helper or TRUE_ORIGIN):
        with capture as store:
            track = rc.solve(src_z_up, skeleton)
    return track, rc.fk_of(track, skeleton), (store[-1] if store else None)


def direction_levers(src_z_up, skeleton, helper) -> dict:
    """|landmark - the point the DIRECTION is measured from|, per shoulder.

    NOT the pivot-to-landmark distance in general: for the legacy anchor the bone still
    pivots at the rig's Shoulder origin while the direction is taken from the torso axis,
    and it is the DIRECTION's lever that sets how much landmark noise becomes bone noise.
    That is the quantity this step's jitter regression turns on, so it is the one reported.
    """
    points = rc.Y_UP_FROM_Z_UP(src_z_up)
    seen: dict = {}
    real = helper or TRUE_ORIGIN

    def recording(world, frame, root_translation, rest, joint_name):
        value = real(world, frame, root_translation, rest, joint_name)
        seen.setdefault(joint_name, {})[frame] = np.asarray(value).copy()
        return value

    with origin(recording):
        rc.solve(src_z_up, skeleton)
    out = {}
    for joint, landmark in (("LeftShoulder", "left_shoulder"),
                            ("RightShoulder", "right_shoulder")):
        frames = sorted(seen.get(joint, {}))
        if not frames:
            continue
        origins = np.stack([seen[joint][f] for f in frames])
        target = points[frames][:, cm.JOINT_INDEX[landmark]]
        d = np.linalg.norm(target - origins, axis=1) * 1000.0
        out[joint] = {"median_mm": round(float(np.median(d)), 1),
                      "p5_mm": round(float(np.percentile(d, 5)), 1),
                      "min_mm": round(float(d.min()), 1)}
    return out


def root_variant_block(subject, mamma_body_id) -> dict:
    src = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")[
        "triangulated_world_positions_z_up_m"]
    src = src[np.isfinite(src).all(axis=(1, 2))]
    points_y = rc.Y_UP_FROM_Z_UP(src)
    skel_sz, _ = sized_skeleton(DETAILED_HUMANOID, src)
    entry: dict = {}

    for rig_label, skel in (("canonical", None), ("sized_resolved", skel_sz)):
        rig: dict = {}
        for arm_label, helper, root_fix in (("D2_alone", None, False),
                                            ("D2_plus_root", hip_drop_removed, True)):
            track, fk, projection = solve_variant(src, skel, helper, root_fix)
            err = rc.score(fk, points_y, skel or DETAILED_HUMANOID)
            # the round trip, with the SAME variant on both passes
            synth = rc.landmarks_from_fk(fk, skel)
            t2, fk2, _ = solve_variant(rc.Z_UP_FROM_Y_UP(synth), skel, helper, root_fix)
            rt = rc.score(fk2, synth, skel or DETAILED_HUMANOID)
            rig[arm_label] = {
                "delivered_on_our_capture": groups(err),
                "roundtrip": groups(rt),
                "ground_projection": {k: v for k, v in (projection or {}).items()
                                      if k not in ("root_before_projection",
                                                   "root_after_projection")},
                "applied_root_correction_mm": round(float(np.median(np.linalg.norm(
                    projection["root_after_projection"] - projection["root_before_projection"],
                    axis=1))) * 1000.0, 2) if projection else None,
                "applied_root_correction_max_mm": round(float(np.max(np.linalg.norm(
                    projection["root_after_projection"] - projection["root_before_projection"],
                    axis=1))) * 1000.0, 2) if projection else None,
                "_track": track,
            }
        rig["temporal_D2_alone_vs_D2_plus_root"] = temporal_block(
            rig["D2_alone"].pop("_track"), rig["D2_plus_root"].pop("_track"))
        entry[rig_label] = rig

    entry["direction_lever_to_the_shoulder_landmark_mm"] = {
        "legacy_anchor_canonical": direction_levers(src, None, legacy_anchor(points_y)),
        "D2_canonical": direction_levers(src, None, None),
        "D2_plus_root_canonical": direction_levers(src, None, hip_drop_removed),
        "D2_plus_root_sized": direction_levers(src, skel_sz, hip_drop_removed),
        "reading": "the shorter this is, the more a millimetre of landmark noise becomes a "
                   "degree of bone noise. It is the constraint that discriminates.",
    }

    pred = np.load(rc.MA3D / f"verts_joints_body_id-{mamma_body_id:02d}.npz",
                   allow_pickle=True)["pred_joints"].astype(np.float64)
    pred = pred[np.isfinite(pred).all(axis=(1, 2))]
    m_src = rc.adapt_mamma(pred)
    m_points_y = rc.Y_UP_FROM_Z_UP(m_src)
    m_skel, _ = sized_skeleton(DETAILED_HUMANOID, m_src)
    arm_b: dict = {}
    for rig_label, skel in (("canonical", None), ("sized_resolved", m_skel)):
        for arm_label, helper, root_fix in (("D2_alone", None, False),
                                            ("D2_plus_root", hip_drop_removed, True)):
            _, fk, _ = solve_variant(m_src, skel, helper, root_fix)
            arm_b[f"{rig_label}_{arm_label}"] = groups(
                rc.score(fk, m_points_y, skel or DETAILED_HUMANOID))
    arm_b["reference"] = REF_ARMB
    entry["arm_B_mamma_joints_in"] = arm_b
    return entry


# ========================================================================== D2b
# THE ROOT PLACED ON THE CAPTURED HIPS. This half SHIPS: `positions_to_body_track`
# now reads
#     root_translation[frame] = pelvis - rest["Hips"] - _leg_root_offset(hips_world, rest)
# with `_leg_root_offset` returning `R_hips . mid(rest[LeftUpperLeg], rest[RightUpperLeg])`,
# so FK's UpperLeg midpoint -- the rig's femoral joint centres, which is what a `left_hip`
# landmark IS -- lands on the captured hip midpoint instead of the rig's `Hips` joint
# landing there and the leg roots hanging 80 mm below it. Every term is the skeleton's own
# rest geometry; no constant arrives, and the 80 is never written down.
#
# WHAT THIS MEANS FOR EVERY D2 FIGURE ABOVE. The D2 bands were measured against a shipped
# converter that had no root fix. To keep them meaning what they meant, `main()` runs every
# pre-existing block inside `with root_offset(zero_offset):` -- the D2-alone arm, reached by
# swapping the offset helper for one that returns the zero vector, through the identical
# code path. `d2_regression_check` then compares the result against `gate-d2-prior.json`,
# the file this run replaces, and FAILS if any D2 band moved. Without that check the
# extension could silently rewrite D2's committed numbers.
SHIPPED_OFFSET = cm._leg_root_offset
REBUILD_ROOT = OUT_DIR / "delivery-root"
PRIOR_GATE = OUT_DIR / "gate-d2-prior.json"

BAND_D2B_ARMS_MM = 5.0        # the round trip, both rigs -- D2's own pre-registered band
BAND_FK_HIP_M = 1e-6          # theorem 1; FK returns float32, so a micron is its floor
BAND_OFFSET_M = 1e-12         # theorem 2, in float64 on the helper's own return value
BAND_ROOT_F32_M = 1e-6        # theorem 2 on the track's float32 root arrays
LIFT_SWEEP_MM = [10.0 * i for i in range(17)]     # 0 .. 160 mm, control (c)

REF_ABSOLUTE = ("the captured hip-landmark midpoint in ABSOLUTE capture world, Z-up "
                "metres, AFTER the ground projection. NOT root-relative -- that is the "
                "whole point: every own-capture figure in D2 removed exactly this")
REF_SILHOUETTE = ("MAMMA's SAM2 masks -- pixels of the actual footage, the one retained "
                  "artifact on this fixture that is not model-mediated")

# ------------------------------------------------------------------ the offset variants
def zero_offset(hips_world, rest):
    """CONTROL (a) and the BEFORE arm: D2 alone, `Hips` back on the captured midpoint."""
    return np.zeros(3)


def sign_flipped_offset(hips_world, rest):
    """CONTROL (b): the identical lift with its SIGN flipped -- the rig 80 mm LOWER.

    Must fail. It is the same magnitude, the same frame, the same code path; only the
    direction differs, so it separates "a lift of the right size" from "the right lift".
    """
    return -SHIPPED_OFFSET(hips_world, rest)


def along_hips_up(metres: float):
    """CONTROL (c): a lift of `metres` along the HIPS' OWN up axis.

    The sweep is the no-constant demonstration. At the skeleton's own value -- read from
    `rest`, 80.0 mm on this rig -- this IS the shipped derivation; at every other value it
    is a tuned constant, and the band must reject all of them.
    """
    def helper(hips_world, rest):
        return _rotate_vector(hips_world, np.asarray((0.0, -float(metres), 0.0)))
    return helper


def world_vertical(metres: float):
    """CONTROL (d): the same 80 mm as a WORLD vertical. THE PLAUSIBLE SHORTCUT.

    It agrees with the derivation exactly when the pelvis is upright and diverges as
    `2*sin(tilt/2)*80 mm` as it leans, so on a take called "pushing and lifting from
    ground" it must fail, and it must fail HARDEST on the most-tilted frames. This is the
    control that discriminates the derivation from a number: a reader who accepts "lift the
    rig 80 mm" without asking *in which frame* gets this.
    """
    def helper(hips_world, rest):
        return np.asarray((0.0, -float(metres), 0.0))
    return helper


class root_offset(object):
    """`with root_offset(helper):` -- swap the module attribute the converter calls."""

    def __init__(self, helper):
        self.helper = helper

    def __enter__(self):
        self.saved = cm._leg_root_offset
        cm._leg_root_offset = self.helper

    def __exit__(self, *exc):
        cm._leg_root_offset = self.saved
        return False


class watch_projection(object):
    """Record the track HANDED TO the real ground projection, and the one it returns.

    `positions_to_body_track` discards the diagnostics and only the projected track ever
    reaches disk, so three of this step's theorems -- which are about the converter's
    output, before the projection adds its own vertical -- are unreachable from an
    artifact. The projection is WRAPPED, never re-implemented (CLAUDE.md).
    """

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self.saved = cm.project_generated_foot_contacts
        real, store = self.saved, self.calls

        def wrapper(track, **kwargs):
            projected, diagnostics = real(track, **kwargs)
            store.append({"incoming": track, "projected": projected,
                          "diagnostics": diagnostics.as_dict(),
                          "kwargs": {k: v for k, v in kwargs.items()}})
            return projected, diagnostics

        cm.project_generated_foot_contacts = wrapper
        return self.calls

    def __exit__(self, *exc):
        cm.project_generated_foot_contacts = self.saved
        return False


def solve_d2b(src_z_up, skeleton=None, offset=None, helper=None):
    """One real solve with an offset helper and an origin helper installed.

    Returns `(pre_projection_track, projected_track, fk, projection_call)`.
    """
    with root_offset(offset if offset is not None else SHIPPED_OFFSET):
        with origin(helper or TRUE_ORIGIN):
            with watch_projection() as calls:
                track = rc.solve(src_z_up, skeleton)
    call = calls[-1]
    return call["incoming"], track, rc.fk_of(track, skeleton), call


def round_trip_d2b(src_z_up, skeleton=None, offset=None):
    """solve -> FK -> landmarks -> solve, with the SAME offset on both passes."""
    _, _, fk1, _ = solve_d2b(src_z_up, skeleton, offset)
    synth = rc.landmarks_from_fk(fk1, skeleton)
    _, _, fk2, _ = solve_d2b(rc.Z_UP_FROM_Y_UP(synth), skeleton, offset)
    return rc.score(fk2, synth, skeleton or DETAILED_HUMANOID), fk1, fk2, synth


def _stats_mm(v):
    return {"median_mm": round(float(np.median(v)) * 1000.0, 2),
            "p5_mm": round(float(np.percentile(v, 5)) * 1000.0, 2),
            "p95_mm": round(float(np.percentile(v, 95)) * 1000.0, 2)}


# ------------------------------------------------------------------ the theorems
def d2b_theorems(src_z_up, skeleton, rig_label) -> dict:
    """Assertions, not measurements. Every one is pre-projection except where it says so."""
    skel = skeleton or DETAILED_HUMANOID
    points_y = rc.Y_UP_FROM_Z_UP(src_z_up)
    seen: list = []

    def recording(hips_world, rest):
        value = SHIPPED_OFFSET(hips_world, rest)
        seen.append((np.asarray(hips_world, dtype=np.float64).copy(),
                     np.asarray(value, dtype=np.float64).copy(),
                     np.asarray(rest["LeftUpperLeg"], dtype=np.float64).copy(),
                     np.asarray(rest["RightUpperLeg"], dtype=np.float64).copy()))
        return value

    pre_fixed, _, _, _ = solve_d2b(src_z_up, skeleton, offset=recording)
    pre_zero, _, _, _ = solve_d2b(src_z_up, skeleton, offset=zero_offset)

    fk_pre = forward_kinematics_positions(
        np.asarray(pre_fixed.root_translation_m, dtype=np.float64),
        np.asarray(pre_fixed.local_rotations_xyzw, dtype=np.float64),
        skeleton=skel).astype(np.float64)
    leg_mid = 0.5 * (fk_pre[:, skel.index("LeftUpperLeg")]
                     + fk_pre[:, skel.index("RightUpperLeg")])
    captured = 0.5 * (points_y[:, cm.JOINT_INDEX["left_hip"]]
                      + points_y[:, cm.JOINT_INDEX["right_hip"]])
    t1 = np.linalg.norm(leg_mid - captured, axis=1)

    hips_world = np.stack([row[0] for row in seen])
    offsets = np.stack([row[1] for row in seen])
    mid_rest = 0.5 * (np.stack([row[2] for row in seen])
                      + np.stack([row[3] for row in seen]))
    half_drop = float(-np.median(mid_rest[:, 1]))
    predicted = _rotate_vector(hips_world, np.broadcast_to(
        np.asarray((0.0, half_drop, 0.0)), offsets.shape))
    t2a = np.linalg.norm((-offsets) - predicted, axis=1)
    lateral = float(np.max(np.abs(mid_rest[:, [0, 2]])))

    root_fixed = np.asarray(pre_fixed.root_translation_m, dtype=np.float64)
    root_zero = np.asarray(pre_zero.root_translation_m, dtype=np.float64)
    n = min(len(root_fixed), len(offsets))
    t2b = np.linalg.norm((root_fixed[:n] - root_zero[:n]) - (-offsets[:n]), axis=1)

    qa = np.asarray(pre_zero.local_rotations_xyzw, dtype=np.float64)
    qb = np.asarray(pre_fixed.local_rotations_xyzw, dtype=np.float64)
    names = list(pre_zero.joint_names)
    differ = {names[i] for i in range(len(names))
              if not np.array_equal(qa[:, i], qb[:, i])}

    return {
        "rig": rig_label,
        "T1_fk_upperleg_midpoint_is_the_captured_hip_midpoint": {
            "max_m": float(t1.max()), "median_m": float(np.median(t1)),
            "band_m": BAND_FK_HIP_M, "ok": bool(t1.max() <= BAND_FK_HIP_M),
            "where": "PRE-PROJECTION -- the converter's own output. The ground projection "
                     "then adds its own vertical, which is what item 3 measures.",
            "why_not_tighter": "forward_kinematics_positions returns float32, so a "
                               "metre-scale coordinate is quantised at ~1e-7 m and a "
                               "micron is its floor over a four-joint chain. The plan's "
                               "1e-6 is that floor, not a slack band.",
        },
        "T2a_the_offset_is_R_hips_times_the_skeletons_own_half_drop": {
            "half_drop_read_from_rest_m": round(half_drop, 6),
            "max_lateral_component_of_mid_rest_m": lateral,
            "max_m": float(t2a.max()), "band_m": BAND_OFFSET_M,
            "ok": bool(t2a.max() <= BAND_OFFSET_M and lateral <= 1e-12),
            "why": "float64, on the helper's own return value, recorded by wrapping it. "
                   "The lateral check is what licenses writing the offset as "
                   "R_hips . (0, 0.08, 0): the two UpperLeg rest offsets are mirror "
                   "images in X, so their midpoint is a pure up-axis vector on this rig "
                   "AND on the sized one (tools/head/sized_skeleton.py scales the hip "
                   "half-span in X only). 0.08 is READ, never written.",
        },
        "T2b_the_root_moved_by_exactly_that_and_nothing_else": {
            "max_m": float(t2b.max()), "band_m": BAND_ROOT_F32_M,
            "ok": bool(t2b.max() <= BAND_ROOT_F32_M),
            "why": "BodyTrack.root_translation_m is float32, so this arm cannot be "
                   "checked at 1e-9 as the plan asked; 1e-12 is checked in float64 in "
                   "T2a instead and this one carries float32's own resolution. "
                   "Deviation from the plan, recorded rather than worked around.",
        },
        "T3_only_the_clavicle_chain_moved_pre_projection": {
            "differing_local_rotations": sorted(differ),
            "ok": bool(differ <= (CLAVICLE_CHAIN | HANDS)),
            "contains_the_chain": bool(CLAVICLE_CHAIN <= differ),
            "why": "PRE-PROJECTION. On the DELIVERED track the set is larger and must be: "
                   "`project_generated_foot_contacts` rewrites the FOOT locals inside "
                   "contact runs, and the runs move when the root does. The corrected "
                   "prediction is on disk, in `d2b_bit_identity`.",
        },
    }


# ------------------------------------------------------------------ absolute placement
def hip_absolute_offset(tracks: Path, subject: int) -> dict:
    """THE FIGURE D2b OWNS AND NO ROOT-RELATIVE INSTRUMENT CAN SEE.

    The delivered rig's own hip joints -- FK's UpperLeg midpoint of the SHIPPED track,
    after the ground projection -- minus the captured hip midpoint, in absolute capture
    world. D2's every own-capture figure removed exactly this vector by construction.
    """
    names = json.loads((tracks / f"subject-{subject:02d}.body-track.json").read_text())[
        "joint_names"]
    base = skeleton_for_joint_names(names)
    track = np.load(tracks / f"subject-{subject:02d}.body-track.npz")
    cap_z = np.asarray(track["triangulated_world_positions_z_up_m"], dtype=np.float64)
    world = forward_kinematics_positions(
        np.asarray(track["root_translation_m"], dtype=np.float64),
        np.asarray(track["local_rotations_xyzw"], dtype=np.float64),
        skeleton=base).astype(np.float64)
    to_z = np.stack([world[..., 0], -world[..., 2], world[..., 1]], axis=-1)
    rig_hip = 0.5 * (to_z[:, names.index("LeftUpperLeg")] + to_z[:, names.index("RightUpperLeg")])
    pelvis = 0.5 * (cap_z[:, cm.JOINT_INDEX["left_hip"]] + cap_z[:, cm.JOINT_INDEX["right_hip"]])
    ok = np.isfinite(pelvis).all(axis=1)
    delta = (rig_hip - pelvis)[ok]
    torso_up = (cap_z[:, cm.JOINT_INDEX["neck"]] - pelvis)[ok]
    torso_up = torso_up / np.linalg.norm(torso_up, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(torso_up[:, 2], -1.0, 1.0)))
    norm = np.linalg.norm(delta, axis=1)
    horizontal = np.linalg.norm(delta[:, :2], axis=1)

    def q(v, scale=1000.0):
        return {"median": round(float(np.median(v)) * scale, 2),
                "p5": round(float(np.percentile(v, 5)) * scale, 2),
                "p95": round(float(np.percentile(v, 95)) * scale, 2)}

    return {
        "reference": REF_ABSOLUTE,
        "frames": int(ok.sum()),
        "component_x_mm": q(delta[:, 0]), "component_y_mm": q(delta[:, 1]),
        "component_z_up_mm": q(delta[:, 2]),
        "horizontal_mm": q(horizontal), "norm_mm": q(norm),
        "pelvis_tilt_from_vertical_deg": {
            "median": round(float(np.median(tilt)), 2),
            "p95": round(float(np.percentile(tilt, 95)), 2),
            "max": round(float(tilt.max()), 2)},
        "correlation_with_pelvis_tilt": {
            "horizontal": round(float(np.corrcoef(horizontal, tilt)[0, 1]), 4),
            "vertical": round(float(np.corrcoef(delta[:, 2], tilt)[0, 1]), 4),
            "norm": round(float(np.corrcoef(norm, tilt)[0, 1]), 4)},
        "note": "the vertical component is dominated by the projection's single uncapped "
                "hoist, which is set by the WORST frame; the horizontal component and the "
                "tilt correlation are what the per-frame root fix moves.",
    }


LEGS_SIX = ("left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle")


def rung11_block(arms: dict, rng) -> dict:
    """Rung 11's own statistic on three builds, LEGS AND ARMS SEPARATED.

    `arms` maps a label to the output of `scoreboard_errors`. Every margin is a paired
    moving-block bootstrap on IDENTICAL drawn frames, block 15, 2000 draws, because the
    per-frame series here has lag-1 autocorrelation near 0.99 and ordinary resampling
    would be invalid (CLAUDE.md).
    """
    out: dict = {"reference": REF_MAMMA, "subjects": {}}
    scopes = (("all", list(sb.PAIRS)), ("arms", list(ARMS_SIX)), ("legs", list(LEGS_SIX)))
    for s in ("subject_00", "subject_01"):
        entry: dict = {}
        for arm in ("canon", "sized"):
            for scope, names in scopes:
                stat, _ = median_of_joint_medians(names)
                stacks = {label: np.stack([data[s][arm][n] for n in names], axis=1)
                          for label, data in arms.items()}
                row = {f"{label}_mm": round(stat(v), 2) for label, v in stacks.items()}
                if "delivered_before_D2" in stacks and "D2b" in stacks:
                    row["margin_delivered_minus_D2b"] = block_bootstrap(
                        stacks["delivered_before_D2"], stacks["D2b"], rng, statistic=stat)
                if "D2" in stacks and "D2b" in stacks:
                    row["margin_D2_minus_D2b"] = block_bootstrap(
                        stacks["D2"], stacks["D2b"], rng, statistic=stat)
                entry[f"{arm}_{scope}"] = row
        out["subjects"][s] = entry
    return out


# ------------------------------------------------------------------ the D2b block
D2B_PREREGISTERED = {
    "1_faithful_swap": (
        "The SHIPPED derivation must reproduce the instrument-side variant D2 measured "
        "(gate-d2-prior.json, root_placement_variant) to 0.00 mm on every figure it "
        "reported: round trip 0.51 / 0.08 canonical and 0.07 / 0.04 sized, delivered arms "
        "on our capture 50.62 / 30.28, hoist 83.01 / 49.06. Anything else means the src "
        "edit is not what was measured."),
    "2_theorems": (
        "FK's UpperLeg midpoint equals the captured hip midpoint per frame; the root moved "
        "by exactly R_hips . (0, 0.08, 0) and by nothing else; PRE-PROJECTION no local "
        "rotation outside the clavicle chain moved. ON DISK the differing set must ALSO "
        "contain the two FEET -- `project_generated_foot_contacts` rewrites foot locals "
        "inside contact runs and the runs move with the root -- and the facing instrument "
        "still cannot move, because it reads Hips, chest, Neck, Head and the mesh nose and "
        "not one of those is a foot."),
    "3_absolute_placement": (
        "MECHANISM, WRITTEN BEFORE THE NUMBERS. The projection adds ONE uncapped scalar "
        "vertical hoist for the whole take, set by the single worst frame, plus small "
        "capped per-contact corrections. The root fix adds R_hips . (0, 0.08, 0) PER "
        "FRAME. So before D2b the delivered hip offset is `hoist.z + correction + "
        "R.(0,-0.08,0)`: a horizontal part near 80*sin(tilt) and a vertical part that the "
        "hoist has already absorbed ON AVERAGE. Prediction: after D2b the HORIZONTAL "
        "component and the tilt CORRELATION collapse, while the median VERTICAL component "
        "barely moves, because the hoist re-solves to the new geometry (142 -> 83 mm and "
        "110 -> 49 mm were measured in D2). The remaining vertical is the legs' surplus "
        "length -- canonical thigh 430 mm against ~400 measured -- and is D5's, not D2b's. "
        "D2b MOVES the placement error from the converter to the projection, where it is "
        "now visible; it does not remove it. RUNG 11: the legs should improve MODESTLY and "
        "NOT to zero -- tens of millimetres, concentrated on tilted frames, with the "
        "vertical residual unchanged -- because only the horizontal, tilt-dependent term "
        "leaves. The arms ride the same root and should improve at least as much, since "
        "they also carry D2's clavicle gain."),
    "4_the_sized_replay_defect": (
        "PRE-REGISTERED AS A RISK: the root is computed from the rig's own UpperLeg rest "
        "offset, while rung 11's `sized` arm REPLAYS canonical rotations and root on a "
        "skeleton whose UpperLeg offset is scaled, so the replayed hips could be misplaced "
        "by R.(0, 0.08*(k-1), 0). Measured either way, and re-solved reported beside "
        "replayed."),
    "5_temporal": (
        "D2b carries NO temporal band. The clavicle-chain frames over the physical ceiling "
        "of 26.67 deg/frame are reported before / D2 / D2b on both rigs, with per-joint "
        "median / p95 / max steps, as D2c's baseline. D2's instrument-side variant "
        "measured 40 and 33 on the canonical rig; the shipped derivation must reproduce "
        "that."),
    "6_contacts": (
        "The foot contacts change with the root (performer 0 went [47, 42] -> [37, 60] "
        "under the variant). Contact counts, foot heights, the applied hoist and the "
        "penetration after projection are reported before and after; penetration after "
        "must be 0.000 mm in every case."),
}


def d2b_subject_block(subject, mamma_body_id) -> dict:
    src = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")[
        "triangulated_world_positions_z_up_m"]
    src = src[np.isfinite(src).all(axis=(1, 2))]
    points_y = rc.Y_UP_FROM_Z_UP(src)
    skel_sz, _ = sized_skeleton(DETAILED_HUMANOID, src)
    legacy = legacy_anchor(points_y)
    entry: dict = {}

    # ---- 1. the faithful swap, and the delivered / round-trip arms on both rigs
    rigs: dict = {}
    tracks_for_temporal: dict = {}
    for rig_label, skel in (("canonical", None), ("sized_resolved", skel_sz)):
        rig: dict = {}
        for arm_label, off in (("D2_alone", zero_offset), ("D2b_shipped", None)):
            pre, projected, fk, call = solve_d2b(src, skel, off)
            rt, _, _, _ = round_trip_d2b(src, skel, off)
            diag = call["diagnostics"]
            before = np.asarray(call["incoming"].root_translation_m, dtype=np.float64)
            after = np.asarray(call["projected"].root_translation_m, dtype=np.float64)
            applied = np.linalg.norm(after - before, axis=1)
            rig[arm_label] = {
                "delivered_on_our_capture": groups(rc.score(fk, points_y,
                                                            skel or DETAILED_HUMANOID)),
                "roundtrip": groups(rt),
                "roundtrip_per_landmark_arms": per_landmark_arms(rt),
                "ground_projection": {
                    "diagnostics": diag, "kwargs": call["kwargs"],
                    "feet": foot_heights(projected),
                    "applied_root_correction_median_mm": round(
                        float(np.median(applied)) * 1000.0, 2),
                    "applied_root_correction_max_mm": round(float(applied.max()) * 1000.0, 2),
                },
            }
            tracks_for_temporal[f"{rig_label}_{arm_label}"] = projected
        rigs[rig_label] = rig
    entry["rigs"] = rigs

    # ---- the pre-D2 arm, for the three-way temporal baseline only
    for rig_label, skel in (("canonical", None), ("sized_resolved", skel_sz)):
        _, projected, _, _ = solve_d2b(src, skel, zero_offset, helper=legacy)
        tracks_for_temporal[f"{rig_label}_before_D2"] = projected

    entry["temporal_baseline_for_D2c"] = {
        "ceiling_deg_per_frame": round(PHYSICAL_CEILING_DEG_PER_FRAME, 2),
        "reference": "a human joint's peak angular rate, ~800 deg/s, at 30 fps",
        "canonical_before_D2_vs_D2": temporal_block(
            tracks_for_temporal["canonical_before_D2"],
            tracks_for_temporal["canonical_D2_alone"]),
        "canonical_D2_vs_D2b": temporal_block(
            tracks_for_temporal["canonical_D2_alone"],
            tracks_for_temporal["canonical_D2b_shipped"]),
        "sized_D2_vs_D2b": temporal_block(
            tracks_for_temporal["sized_resolved_D2_alone"],
            tracks_for_temporal["sized_resolved_D2b_shipped"]),
        "note": "no band. D2b changes where the rig SITS, and the clavicle direction with "
                "it; whether the arm still travels like a body is D2c's question and this "
                "is its baseline.",
    }

    # ---- 2. the theorems
    entry["theorems_canonical"] = d2b_theorems(src, None, "canonical")
    entry["theorems_sized"] = d2b_theorems(src, skel_sz, "sized")

    # ---- the controls, every one through the identical code path
    controls: dict = {}
    e_sign, _, _, _ = round_trip_d2b(src, None, sign_flipped_offset)
    _, _, fk_sign, _ = solve_d2b(src, None, sign_flipped_offset)
    controls["b_sign_flipped"] = {
        "roundtrip": groups(e_sign),
        "delivered_on_our_capture": groups(rc.score(fk_sign, points_y, DETAILED_HUMANOID)),
        "band_arms_median_mm": BAND_D2B_ARMS_MM,
        "must": "FAIL",
        "why": "the same magnitude, the same frame, the opposite direction. It separates "
               "'a lift of the right size' from 'the right lift'.",
    }

    sweep: dict = {}
    for mm in LIFT_SWEEP_MM:
        e, _, _, _ = round_trip_d2b(src, None, along_hips_up(mm / 1000.0))
        g = groups(e)
        sweep[f"{mm:.0f}"] = {"arms_median_mm": g["arms"], "legs_median_mm": g["legs"],
                              "torso_median_mm": g["torso"]}
    passing = [k for k, v in sweep.items() if v["arms_median_mm"] <= BAND_D2B_ARMS_MM]
    skeleton_own_mm = round(-500.0 * (float(DETAILED_HUMANOID.joints[
        DETAILED_HUMANOID.index("LeftUpperLeg")].rest_translation_m[1])
        + float(DETAILED_HUMANOID.joints[
            DETAILED_HUMANOID.index("RightUpperLeg")].rest_translation_m[1])), 1)
    controls["c_lift_sweep_along_the_hips_up_axis"] = {
        "swept_mm": "0 .. 160 in steps of 10, along the hips' own up axis",
        "per_lift": sweep,
        "values_passing_the_band": passing,
        "the_skeletons_own_value_mm": skeleton_own_mm,
        "band_arms_median_mm": BAND_D2B_ARMS_MM,
        "why": "NO GATE A CONSTANT CAN PASS, in its strongest available form: the band is "
               "reached only at the value the skeleton itself supplies, which the shipped "
               "code READS from `rest` and never writes down.",
    }

    e_wv, _, _, _ = round_trip_d2b(src, None, world_vertical(skeleton_own_mm / 1000.0))
    _, _, fk_wv, _ = solve_d2b(src, None, world_vertical(skeleton_own_mm / 1000.0))
    err_wv = rc.score(fk_wv, points_y, DETAILED_HUMANOID)
    pelvis = 0.5 * (src[:, cm.JOINT_INDEX["left_hip"]] + src[:, cm.JOINT_INDEX["right_hip"]])
    up = src[:, cm.JOINT_INDEX["neck"]] - pelvis
    up = up / np.linalg.norm(up, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1.0, 1.0)))
    steep = tilt >= np.percentile(tilt, 75.0)
    _, _, fk_ship, _ = solve_d2b(src, None, None)
    err_ship = rc.score(fk_ship, points_y, DETAILED_HUMANOID)
    controls["d_world_vertical_instead_of_the_hips_frame"] = {
        "lift_mm": skeleton_own_mm,
        "roundtrip": groups(e_wv),
        "delivered_arms_median_mm": round(float(np.median(np.concatenate(
            [err_wv[n] for n in rc.GROUPS["arms"]]))), 2),
        "delivered_arms_on_the_most_tilted_quartile_mm": round(float(np.median(
            np.concatenate([err_wv[n][steep] for n in rc.GROUPS["arms"]]))), 2),
        "shipped_arms_on_the_most_tilted_quartile_mm": round(float(np.median(
            np.concatenate([err_ship[n][steep] for n in rc.GROUPS["arms"]]))), 2),
        "delivered_legs_on_the_most_tilted_quartile_mm": round(float(np.median(
            np.concatenate([err_wv[n][steep] for n in rc.GROUPS["legs"]]))), 2),
        "pelvis_tilt_quartile_threshold_deg": round(float(np.percentile(tilt, 75.0)), 2),
        "band_arms_median_mm": BAND_D2B_ARMS_MM,
        "must": "FAIL, and hardest on the tilted frames",
        "why": "THE PLAUSIBLE SHORTCUT. 'Lift the rig 80 mm' is right only while the "
               "pelvis is upright; the derivation is in the HIPS' frame and this is not. "
               "It is the control that discriminates a derivation from a number.",
    }
    controls["e_legs_and_torso_are_0.00_under_every_variant"] = {
        "sweep": sorted({v["legs_median_mm"] for v in sweep.values()}
                        | {v["torso_median_mm"] for v in sweep.values()}),
        "sign_flipped": [groups(e_sign)["legs"], groups(e_sign)["torso"]],
        "world_vertical": [groups(e_wv)["legs"], groups(e_wv)["torso"]],
        "shipped": [rigs["canonical"]["D2b_shipped"]["roundtrip"]["legs"],
                    rigs["canonical"]["D2b_shipped"]["roundtrip"]["torso"]],
        "why": "the legs are measured landmark-to-landmark and no root or origin enters "
               "them. 0.00 here is NOT evidence the placement is right -- it is evidence "
               "that a translation-invariant chain scored root-relatively cannot see a "
               "translation. That is D2b's whole point.",
    }
    entry["controls"] = controls

    # ---- 4. the sized replay, and whether the pre-registered defect occurs
    canon_mid = 0.5 * (np.asarray(DETAILED_HUMANOID.joints[
        DETAILED_HUMANOID.index("LeftUpperLeg")].rest_translation_m, dtype=np.float64)
        + np.asarray(DETAILED_HUMANOID.joints[
            DETAILED_HUMANOID.index("RightUpperLeg")].rest_translation_m, dtype=np.float64))
    sized_mid = 0.5 * (np.asarray(skel_sz.joints[
        skel_sz.index("LeftUpperLeg")].rest_translation_m, dtype=np.float64)
        + np.asarray(skel_sz.joints[
            skel_sz.index("RightUpperLeg")].rest_translation_m, dtype=np.float64))
    entry["sized_replay_defect"] = {
        "canonical_upperleg_midpoint_rest_m": [float(v) for v in canon_mid],
        "sized_upperleg_midpoint_rest_m": [float(v) for v in sized_mid],
        "canonical_hips_rest_m": [float(v) for v in np.asarray(
            DETAILED_HUMANOID.joints[DETAILED_HUMANOID.index("Hips")].rest_translation_m)],
        "sized_hips_rest_m": [float(v) for v in np.asarray(
            skel_sz.joints[skel_sz.index("Hips")].rest_translation_m)],
        "misplacement_of_the_replayed_sized_hips_mm": round(
            float(np.linalg.norm(sized_mid - canon_mid)) * 1000.0, 4),
        "occurs": bool(np.linalg.norm(sized_mid - canon_mid) > 1e-9),
        "finding": "PRE-REGISTERED AND DID NOT OCCUR, and the mechanism is the reason, not "
                   "luck: `tools/head/sized_skeleton.py` scales the hip half-span in X "
                   "ONLY -- `rest_translation_m=(off[0]*k, off[1], off[2])` -- and never "
                   "touches `Hips`. The two UpperLeg offsets are mirror images in X, so "
                   "their MIDPOINT is unchanged by any hip-span sizing and the replayed "
                   "sized rig's hips land exactly where the canonical ones do. The check "
                   "stays in the gate: a future sizing that scaled the UpperLeg Y would "
                   "reintroduce the defect and this assertion would catch it.",
    }

    # ---- arm B, on its own reference
    pred = np.load(rc.MA3D / f"verts_joints_body_id-{mamma_body_id:02d}.npz",
                   allow_pickle=True)["pred_joints"].astype(np.float64)
    pred = pred[np.isfinite(pred).all(axis=(1, 2))]
    m_src = rc.adapt_mamma(pred)
    m_points_y = rc.Y_UP_FROM_Z_UP(m_src)
    m_skel, _ = sized_skeleton(DETAILED_HUMANOID, m_src)
    arm_b: dict = {"reference": REF_ARMB, "mamma_body_id": f"body_id-{mamma_body_id:02d}"}
    for rig_label, skel in (("canonical", None), ("sized_resolved", m_skel)):
        for arm_label, off in (("D2_alone", zero_offset), ("D2b_shipped", None)):
            _, _, fk_b, _ = solve_d2b(m_src, skel, off)
            arm_b[f"{rig_label}_{arm_label}"] = groups(
                rc.score(fk_b, m_points_y, skel or DETAILED_HUMANOID))
    arm_b["not_comparable_with"] = ("the own-capture arms -- a different body, different "
                                    "poses, a different joint convention. Never one axis.")
    entry["arm_B_mamma_joints_in"] = arm_b
    return entry


def d2b_bit_identity() -> dict:
    """The D2 rebuild against the D2b rebuild, on disk. Both are delivered tracks."""
    out: dict = {"held": True, "checks": {},
                 "compares": f"{REBUILD} (D2 alone) against {REBUILD_ROOT} (D2b)"}

    def check(name, ok, detail=""):
        out["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            out["held"] = False

    feet = {"LeftFoot", "RightFoot"}
    for s in (0, 1):
        a = np.load(REBUILD / f"subject-{s:02d}.body-track.npz")
        b = np.load(REBUILD_ROOT / f"subject-{s:02d}.body-track.npz")
        names = json.loads((REBUILD / f"subject-{s:02d}.body-track.json").read_text())[
            "joint_names"]
        for key in ("triangulated_world_positions_z_up_m",
                    "raw_triangulated_world_positions_z_up_m", "ticks"):
            same = (a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
                    and a[key].tobytes() == b[key].tobytes())
            check(f"subject_{s:02d}.{key}_byte_identical", same, str(a[key].shape))
        check(f"subject_{s:02d}.root_translation_m_MUST_differ",
              not np.array_equal(a["root_translation_m"], b["root_translation_m"]),
              "the whole point of the step")
        qa, qb = a["local_rotations_xyzw"], b["local_rotations_xyzw"]
        differ = {names[i] for i in range(len(names))
                  if not np.array_equal(qa[:, i], qb[:, i])}
        out["checks"][f"subject_{s:02d}.differing_local_rotations"] = {
            "ok": True, "detail": sorted(differ)}
        check(f"subject_{s:02d}.differing_locals_are_the_clavicle_chain_the_hands_and_the_feet",
              differ <= (CLAVICLE_CHAIN | HANDS | feet),
              f"unexpected: {sorted(differ - CLAVICLE_CHAIN - HANDS - feet)}")
        check(f"subject_{s:02d}.no_torso_neck_head_or_leg_local_moved",
              not (differ & {"Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
                             "LeftEye", "RightEye", "LeftUpperLeg", "RightUpperLeg",
                             "LeftLowerLeg", "RightLowerLeg", "LeftToes", "RightToes"}),
              sorted(differ))
        contacts = {"D2": [int(a["foot_contacts"][:, i].sum()) for i in (0, 1)],
                    "D2b": [int(b["foot_contacts"][:, i].sum()) for i in (0, 1)]}
        out["checks"][f"subject_{s:02d}.foot_contact_frames"] = {"ok": True,
                                                                 "detail": str(contacts)}

    for name in sorted(p.name for p in (REBUILD / "work").glob("*observations.jsonl")):
        x = (REBUILD / "work" / name).read_bytes()
        y = (REBUILD_ROOT / "work" / name).read_bytes()
        check(f"work/{name}_byte_identical", x == y, f"{len(x)} vs {len(y)} bytes")
    out["means"] = (
        "The FEET are in the differing set and MUST be: `project_generated_foot_contacts` "
        "rewrites the foot locals inside contact runs, and the runs move when the root "
        "does. The plan's set was the PRE-PROJECTION one (asserted separately, in "
        "`theorems_*.T3`). The facing instrument still cannot move under this change: it "
        "reads the Hips, chest, Neck and Head rotations and the mesh nose, not one of "
        "which is a foot, and the triangulated positions, which are the same bytes. It is "
        "measured anyway -- facing-d2b.json.")
    return out


def d2_regression_check(report) -> dict:
    """Did extending this gate move any D2 number? It must not.

    Every pre-existing block now runs inside `with root_offset(zero_offset)`, which is
    supposed to be the D2 converter exactly. This compares the result against
    `gate-d2-prior.json` -- the file this run replaces -- band by band.
    """
    if not PRIOR_GATE.exists():
        return {"ok": None, "why": "no prior gate.json on disk to compare against"}
    prior = json.loads(PRIOR_GATE.read_text())
    rows, worst = [], 0.0
    index = {r["band"]: r for r in prior.get("gate", [])}
    for r in report["gate"]:
        old = index.get(r["band"])
        if old is None:
            continue
        for field in ("before", "after"):
            a, b = old.get(field), r.get(field)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                delta = abs(float(a) - float(b))
                worst = max(worst, delta)
                if delta > 0.005:
                    rows.append({"band": r["band"], "field": field, "prior": a, "now": b})
    return {"ok": not rows, "max_abs_difference": round(worst, 6), "moved": rows,
            "band": 0.005,
            "why": "the D2 bands are measured through the zero-offset helper, which is the "
                   "pre-D2b converter through the identical code path. If any of them "
                   "moved, this extension rewrote D2's committed figures and the report "
                   "above cannot be read as D2's."}


def _g(node, *path, default=None):
    """Nested `dict.get`, so a missing prior figure reads as None instead of raising."""
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _zero_offset_root_variant(subject, mamma_body_id) -> dict:
    """D2's root-placement variant, re-measured through the zero-offset helper.

    It was written when `src/` had no root fix. Running it unchanged against a shipped
    fix would double-apply the offset, so the D2 converter is restored around it exactly
    as it is around `subject_block`.
    """
    with root_offset(zero_offset):
        return root_variant_block(subject, mamma_body_id)


SILHOUETTE_D2B = OUT_DIR / "silhouette-d2b.json"
SILHOUETTE_D2 = OUT_DIR / "silhouette-d2.json"
SILHOUETTE_COMMITTED = ROOT / "artifacts/compare/silhouette.json"
SILHOUETTE_REGEN = (
    "for arm in delivery delivery-root: .venv/bin/python tools/compare/silhouette.py "
    "--delivery artifacts/compare/d2-clavicle/$arm "
    "--work artifacts/compare/d2-clavicle/silhouette-work-$arm "
    "--out artifacts/compare/d2-clavicle/silhouette-$arm.json  "
    "(written as silhouette-d2.json and silhouette-d2b.json). DEVIATION FROM THE PLAN, "
    "recorded: the plan said `--work <rebuild>/work`; a dedicated work directory is used "
    "instead, seeded with `artifacts/compare/i6`'s MAMMA-derived caches "
    "(masks-960x540*.npz, mean-body-0*.npy). Those read MAMMA's masks and mean bodies and "
    "never our track -- the ORACLE bit-identity check below is the proof that reusing "
    "them changed nothing.")


def _silhouette_rows(report, arm_name="ours_delivered") -> dict:
    """One arm's IoU / precision / recall per camera per subject, flattened.

    `silhouette.py` writes `arms.<arm>.<camera>.<subject>.{iou,precision,recall}.median`.
    Precision and recall travel with the IoU and are never collapsed into it: the
    degenerate a mesh instrument actually produces -- something too big -- buys recall
    with precision, which an IoU alone would hide.
    """
    rows: dict = {}
    arm = _g(report, "arms", arm_name, default={})
    for camera, per_camera in (arm or {}).items():
        if not isinstance(per_camera, dict):
            continue
        for subject, cell in per_camera.items():
            if not isinstance(cell, dict) or "iou" not in cell:
                continue
            rows[f"{camera}/{subject}"] = {
                "iou_median": round(float(_g(cell, "iou", "median", default=float("nan"))), 4),
                "precision_median": round(
                    float(_g(cell, "precision", "median", default=float("nan"))), 4),
                "recall_median": round(
                    float(_g(cell, "recall", "median", default=float("nan"))), 4),
                "frames_scored": cell.get("frames_scored"),
            }
    return rows


def _delivered_shift() -> dict:
    """How far the delivered rig actually MOVED between the two rebuilds, in metres.

    A measurement, not a mechanism. The silhouette scores a projection of the whole mesh,
    so before anyone reasons about why an overlap changed, the size of the displacement it
    is pricing belongs on the page. Reported in the rig's own Y-up frame, per joint, as the
    median signed vertical move and the median displacement norm.
    """
    if not (REBUILD_ROOT / "subject-00.body-track.npz").exists() or \
            not (REBUILD / "subject-00.body-track.npz").exists():
        return {}
    out: dict = {"units": "mm, in the rig's Y-up world, D2b minus D2 (positive = higher)"}
    for subject in (0, 1):
        names = json.loads((REBUILD / f"subject-{subject:02d}.body-track.json").read_text())[
            "joint_names"]
        base = skeleton_for_joint_names(names)
        fk = {}
        for label, path in (("D2", REBUILD), ("D2b", REBUILD_ROOT)):
            t = np.load(path / f"subject-{subject:02d}.body-track.npz")
            fk[label] = forward_kinematics_positions(
                np.asarray(t["root_translation_m"], dtype=np.float64),
                np.asarray(t["local_rotations_xyzw"], dtype=np.float64),
                skeleton=base).astype(np.float64)
        delta = fk["D2b"] - fk["D2"]
        row = {}
        for name in ("Hips", "UpperChest", "Head", "LeftFoot", "RightFoot",
                     "LeftUpperLeg", "LeftHand"):
            i = names.index(name)
            row[name] = {
                "vertical_median_mm": round(float(np.median(delta[:, i, 1])) * 1000.0, 2),
                "displacement_median_mm": round(
                    float(np.median(np.linalg.norm(delta[:, i], axis=1))) * 1000.0, 2)}
        out[f"subject_{subject:02d}"] = row
    return out


SILHOUETTE_VS_TILT = OUT_DIR / "silhouette-vs-tilt.json"
SILHOUETTE_PARTWISE = OUT_DIR / "silhouette-partwise.json"


def silhouette_partwise_block() -> dict:
    """The part-wise cut, folded in rather than recomputed.

    `tools/compare/silhouette_partwise.py` writes it. It splits the mesh by dominant skin
    weight and scores ARMS and TORSO+LEGS separately, which is the only way this lane has
    to isolate D2b's OWN silhouette cost: the torso+legs vertices carry no clavicle-chain
    weight, so between D2 and D2b they move by exactly the root shift and nothing else.
    It also carries the control I6 never had -- a body with both arms folded across the
    chest, scored against the same person masks.
    """
    if not SILHOUETTE_PARTWISE.exists():
        return {"available": False,
                "why_absent": f"{SILHOUETTE_PARTWISE} not written yet",
                "regenerate": "PYTHONPATH=$PWD/src .venv/bin/python "
                              "tools/compare/silhouette_partwise.py"}
    report = json.loads(SILHOUETTE_PARTWISE.read_text())
    report["available"] = True
    report["figure"] = "artifacts/compare/d2-clavicle/silhouette-partwise.png"
    return report


def silhouette_vs_tilt_block() -> dict:
    """The follow-up measurement, folded in rather than recomputed.

    `tools/compare/silhouette_vs_tilt.py` writes it; this gate does not own it and does
    not re-derive it. It answers the one question section 12.6 left open -- whether the
    silhouette's fall tracks trunk tilt -- and it is instrument-only.
    """
    if not SILHOUETTE_VS_TILT.exists():
        return {"available": False,
                "why_absent": f"{SILHOUETTE_VS_TILT} not written yet",
                "regenerate": "PYTHONPATH=$PWD/src .venv/bin/python "
                              "tools/compare/silhouette_vs_tilt.py"}
    report = json.loads(SILHOUETTE_VS_TILT.read_text())
    report["available"] = True
    report["figure"] = "artifacts/compare/d2-clavicle/silhouette-vs-tilt.png"
    return report


def silhouette_block() -> dict:
    """I6 on the D2b rebuild against the committed I6 report. SEPARATE REFERENCE.

    A silhouette is scored against MAMMA's SAM2 masks -- pixels of the actual footage --
    and it is the only figure in this gate whose reference is not another model. It is
    also blind to depth, to a left/right mirror of a fore-aft symmetric pose, and to
    everything inside the outline, so it cannot separate a shape error from a pose error.
    """
    out: dict = {"reference": REF_SILHOUETTE, "regenerate": SILHOUETTE_REGEN,
                 "available": SILHOUETTE_D2B.exists()}
    if not SILHOUETTE_D2B.exists():
        out["why_absent"] = f"{SILHOUETTE_D2B} not written yet"
        return out
    after = json.loads(SILHOUETTE_D2B.read_text())
    out["after_per_camera_subject"] = _silhouette_rows(after)
    out["after_oracle_mamma_mesh"] = _silhouette_rows(after, "ORACLE_mamma_mesh")
    out["after_control_mean_body"] = _silhouette_rows(after, "control_mean_body")
    if SILHOUETTE_D2.exists():
        out["d2_alone_per_camera_subject"] = _silhouette_rows(
            json.loads(SILHOUETTE_D2.read_text()))
        out["d2_alone_note"] = ("the D2 rebuild, so before/D2/D2b is a three-way "
                                "attribution and not one difference carrying two changes.")
    out["how_far_the_delivered_rig_moved"] = _delivered_shift()
    if SILHOUETTE_COMMITTED.exists():
        before = json.loads(SILHOUETTE_COMMITTED.read_text())
        out["before_per_camera_subject"] = _silhouette_rows(before)
        out["before_oracle_mamma_mesh"] = _silhouette_rows(before, "ORACLE_mamma_mesh")
        out["before_control_mean_body"] = _silhouette_rows(before, "control_mean_body")
        out["oracle_and_control_must_be_unchanged"] = {
            "max_abs_oracle_iou_difference": max(
                (abs(v["iou_median"] - out["after_oracle_mamma_mesh"][k]["iou_median"])
                 for k, v in out["before_oracle_mamma_mesh"].items()
                 if k in out["after_oracle_mamma_mesh"]), default=None),
            "why": "the oracle is MAMMA's own mesh and the control is a mean body; "
                   "neither reads our track, so a rebuild of ours must leave them "
                   "bit-stable. Anything else means the two runs are not comparable.",
        }
        out["before_note"] = ("the COMMITTED artifacts/compare/silhouette.json, which was "
                              "measured on the delivered build -- pre-D2 and pre-D2b. It "
                              "is the only before arm available without a third render.")
        pairs = [(b.get("iou_median"), out["after_per_camera_subject"][k].get("iou_median"))
                 for k, b in out["before_per_camera_subject"].items()
                 if k in out["after_per_camera_subject"]]
        pairs = [(a, b) for a, b in pairs if isinstance(a, (int, float))
                 and isinstance(b, (int, float))]
        if pairs:
            out["before_iou"] = round(float(np.median([a for a, _ in pairs])), 4)
            out["after_iou"] = round(float(np.median([b for _, b in pairs])), 4)
            out["cells_where_iou_fell"] = sum(1 for a, b in pairs if b < a)
            out["cells"] = len(pairs)
            if "d2_alone_per_camera_subject" in out:
                d2 = [out["d2_alone_per_camera_subject"][k]["iou_median"]
                      for k in out["before_per_camera_subject"]
                      if k in out["d2_alone_per_camera_subject"]]
                out["d2_alone_iou"] = round(float(np.median(d2)), 4) if d2 else None
            out["note"] = ("median over the eight camera-subject cells of the per-cell "
                           "median IoU. Precision and recall are reported per cell and "
                           "never collapsed into IoU: the degenerate a mesh instrument "
                           "actually produces -- something too big -- buys recall with "
                           "precision.")
    return out


def d2b_verdicts(report) -> list:
    rows = []

    def row(name, band, before, after, interval, ok, reference, note=""):
        rows.append({"band": name, "band_value": band, "before": before, "after": after,
                     "interval": interval, "verdict": "PASS" if ok else "FAIL",
                     "reference": reference, "note": note})

    d2b = report["d2b_root_placement"]
    chain6 = ("LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
              "LeftLowerArm", "RightLowerArm")

    def over_ceiling(block, when):
        return sum(_g(block, "joints", n, when, "frames_over_the_physical_ceiling",
                      default=0) for n in chain6)

    for s in ("subject_00", "subject_01"):
        d = d2b["subjects"][s]
        canon = d["rigs"]["canonical"]
        sized = d["rigs"]["sized_resolved"]

        # ---- FAITHFUL SWAP. The reference arm is THIS RUN's `root_placement_variant`,
        # which is recomputed from source under `zero_offset` on every run and is therefore
        # regenerable from a fresh checkout. `gate-d2-prior.json` is a gitignored artifact
        # and is used only by `d2_regression_check`, which degrades to `ok: null` without it.
        pv = _g(report, "root_placement_variant", "subjects", s, default={})
        pairs = [
            ("round trip, canonical", _g(pv, "canonical", "D2_plus_root", "roundtrip", "arms"),
             canon["D2b_shipped"]["roundtrip"]["arms"]),
            ("round trip, sized", _g(pv, "sized_resolved", "D2_plus_root", "roundtrip", "arms"),
             sized["D2b_shipped"]["roundtrip"]["arms"]),
            ("delivered arms, canonical",
             _g(pv, "canonical", "D2_plus_root", "delivered_on_our_capture", "arms"),
             canon["D2b_shipped"]["delivered_on_our_capture"]["arms"]),
            ("delivered arms, sized",
             _g(pv, "sized_resolved", "D2_plus_root", "delivered_on_our_capture", "arms"),
             sized["D2b_shipped"]["delivered_on_our_capture"]["arms"]),
            ("hoist, canonical (the projection's uncapped vertical term, mm)",
             round(1000.0 * _g(pv, "canonical", "D2_plus_root", "ground_projection",
                               "diagnostics", "ground_penetration_before_m", default=0.0), 2),
             round(1000.0 * _g(canon, "D2b_shipped", "ground_projection", "diagnostics",
                               "ground_penetration_before_m", default=0.0), 2)),
            ("clavicle-chain frames over the physical ceiling, canonical",
             over_ceiling(_g(pv, "canonical", "temporal_D2_alone_vs_D2_plus_root",
                             default={}), "after"),
             over_ceiling(_g(d, "temporal_baseline_for_D2c", "canonical_D2_vs_D2b",
                             default={}), "after")),
        ]
        worst = max((abs(a - b) for _, a, b in pairs if a is not None and b is not None),
                    default=None)
        row(f"FAITHFUL SWAP. the shipped derivation reproduces D2's instrument-side "
            f"variant, {s}", "<= 0.005 mm (and exactly, on the integer frame counts) on "
            "every figure D2 reported",
            None, worst, None, worst is not None and worst <= 0.005, REF_ROUNDTRIP,
            "candidate and control differ in ONE expression and nothing else; the "
            "reference arm is recomputed from source in this same run under the "
            "zero-offset helper, so it regenerates. "
            + "; ".join(f"{k}: variant {a} vs shipped {b}" for k, a, b in pairs))

        # ---- THEOREMS
        for label, key in (("canonical", "theorems_canonical"), ("sized", "theorems_sized")):
            t = d[key]
            for name in ("T1_fk_upperleg_midpoint_is_the_captured_hip_midpoint",
                         "T2a_the_offset_is_R_hips_times_the_skeletons_own_half_drop",
                         "T2b_the_root_moved_by_exactly_that_and_nothing_else",
                         "T3_only_the_clavicle_chain_moved_pre_projection"):
                row(f"THEOREM {name.split('_')[0]}, {label} rig, {s}",
                    t[name].get("band_m", "set equality"), None,
                    t[name].get("max_m"), None, t[name]["ok"],
                    "the rig's own geometry, asserted not measured", t[name].get("why", ""))

        # ---- THE ROUND TRIP
        for label, rig in (("canonical", canon), ("sized", sized)):
            worst_lm = max(rig["D2b_shipped"]["roundtrip_per_landmark_arms"].values())
            row(f"A(D2b). round-trip arms, all six landmarks <= {BAND_D2B_ARMS_MM} mm, "
                f"{label} rig, {s}", f"<= {BAND_D2B_ARMS_MM} mm median",
                rig["D2_alone"]["roundtrip"]["arms"], rig["D2b_shipped"]["roundtrip"]["arms"],
                None, worst_lm <= BAND_D2B_ARMS_MM, REF_ROUNDTRIP,
                "D2's OWN pre-registered band, which D2 failed at 67/79 mm on a premise "
                "that was false. The before arm is D2 alone, through the same code path.")
            row(f"A(D2b). round-trip legs and torso stay 0.00, {label} rig, {s}",
                "== 0.00 mm exact", rig["D2_alone"]["roundtrip"]["legs"],
                rig["D2b_shipped"]["roundtrip"]["legs"], None,
                rig["D2b_shipped"]["roundtrip"]["legs"] == 0.0
                and rig["D2b_shipped"]["roundtrip"]["torso"] == 0.0, REF_ROUNDTRIP,
                "and it proves nothing about placement: a translation-invariant chain "
                "scored root-relatively cannot see a translation.")

        # ---- THE CONTROLS
        c = d["controls"]
        row(f"CONTROL (b). the lift with its sign flipped must FAIL, {s}",
            f"> {BAND_D2B_ARMS_MM} mm", None, c["b_sign_flipped"]["roundtrip"]["arms"],
            None, c["b_sign_flipped"]["roundtrip"]["arms"] > BAND_D2B_ARMS_MM,
            REF_ROUNDTRIP, c["b_sign_flipped"]["why"])
        sw = c["c_lift_sweep_along_the_hips_up_axis"]
        best = min(sw["per_lift"], key=lambda k: sw["per_lift"][k]["arms_median_mm"])
        row(f"CONTROL (c1). the sweep's MINIMUM is at the skeleton's own lift, {s}",
            f"argmin over 0..160 mm == {sw['the_skeletons_own_value_mm']} mm", None,
            float(best), None,
            float(best) == sw["the_skeletons_own_value_mm"], REF_ROUNDTRIP,
            f"the curve: " + ", ".join(f"{k}:{v['arms_median_mm']}"
                                       for k, v in sw["per_lift"].items()))
        row(f"CONTROL (c2). ONLY the skeleton's own lift clears the band, {s}",
            f"exactly one of 17 lifts <= {BAND_D2B_ARMS_MM} mm, and it is "
            f"{sw['the_skeletons_own_value_mm']} mm", None,
            len(sw["values_passing_the_band"]),
            None, sw["values_passing_the_band"] == [f"{sw['the_skeletons_own_value_mm']:.0f}"],
            REF_ROUNDTRIP,
            f"passing lifts: {sw['values_passing_the_band']}. REPORTED AS IT FELL AND NOT "
            "TIGHTENED: where a neighbouring 10 mm step also clears 5 mm, this row FAILS "
            "and says so. The band alone does not uniquely select the derivation on that "
            "subject; c1 and c3 are what do. " + sw["why"])
        wv = c["d_world_vertical_instead_of_the_hips_frame"]
        row(f"CONTROL (d). the same lift as a WORLD vertical must FAIL, {s}",
            f"> {BAND_D2B_ARMS_MM} mm, and worse on the tilted quartile", None,
            wv["roundtrip"]["arms"], None, wv["roundtrip"]["arms"] > BAND_D2B_ARMS_MM,
            REF_ROUNDTRIP,
            f"delivered arms on the most-tilted quartile: {wv['delivered_arms_on_the_most_tilted_quartile_mm']} "
            f"mm against the shipped {wv['shipped_arms_on_the_most_tilted_quartile_mm']} mm. "
            + wv["why"])
        legs = c["e_legs_and_torso_are_0.00_under_every_variant"]
        row(f"CONTROL (e). legs and torso read 0.00 under EVERY variant, {s}",
            "== 0.00 mm", None, max(legs["sweep"] + legs["sign_flipped"]
                                    + legs["world_vertical"] + legs["shipped"]), None,
            max(legs["sweep"] + legs["sign_flipped"] + legs["world_vertical"]
                + legs["shipped"]) == 0.0, REF_ROUNDTRIP, legs["why"])

        # ---- ABSOLUTE PLACEMENT
        abs_block = d2b["absolute_hip_placement"]["subjects"][s]
        for label in ("D2", "D2b"):
            if label not in abs_block:
                continue
        b0, b2 = abs_block.get("D2"), abs_block.get("D2b")
        if b0 and b2:
            row(f"ABSOLUTE. the delivered hip joints against the captured hips, "
                f"HORIZONTAL, {s}", "reported with p5/p95; D2b's own figure, and no "
                "root-relative instrument can see it",
                b0["horizontal_mm"]["median"], b2["horizontal_mm"]["median"],
                [b2["horizontal_mm"]["p5"], b2["horizontal_mm"]["p95"]], True,
                REF_ABSOLUTE,
                f"tilt correlation {b0['correlation_with_pelvis_tilt']['horizontal']} -> "
                f"{b2['correlation_with_pelvis_tilt']['horizontal']}")
            row(f"ABSOLUTE. the same, VERTICAL (capture +Z), {s}",
                "reported. The projection's ONE uncapped hoist dominates it",
                b0["component_z_up_mm"]["median"], b2["component_z_up_mm"]["median"],
                [b2["component_z_up_mm"]["p5"], b2["component_z_up_mm"]["p95"]], True,
                REF_ABSOLUTE,
                "what remains is the legs' surplus length -- canonical thigh 430 mm "
                "against ~400 measured -- and it is D5's. D2b MOVES the placement error "
                "from the converter to the projection, where it is now visible.")

        # ---- RUNG 11
        r11 = d2b.get("rung11", {}).get("subjects", {}).get(s, {})
        for scope in ("all", "arms", "legs"):
            k = f"canon_{scope}"
            if k not in r11:
                continue
            e = r11[k]
            margin = e.get("margin_D2_minus_D2b", {})
            row(f"RUNG 11. agreement with MAMMA, canonical rig, {scope.upper()}, {s}",
                "reported, no band -- MAMMA is a reference, not a target",
                e.get("D2_mm"), e.get("D2b_mm"), margin.get("ci95_mm"), True, REF_MAMMA,
                f"delivered (pre-D2) {e.get('delivered_before_D2_mm')}; "
                f"p(wrong sign) {margin.get('p_wrong_sign')}. Positive margin means D2b "
                "agrees better. ABSOLUTE capture world -- this is the arm that can see "
                "placement at all.")

        # ---- CONTACTS AND HOIST
        for label, rig in (("canonical", canon), ("sized", sized)):
            g0 = rig["D2_alone"]["ground_projection"]
            g2 = rig["D2b_shipped"]["ground_projection"]
            row(f"CONTACTS. penetration after projection is 0, {label} rig, {s}",
                "< 0.001 mm -- the projection drives it to float32's floor, a few "
                "NANOmetres, not to a bit-exact zero",
                round(1000.0 * g0["diagnostics"].get("ground_penetration_after_m", 0.0), 4),
                round(1000.0 * g2["diagnostics"].get("ground_penetration_after_m", 0.0), 4),
                None,
                abs(g2["diagnostics"].get("ground_penetration_after_m", 0.0)) < 1e-6,
                "the rig's own feet against the estimated floor",
                f"contacts {g0['diagnostics'].get('contact_frames')} -> "
                f"{g2['diagnostics'].get('contact_frames')}; applied root correction "
                f"{g0['applied_root_correction_median_mm']} -> "
                f"{g2['applied_root_correction_median_mm']} mm median")

    # ---- CONTROL (c3): a SHIPPED constant is ONE number for the whole pipeline, so the
    # question a tuned lift has to answer is not "does some value pass on this subject" but
    # "does one value pass on BOTH". This is the same formulation D2's scalar sweep used
    # (its cross-subject transfer figures), not a band invented after the fact.
    sweeps = {s: d2b["subjects"][s]["controls"]["c_lift_sweep_along_the_hips_up_axis"]
              for s in ("subject_00", "subject_01")}
    own = sweeps["subject_00"]["the_skeletons_own_value_mm"]
    both = [k for k in sweeps["subject_00"]["per_lift"]
            if all(v["per_lift"][k]["arms_median_mm"] <= BAND_D2B_ARMS_MM
                   for v in sweeps.values())]
    row("CONTROL (c3). no lift OTHER than the skeleton's own clears the band on BOTH "
        "subjects", f"the only lift passing on both is {own} mm", None, len(both), None,
        both == [f"{own:.0f}"], REF_ROUNDTRIP,
        f"lifts clearing {BAND_D2B_ARMS_MM} mm on both subjects: {both}. NO GATE A "
        "CONSTANT CAN PASS, in the form that matters for something that ships: a constant "
        "is ONE number for every performer. 90 mm clears the band on subject 1 (4.64 mm) "
        "and misses it on subject 0 (6.46 mm). The shipped code chooses nothing -- it "
        "READS the value from `rest`, and a differently proportioned rig gets a different "
        "one, which is what `test_root_placement.py::test_the_sized_round_trip_closes_too` "
        "pins.")

    bi = d2b.get("bit_identity")
    if bi:
        row("THEOREM. on disk: positions, raw positions, ticks and every observation file "
            "byte-identical; the differing locals are the clavicle chain, the hands and "
            "the FEET", "set equality", None, None, None, bi["held"],
            "the D2 rebuild against the D2b rebuild", bi["means"])
    sil = d2b.get("silhouette")
    if sil and sil.get("available") and sil.get("before_iou") is not None:
        row("I6. silhouette IoU against MAMMA's SAM2 masks must NOT fall, D2b rebuild",
            "median over the eight camera-subject cells must not be below the committed "
            "I6 figure",
            sil.get("before_iou"), sil.get("after_iou"), None,
            sil.get("after_iou") >= sil.get("before_iou"), REF_SILHOUETTE,
            f"D2 alone: {sil.get('d2_alone_iou')}. IoU fell in "
            f"{sil.get('cells_where_iou_fell')} of {sil.get('cells')} cells, and BOTH "
            f"precision and recall fell in each, so the mesh moved OFF the pixels rather "
            f"than changing size. The ORACLE (MAMMA's own mesh) is bit-identical between "
            f"the two runs "
            f"(max IoU difference "
            f"{_g(sil, 'oracle_and_control_must_be_unchanged', 'max_abs_oracle_iou_difference')}"
            f"), so the two runs are comparable and the fall is real. "
            + sil.get("note", ""))
    svt = d2b.get("silhouette_vs_tilt") or {}
    if svt.get("available"):
        for s in ("subject_00", "subject_01"):
            terc = _g(svt, "terciles", s, "rows", default={})
            if not terc:
                continue
            up = _g(terc, "upright", "D2b_minus_delivered", default={})
            bent = _g(terc, "most_bent", "D2b_minus_delivered", default={})
            ci = up.get("ci95") or [None, None]
            row(f"MECHANISM. on UPRIGHT frames D2b's IoU equals the delivered build's, {s}",
                "the 95 % interval of the paired difference must contain 0",
                _g(terc, "upright", "delivered_iou"), up.get("median_iou_difference"),
                ci, ci[0] is not None and ci[0] <= 0.0 <= ci[1], REF_SILHOUETTE,
                f"tilt {_g(terc, 'upright', 'tilt_range_deg')} deg over "
                f"{_g(terc, 'upright', 'frames')} frames. PRE-REGISTERED before the "
                "instrument ran. This row and the next are the two halves of the "
                "coordinator's hypothesis; together they say whether the fall is the "
                "trunk-axis offset or the mesh's shape.")
            strict = _g(svt, "strict_upright_band_the_hypothesis_names", s, default={})
            sd = strict.get("D2b_minus_delivered") or {}
            sci = sd.get("ci95") or [None, None]
            row(f"MECHANISM. the same, at the threshold the hypothesis NAMES "
                f"(trunk tilt <= 10 deg), {s}",
                "the 95 % interval of the paired difference must contain 0",
                strict.get("delivered_iou"), sd.get("median_iou_difference"), sci,
                sci[0] is not None and sci[0] <= 0.0 <= sci[1], REF_SILHOUETTE,
                f"{strict.get('frames')} frames. On this band the whole rig moved "
                f"{_g(strict, 'shift_D2b_minus_delivered_mm', 'norm_median')} mm while the "
                f"delivered hands moved "
                f"{_g(strict, 'joint_displacements_mm', 'LeftHand', 'D2b_minus_delivered_median_mm')} / "
                f"{_g(strict, 'joint_displacements_mm', 'RightHand', 'D2b_minus_delivered_median_mm')} mm: "
                "the clavicle chain is re-aimed by both D2 and D2b. A fall of this size "
                "against a 10 mm body displacement is not the root placement.")
            row(f"MECHANISM. the fall GROWS with trunk tilt, {s}",
                "most-bent tercile must be worse than the upright tercile",
                up.get("median_iou_difference"), bent.get("median_iou_difference"),
                bent.get("ci95"),
                (bent.get("median_iou_difference") is not None
                 and up.get("median_iou_difference") is not None
                 and bent["median_iou_difference"] < up["median_iou_difference"]),
                REF_SILHOUETTE,
                f"the ORACLE's own tilt dependence over the same terciles is "
                f"{_g(terc, 'most_bent', 'ORACLE_minus_its_own_upright_tercile')} IoU -- "
                "it reads none of our track, so whatever it does with tilt is the masks' "
                "and the rasteriser's and must be subtracted from any reading of ours.")
        row("MECHANISM. overall verdict on the pre-registered hypothesis",
            "TILT-DEPENDENT => the trunk-axis offset and the missing pelvis frame; "
            "FLAT IN TILT => the mesh's shape (D5/D6)", None, None, None, True,
            REF_SILHOUETTE,
            f"{svt.get('verdict')}. {svt.get('verdict_reading', '')} "
            f"{svt.get('what_actually_moved', '')}")
        lvf = _g(svt, "lateral_shift_vs_iou_fall", default={})
        row("MECHANISM. does the fall track the silhouette-VISIBLE part of the shift?",
            "a NEGATIVE rank correlation over the eight camera-subject cells would mean "
            "the silhouette is pricing the displacement it can actually see",
            None, lvf.get("spearman"), None, True, REF_SILHOUETTE,
            f"Pearson {lvf.get('pearson')}, Spearman {lvf.get('spearman')} -- the WRONG "
            "SIGN for that mechanism. Eight cells is a reading, not a test, and it is "
            "quoted as one. A shift along a camera's viewing ray is depth and a silhouette "
            "cannot see it; the per-camera ray angles are in `per_camera_ray`.")

    spw = d2b.get("silhouette_partwise") or {}
    if spw.get("available"):
        for s in ("subject_00", "subject_01"):
            outcome = _g(spw, "pre_registered_outcomes", s, default={})
            if not outcome:
                continue
            same = outcome["1a_torso_unchanged_delivered_to_D2"]
            row(f"PARTWISE. torso+legs-only IoU is UNCHANGED delivered -> D2, {s}",
                "the interval must contain 0 -- those vertices carry no clavicle-chain "
                "weight and D2 moves no root, so this arm can only be identical",
                None, same["difference"], same["ci95"], same["met"], REF_SILHOUETTE,
                "a failure here indicts the SPLIT, not the pipeline.")
            own = outcome["1a_D2b_own_cost_on_torso_and_legs"]
            row(f"PARTWISE. D2b's OWN silhouette cost, torso+legs only, {s}",
                "material == the interval clears -0.003 IoU. Between D2 and D2b these "
                "vertices move by exactly the root shift and nothing else",
                None, own["difference"], own["ci95"], not own["material"],
                REF_SILHOUETTE,
                "THE ISOLATED FIGURE, and the one that says whether the root placement has "
                "a silhouette cost of its own that no joint instrument can see. PASS here "
                "means it does not.")
            arm = outcome["1b_arm_precision_falls"]
            row(f"PARTWISE. arms-only precision, {s}",
                "reported: arm pixels inside the person mask, with the oracle as ceiling",
                arm["delivered"], arm["D2b"], None, True, REF_SILHOUETTE,
                f"delivered {arm['delivered']} -> D2 {arm['D2']} -> D2b {arm['D2b']}; "
                f"ORACLE ceiling {arm['ORACLE_ceiling']}. Pre-registered to fall, and it "
                f"{'did' if arm['met'] else 'did NOT fall monotonically'}.")
            hid = outcome["1c_hidden_fraction_falls_delivered_to_D2"]
            row(f"PARTWISE. arm pixels hidden inside the body's own torso raster, {s}",
                "reported: the fraction a person mask cannot charge for",
                hid["delivered"], hid["D2b"], None, True, REF_SILHOUETTE,
                f"delivered {hid['delivered']} -> D2 {hid['D2']} -> D2b {hid['D2b']}; "
                f"ORACLE {hid['ORACLE']}. Pre-registered to fall delivered -> D2, and it "
                f"{'did' if hid['met'] else 'did NOT'}.")
            fold = outcome["2_folded_arms_score_higher"]
            row(f"I6 CONTROL. a body with both arms FOLDED ACROSS THE CHEST must not "
                f"score higher than the delivery, {s}",
                "whole-person IoU and precision must NOT rise",
                None, fold["whole_iou"]["median_difference"],
                fold["whole_iou"]["ci95"], not fold["met"], REF_SILHOUETTE,
                f"IoU {fold['whole_iou']['median_difference']} "
                f"{fold['whole_iou']['ci95']}, precision "
                f"{fold['whole_precision']['median_difference']} "
                f"{fold['whole_precision']['ci95']}, recall "
                f"{fold['whole_recall']['median_difference']} "
                f"{fold['whole_recall']['ci95']}. NO GATE A CONSTANT CAN PASS, asked of I6 "
                "itself: a person mask is ONE blob, so a limb inside the outline is free. "
                "If a limb-collapsed body wins, the mask cannot be read as a "
                "limb-placement gate and every arm figure scored through it inherits that.")
        corr = _g(spw, "part_split", "correspondence_check", default={})
        worst = max((v["displacement_separation"]["torso_vertex_displacement_median_mm"]
                     for v in corr.values()), default=None)
        row("PARTWISE. the split labels the vertices it thinks it does",
            "under a CLAVICLE-ONLY change (delivered -> D2, where the root moves 0.00 mm) "
            "the torso+legs set's median vertex displacement must be ~0 while the arm "
            "set's is large",
            None, worst, None, worst is not None and worst < 0.01,
            "the delivered mesh against the body asset it was built from",
            "the asset's weights label ASSET vertices; the raster uses BLENDER-exported "
            "ones, and equal counts (13380 / 26756) are only suggestive. This is the "
            "check, and it needs no anatomy: a scrambled vertex order would make the two "
            "labelled sets random halves of one mesh with identical displacement "
            "distributions. Measured: arm "
            + " / ".join(str(v["displacement_separation"]["arm_vertex_displacement_median_mm"])
                         for v in corr.values())
            + " mm against torso "
            + " / ".join(str(v["displacement_separation"]["torso_vertex_displacement_median_mm"])
                         for v in corr.values())
            + " mm. A SECOND, weaker check -- agreement with a nearest-FK-joint labelling -- "
              "reads "
            + " / ".join(str(v["nearest_fk_joint"]["agreement_with_the_nearest_fk_joint"])
                         for v in corr.values())
            + " against a shuffled-label chance level of "
            + " / ".join(str(v["nearest_fk_joint"]["chance_level_same_labels_shuffled"])
                         for v in corr.values())
            + ". It does not reach 1.0 because nearest-JOINT and nearest-WEIGHT disagree "
              "at the shoulder cap and the armpit. AN EARLIER VERSION OF THIS ROW DEMANDED "
              ">= 0.90 AGAINST THAT SECOND CHECK; that band was invented without asking "
              "what the criterion does at the shoulder, and it was replaced by the "
              "control-derived one above. The substitution is recorded rather than hidden.")

        for s in ("subject_00", "subject_01"):
            e = _g(spw, "subjects", s, default={})
            m = e.get("margins") or {}
            if not m:
                continue
            r = m["whole_iou__ROOT_ONLY_minus_D2"]
            c = m["whole_iou__CLAVICLE_ONLY_minus_D2"]
            row(f"PARTWISE (EXACT). the ROOT's own whole-person silhouette cost, {s}",
                "the interval must contain 0 for the root to be free of cost",
                None, r["median_difference"], r["ci95"],
                r["ci95"] is not None and r["ci95"][0] <= 0.0 <= r["ci95"][1],
                REF_SILHOUETTE,
                "AN EXACT ISOLATION, not a part-wise proxy: `root_translation` enters "
                "forward kinematics only at the Root joint, so changing it translates every "
                "skinned vertex by exactly that vector. This arm is D2's OWN rendered mesh "
                "moved by the per-frame root delta -- same pose, same mesh, same pixels "
                "except the translation. FAIL here means the root placement has a "
                "silhouette cost of its own that no joint instrument in this lane can see.")
            row(f"PARTWISE (EXACT). the clavicle re-aim D2b induces, at D2's root, {s}",
                "reported beside the root's, on the same axis; the two are near-additive",
                None, c["median_difference"], c["ci95"], True, REF_SILHOUETTE,
                f"root {r['median_difference']} + clavicle {c['median_difference']} against "
                f"a measured D2 -> D2b whole-person change of "
                f"{round(_g(e, 'whole_iou', 'D2b', default=0.0) - _g(e, 'whole_iou', 'D2', default=0.0), 5)}. "
                "The two terms are separable because one of them is a rigid translation.")
            u = e.get("upright_band_tilt_le_10deg") or {}
            ru = u.get("torso_iou__ROOT_ONLY_D2_pose_at_D2b_root_minus_D2") or {}
            row(f"PARTWISE (EXACT). the root's own cost on UPRIGHT frames only "
                f"(tilt <= 10 deg), {s}",
                "the interval must contain 0", None, ru.get("median_difference"),
                ru.get("ci95"),
                ru.get("ci95") is not None and ru["ci95"][0] <= 0.0 <= ru["ci95"][1],
                REF_SILHOUETTE,
                f"{u.get('frames')} frames, where the root moves 10-13 mm. Section 13 found "
                "the whole-person fall at FULL SIZE on these frames; if the root were "
                "carrying it, this row would be large. It is the cross-check between the "
                "two passes.")

    reg = report.get("d2_regression_check", {})
    if reg.get("ok") is not None:
        row("META. extending this gate did not move any D2 figure",
            "<= 0.005 mm on every D2 band", None, reg.get("max_abs_difference"), None,
            bool(reg["ok"]), "gate-d2-prior.json, the file this run replaces", reg["why"])
    return rows


D2B_BLIND_TO = (
    "THE HOIST THAT REMAINS. D2b does not put the character on the ground correctly; it "
    "moves the placement error OUT of the converter and INTO the ground projection, where "
    "it is now visible as one uncapped vertical scalar per take (~83 mm and ~49 mm). That "
    "residue is the legs' surplus length -- the canonical thigh is 430 mm against roughly "
    "400 measured on these performers -- and it is D5's, not this step's. "
    "TWIST, unchanged from D2: `_world_for_bone` is the minimal rotation from the parent's "
    "frame, so the arm inherits the clavicle's roll and no joint-origin score in this lane "
    "can see it. "
    "THE TEMPORAL DEFECT. D2 left the clavicle chain jittering above a human's peak joint "
    "rate. D2b reports the counts and carries NO band for them; D2c owns that question. "
    "SELF-REFERENCE. The round trip scores the converter against its own output, so it "
    "cannot see detector error, triangulation error, or any convention shared between "
    "input and reference -- and its own hip convention WAS exactly such a defect, which is "
    "why it took a change of reference frame to expose it. Now that the convention agrees, "
    "the round trip's 0.5 mm is a consistency figure and NOT an accuracy figure. "
    "THE SCOREBOARD is agreement with an instrument, not accuracy; neither side has ground "
    "truth, and a change that improves agreement could be moving toward MAMMA's error. "
    "THE SILHOUETTE is blind to depth, to left/right mirroring of a fore-aft symmetric "
    "pose, and to everything inside the outline."
)


# ------------------------------------------------------------------ verdicts
def verdicts(report) -> list:
    rows = []

    def row(name, band, before, after, interval, ok, reference, note=""):
        rows.append({"band": name, "band_value": band, "before": before, "after": after,
                     "interval": interval, "verdict": "PASS" if ok else "FAIL",
                     "reference": reference, "note": note})

    for s in ("subject_00", "subject_01"):
        d = report["subjects"][s]
        rt = d["roundtrip"]
        after_arms = rt["after"]["per_group"]["arms"]
        before_arms = rt["before_via_legacy_anchor_swap"]["per_group"]["arms"]
        worst = max(rt["after"]["per_landmark_arms"].values())
        row(f"A. round-trip arms, all six landmarks <= {BAND_ARMS_MM} mm, {s}",
            f"<= {BAND_ARMS_MM} mm median", before_arms, after_arms,
            d["margins"]["roundtrip_arms_before_minus_after"]["ci95_mm"],
            worst <= BAND_ARMS_MM, REF_ROUNDTRIP,
            "PRE-REGISTERED AND FAILED. The premise behind the band -- that the clavicle "
            "does not share the legs' root-placement problem -- is false; see the "
            "attribution control.")
        row(f"A. round-trip legs and torso stay 0.00, {s}", "== 0.00 mm exact",
            rt["before_via_legacy_anchor_swap"]["per_group"]["legs"],
            rt["after"]["per_group"]["legs"], None,
            rt["after"]["per_group"]["legs"] == 0.0 and rt["after"]["per_group"]["torso"] == 0.0,
            REF_ROUNDTRIP, "the legs are untouched BY CONSTRUCTION, and under every variant.")
        c = d["CONTROL_hip_drop_removed_pass2"]
        row(f"ATTRIBUTION. round-trip arms with the rig's hip drop out of pass 2 <= "
            f"{BAND_ATTRIBUTION_MM} mm, {s}",
            f"<= {BAND_ATTRIBUTION_MM} mm median", after_arms, c["per_group"]["arms"], None,
            c["per_group"]["arms"] <= BAND_ATTRIBUTION_MM, REF_ROUNDTRIP,
            "passing means 100 % of the A failure is root placement and 0 % is the clavicle.")
        f = d["CONTROL_legacy_anchor_reproduces_the_committed_report"]
        row(f"B(a). the legacy-anchor swap reproduces the committed report, {s}",
            "<= 0.01 mm", f["committed_arms_median_mm"], f["swapped_arms_median_mm"], None,
            f["abs_difference_mm"] <= 0.01, REF_ROUNDTRIP, f["why"])
        sw = d["CONTROL_legacy_scalar_sweep"]
        row(f"B(b). NO on-torso-axis scalar reaches the band, {s}",
            f"best of 17 scalars must be > {BAND_ARMS_MM} mm", None,
            sw["best_arms_median_mm"], None,
            sw["best_arms_median_mm"] > BAND_ARMS_MM, REF_ROUNDTRIP,
            f"best scalar {sw['best_scalar']}. " + sw["why"])
        uc = d["CONTROL_upperchest_origin"]
        row(f"B(c). the UpperChest origin fails, {s}", f"> {BAND_ARMS_MM} mm", None,
            uc["per_group"]["arms"], None, uc["per_group"]["arms"] > BAND_ARMS_MM,
            REF_ROUNDTRIP, uc["why"])
        legs_ok = all(v["legs_median_mm"] == 0.0 for v in sw["per_scalar"].values()) and \
            uc["per_group"]["legs"] == 0.0 and c["per_group"]["legs"] == 0.0
        row(f"B(d). legs read 0.00 under EVERY variant, {s}", "== 0.00 mm", None, 0.0, None,
            legs_ok, REF_ROUNDTRIP,
            "the legs are untouched by construction, not by luck: their directions are "
            "landmark-to-landmark and no origin helper enters them.")
        cap = d["on_our_capture_canonical"]
        row(f"DELIVERY. arms on our own capture, canonical rig, {s}",
            "reported, no band -- this is the delivered configuration",
            cap["before"]["arms"], cap["after"]["arms"],
            d["margins"]["on_capture_arms_before_minus_after"]["ci95_mm"],
            True, REF_OURS, "one solve, real landmarks: the arm the round-trip "
                            "contamination does not reach.")
        t = d["temporal"]
        extra = t["clavicle_chain_extra_frames_over_the_ceiling"]
        row(f"TEMPORAL. clavicle-chain frames above a human's peak joint rate, {s}",
            f"must not GROW; ceiling {t['ceiling_deg_per_frame']} deg/frame (~800 deg/s)",
            sum(t["joints"][n]["before"]["frames_over_the_physical_ceiling"] for n in
                ("LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
                 "LeftLowerArm", "RightLowerArm")),
            sum(t["joints"][n]["after"]["frames_over_the_physical_ceiling"] for n in
                ("LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
                 "LeftLowerArm", "RightLowerArm")),
            None, extra <= 0, "the rig's own joint angles against human physiology",
            "THE POSITIONAL SCORE CANNOT SEE THIS. D2 measures the direction from an "
            "origin 60-170 mm from the landmark where the old anchor sat ~400 mm away, and "
            "a short lever arm amplifies landmark noise into direction noise. The arm root "
            "lands far better and travels worse; both are true and neither is evidence "
            "about the other.")
        row(f"TEMPORAL. legs, neck and head step-angles bit-identical, {s}",
            "identical", None, None, None, t["legs_head_and_neck_are_bit_identical"],
            "the rig's own joint angles",
            "the blast radius, stated temporally as well as positionally.")
        b = d["arm_B_mamma_joints_in"]
        row(f"C(arm B). MAMMA's own joints in, canonical arms, {s}",
            "reported on its OWN reference; never differenced with the rows above",
            b["canonical_before"]["arms"], b["canonical_after"]["arms"],
            b["margins"]["canonical_arms_before_minus_after"]["ci95_mm"], True, REF_ARMB,
            "MAMMA REPORTS AND NEVER SELECTS: nothing here chooses anything.")

    bi = report.get("bit_identity")
    if bi:
        row("C. bit-identity theorems", "every listed array identical", None, None, None,
            bi["held"], "the delivered build vs the branch rebuild", bi["means"])
    scb = report.get("scoreboard")
    if scb:
        for s, e in scb["subjects"].items():
            row(f"E. scoreboard `capture` arm identical, {s}", "== 0 mm", None,
                e["capture_must_be_identical"]["max_abs_difference_mm"], None,
                e["capture_must_be_identical"]["max_abs_difference_mm"] == 0.0,
                REF_MAMMA, e["capture_must_be_identical"]["why"])
            for arm in ("canon", "sized"):
                k = f"{arm}_all_joints"
                row(f"E. scoreboard `{arm}`, all joints, {s}",
                    "reported, no band -- MAMMA is a reference, not a target",
                    e[k]["before_mm"], e[k]["after_mm"],
                    e[k]["margin_before_minus_after"]["ci95_mm"], True, REF_MAMMA,
                    "positive margin means D2 improved agreement.")
                k6 = f"{arm}_arms_six"
                row(f"E. scoreboard `{arm}`, the six arm joints, {s}",
                    "reported, no band", e[k6]["before_mm"], e[k6]["after_mm"],
                    e[k6]["margin_before_minus_after"]["ci95_mm"], True, REF_MAMMA, "")
    return rows


BLIND_TO = (
    "TWIST. `_world_for_bone` is the MINIMAL rotation from the parent's frame, so the "
    "upper arm, forearm and hand inherit the clavicle's roll. D2 turns the clavicle by "
    "15-19 degrees and the arm's twist -- and the delivered hand orientation -- moves with "
    "it by some fraction of that. No instrument in this lane scores it: a round trip is "
    "positional, the silhouette is a mask, and a hand is a few pixels across in a body "
    "camera. Only a close-up picture can, and one was rendered. "
    "PROPORTIONS. The canonical rig's shoulder is far wider than these performers', so "
    "aiming the clavicle correctly on it puts the arm root further out; that is D5 and "
    "this gate cannot separate it. "
    "ABSOLUTE PLACEMENT. Every own-capture figure is root-relative, so the deliberate "
    "~90 mm ground projection AND the 80 mm root-placement offset that dominates the "
    "round trip are outside the score by construction -- the round trip only sees the "
    "latter because one direction is now measured in the rig's frame. "
    "SELF-REFERENCE. The round trip scores the converter against its own output, so it "
    "cannot see detector error, triangulation error, or any convention shared between "
    "input and reference. Its 0.00 mm on the legs was never evidence the root placement "
    "is right; it is evidence that a translation-invariant chain scored root-relatively "
    "cannot see a translation. "
    "THE SCOREBOARD is agreement with an instrument, not accuracy, and neither side has "
    "ground truth."
)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ours = np.stack([
        np.load(TRACKS / f"subject-{s:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
        for s in (0, 1)])
    mapping = subject_map.mamma_index_for(ours)
    rng = np.random.default_rng(rc.RNG_SEED)

    report: dict = {
        "instrument": "tools/compare/d2_clavicle_gate.py",
        "step": "D2 -- the clavicle origin",
        "regenerate": "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d2_clavicle_gate.py",
        "autoanim_gnm_resolved_to": str(Path(autoanim_gnm.__file__).resolve()),
        "subject_correspondence": {f"our_{k}": f"body_id-{v:02d}" for k, v in mapping.items()},
        "subject_correspondence_source": "tools/head/subject_map.py, 3D pelvis agreement",
        "bands": {"roundtrip_arms_median_mm": BAND_ARMS_MM,
                  "roundtrip_legs_and_torso_median_mm": BAND_LEGS_MM,
                  "attribution_control_arms_median_mm": BAND_ATTRIBUTION_MM},
        "blind_to": BLIND_TO,
        "subjects": {},
    }
    # EVERY D2 BLOCK RUNS THROUGH THE ZERO-OFFSET HELPER. D2's figures were measured
    # against a converter with no root fix; D2b ships one. `zero_offset` is that converter,
    # reached by swapping one module attribute, so the bands below keep meaning what they
    # meant -- and `d2_regression_check` proves it against the file this run replaces.
    with root_offset(zero_offset):
        for s in (0, 1):
            report["subjects"][f"subject_{s:02d}"] = subject_block(s, mapping[s], rng, report)

    if REBUILD.exists() and (REBUILD / "subject-00.body-track.npz").exists():
        report["bit_identity"] = bit_identity()
        before = scoreboard_errors(TRACKS, mapping)
        after = scoreboard_errors(REBUILD, mapping)
        report["scoreboard"] = scoreboard_block(before, after, rng)
    else:
        report["bit_identity"] = None
        report["scoreboard"] = None
        report["rebuild_missing"] = str(REBUILD)

    report["root_placement_variant"] = {
        "shipping": "NOTHING WHEN THIS WAS WRITTEN (D2). It SHIPPED as D2b on 2026-09-03, "
                    "and this block is retained unchanged, measured through the "
                    "zero-offset helper, as the record of what was predicted and found "
                    "before the derivation was in the source. `d2b_root_placement` below "
                    "measures the shipped one and checks it reproduces this to 0.00 mm.",
        "derivation": "root_translation = pelvis - rest[Hips] - R_hips . mid(rest[LeftUpperLeg], "
                      "rest[RightUpperLeg]), from FK's UpperLegMid = root + rest[Hips] + "
                      "R_hips . mid set equal to the captured hip midpoint. No constant: "
                      "every term is the skeleton's own rest geometry.",
        "pre_registered_prediction": ROOT_VARIANT_PREDICTION,
        "projection_note": "project_generated_foot_contacts hardcodes DETAILED_HUMANOID, so "
                           "it measures a SIZED solve's feet on CANONICAL rest lengths. "
                           "Pre-existing, unchanged here, and it limits what the sized "
                           "ground figures can be asked to mean.",
        "subjects": {f"subject_{s:02d}": _zero_offset_root_variant(s, mapping[s])
                     for s in (0, 1)},
    }
    report["gate"] = verdicts(report)
    report["verdict"] = "PASS" if all(r["verdict"] == "PASS" for r in report["gate"]) else "FAIL"
    report["d2_regression_check"] = d2_regression_check(report)

    # ------------------------------------------------------------------ D2b, as shipped
    d2b: dict = {
        "step": "D2b -- the root placed on the captured hips",
        "shipping": "src/autoanim_gnm/commercial_multiview.py: `_leg_root_offset`, and the "
                    "`root_translation[frame]` line that subtracts it.",
        "derivation": "root_translation = pelvis - rest[Hips] - R_hips . mid(rest[LeftUpperLeg], "
                      "rest[RightUpperLeg]), from FK's UpperLegMid = root + rest[Hips] + "
                      "R_hips . mid set equal to the captured hip midpoint. No constant: "
                      "every term is the skeleton's own rest geometry.",
        "pre_registered": D2B_PREREGISTERED,
        "blind_to": D2B_BLIND_TO,
        "bands": {"roundtrip_arms_median_mm": BAND_D2B_ARMS_MM,
                  "theorem_fk_hip_m": BAND_FK_HIP_M,
                  "theorem_offset_m": BAND_OFFSET_M,
                  "theorem_root_float32_m": BAND_ROOT_F32_M},
        "subjects": {f"subject_{s:02d}": d2b_subject_block(s, mapping[s]) for s in (0, 1)},
    }
    d2b["absolute_hip_placement"] = {"reference": REF_ABSOLUTE, "subjects": {}}
    arms_for_rung11: dict = {}
    for label, path in (("delivered_before_D2", TRACKS), ("D2", REBUILD),
                        ("D2b", REBUILD_ROOT)):
        if not (path / "subject-00.body-track.npz").exists():
            continue
        arms_for_rung11[label] = scoreboard_errors(path, mapping)
        for s in (0, 1):
            d2b["absolute_hip_placement"]["subjects"].setdefault(
                f"subject_{s:02d}", {})[label] = hip_absolute_offset(path, s)
    if len(arms_for_rung11) > 1:
        d2b["rung11"] = rung11_block(arms_for_rung11, rng)
    if (REBUILD_ROOT / "subject-00.body-track.npz").exists() and \
            (REBUILD / "subject-00.body-track.npz").exists():
        d2b["bit_identity"] = d2b_bit_identity()
    d2b["silhouette"] = silhouette_block()
    d2b["silhouette_vs_tilt"] = silhouette_vs_tilt_block()
    d2b["silhouette_partwise"] = silhouette_partwise_block()
    report["d2b_root_placement"] = d2b
    report["d2b_gate"] = d2b_verdicts(report)
    report["d2b_verdict"] = ("PASS" if all(r["verdict"] == "PASS" for r in report["d2b_gate"])
                             else "FAIL")
    (OUT_DIR / "gate.json").write_text(json.dumps(report, indent=2, default=float))
    console(report)
    print(f"\nwrote {OUT_DIR / 'gate.json'}")
    return 0


def console(report) -> None:
    print(f"=== D2, the clavicle origin.  autoanim_gnm: {report['autoanim_gnm_resolved_to']}")
    print(f"subject map: " + ", ".join(f"{k} -> {v}" for k, v in
                                       report["subject_correspondence"].items()))
    print(f"\n{'band':<72} {'before':>9} {'after':>9}  verdict")
    for r in report["gate"]:
        b = "-" if r["before"] is None else f"{r['before']:9.2f}"
        a = "-" if r["after"] is None else f"{r['after']:9.2f}"
        print(f"{r['band'][:72]:<72} {b:>9} {a:>9}  {r['verdict']}")
    print(f"\nOVERALL (D2 as a step): {report['verdict']}")
    if report.get("d2b_gate"):
        print(f"\n=== D2b, the root placed on the captured hips")
        print(f"{'band':<72} {'before':>9} {'after':>9}  verdict")
        for r in report["d2b_gate"]:
            b = "-" if not isinstance(r["before"], (int, float)) else f"{r['before']:9.2f}"
            a = "-" if not isinstance(r["after"], (int, float)) else f"{r['after']:9.4g}"
            print(f"{r['band'][:72]:<72} {b:>9} {a:>9}  {r['verdict']}")
        print(f"\nD2b VERDICT: {report['d2b_verdict']}")


if __name__ == "__main__":
    raise SystemExit(main())
