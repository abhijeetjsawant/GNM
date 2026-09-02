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
        "shipping": "NOTHING. Every swap here is instrument-side; src/ is untouched by it.",
        "derivation": "root_translation = pelvis - rest[Hips] - R_hips . mid(rest[LeftUpperLeg], "
                      "rest[RightUpperLeg]), from FK's UpperLegMid = root + rest[Hips] + "
                      "R_hips . mid set equal to the captured hip midpoint. No constant: "
                      "every term is the skeleton's own rest geometry.",
        "pre_registered_prediction": ROOT_VARIANT_PREDICTION,
        "projection_note": "project_generated_foot_contacts hardcodes DETAILED_HUMANOID, so "
                           "it measures a SIZED solve's feet on CANONICAL rest lengths. "
                           "Pre-existing, unchanged here, and it limits what the sized "
                           "ground figures can be asked to mean.",
        "subjects": {f"subject_{s:02d}": root_variant_block(s, mapping[s]) for s in (0, 1)},
    }
    report["gate"] = verdicts(report)
    report["verdict"] = "PASS" if all(r["verdict"] == "PASS" for r in report["gate"]) else "FAIL"
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
    print(f"\nOVERALL: {report['verdict']}")


if __name__ == "__main__":
    raise SystemExit(main())
