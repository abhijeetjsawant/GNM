#!/usr/bin/env python3
"""D7c's gate: the pelvis on the rig's own rest, measured on exact truth and on the take.

WHAT IT MEASURES, and against what.

THE ORACLE (O1, O2, O3). The D3 gate's six exact-skeleton bodies: a perturbed per-performer
rest, forward-kinematicked with the delivered subject-0 motion, turned back into the 19
landmarks by `retarget_cost.landmarks_from_fk`, and fed to the REAL converter. On such a body
the truth is known exactly, and the rig's own rest triangle
``{Spine, LeftUpperLeg, RightUpperLeg}`` about the leg-root midpoint is CONGRUENT to the
observed one -- so a pelvis fit that reads the rig's rest recovers the truth's rotation
exactly and one that reads SOMA-77's rest cannot. The shipped fit reads **6.865 deg** of pitch
about the hip line on every frame of every seed; that is the defect, and it is a constant of
the convention, not noise.

  * O1 pelvis vs truth, the `Spine` and `Hips` origins hoist-subtracted, the torso group on
    the ABSOLUTE row, and -- the clause that discriminates the wrong-origin control -- the
    UNNORMALISED three-point positional residual in METRES. The wrong-origin template reads
    0.000 deg of tilt, because the rig's rest pelvis is symmetric enough that an 80 mm origin
    error is absorbed entirely into a translation the rotation-only fit never sees; only the
    metre residual exposes it. A residual between NORMALISED frames would lose it.
  * O2 the legs: the exact hip line removes the shipped fit's 0.033 deg of yaw, so the leg
    roots move by ~0.06 mm and the legs, feet and toes stay within 0.1 mm of the baseline's
    forward kinematics, with identical contacts and the hoist within 0.05 mm. Bit-identity is
    NOT claimed and must not be: a pelvis frame is whole-take.
  * O3 the D3 gate's own ALIGNED gauge (`retarget_cost.score`), REPORTED. That gauge subtracts
    the leg-root midpoint per frame and is structurally blind to a root move (CLAUDE.md, D9b);
    every banded oracle number here is on the ABSOLUTE row instead.

THE TAKE. Read from a build's own delivered bytes plus the converter inputs the delivery
script's watcher dumped (`spine_world_z_up_m` is NOT in the delivered `.npz`, and every pelvis
figure is measured against SOMA-77's `Spine1`). The delivered pelvis +Y against
``Spine1 - hip midpoint``, the pelvis lever guard's demoted frames with the pre-guard median
FROZEN from the unchanged input, the pitch change and the root move hoist-subtracted.

WHAT THIS INSTRUMENT IS BLIND TO, stated before any number is read:

  * **It cannot resolve the convention.** Everything above is measured in the RIG's own frame
    against the RIG's own rest. That SOMA-77's `Spine1` lies on the rig's ``Hips -> Spine``
    axis seen from the hip midpoint is an assumption this step does not test and cannot: no
    landmark instrument resolves a constant change of frame, the exact oracle is fed the rig's
    own `Spine` and therefore agrees by construction, and the photographs judge delivered
    consequences only. The anatomical question goes to lane H's marker session.
  * **A length invariant cannot score direction.** The guard's ceiling is a LENGTH rule; a
    same-length rotation of the hip line passes it, and mode (b) gives that rotation full
    authority. Selector S decides that trade on synthetic truth; the take cannot.
  * **The oracle's truth motion is D9b's delivered pelvis motion**, produced by the very fit
    being replaced. O1's exactness is unaffected -- a congruence is a congruence whatever the
    motion -- but nothing here says how much real pelvic motion a performer has.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_pelvis_rest_gate.py \\
        --oracle --oracle-save artifacts/compare/d7c-pelvis-rest/oracle-shipped \\
        --take artifacts/compare/d7c-pelvis-rest/delivery-hygiene --take-label shipped \\
        --out artifacts/compare/d7c-pelvis-rest/instrument.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "tools/swap-harness", "scripts"):
    if str(ROOT / _relative) not in sys.path:
        sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

import autoanim_gnm.commercial_multiview as cm  # noqa: E402

if str(ROOT / "tools/compare") not in sys.path:                      # noqa: E402
    sys.path.insert(0, str(ROOT / "tools/compare"))
from d7c_source_fingerprint import fingerprint_now as source_fingerprint  # noqa: E402
from autoanim_gnm.body import (  # noqa: E402
    DETAILED_HUMANOID, forward_kinematics_positions, skeleton_for_track_dict,
    _quaternion_multiply)
import d3_skeleton_gate as d3  # noqa: E402
import d9b_hoist_gate as d9b  # noqa: E402
import d7c_pelvis_estimators as est  # noqa: E402
import retarget_cost as rc  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d7c-pelvis-rest"
SHIPPED = ROOT / "artifacts/commercial-multiview-soma77"
HOIST_REPORT_CUT_MM = d9b.HOIST_REPORT_CUT_MM

# The card's O1/O2 bands, fixed here before a number is read.
O1_TILT_DEG = 0.01
O1_ORIGIN_MM = 0.01
O1_RESIDUAL_M = 1.0e-6
O2_LEG_MM = 0.1
O2_HOIST_MM = 0.05

LEG_JOINTS = ("LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
              "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes")
# A pelvis frame is whole-take: the arms and trunk ride it by design. These are the joints
# O2 claims are unmoved, and they are exactly the ones the exact hip line does not rotate.
REPORTED_JOINTS = ("Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
                   "LeftShoulder", "LeftUpperArm", "LeftHand",
                   "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes")


def summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), "median": round(float(np.median(values)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "max": round(float(values.max()), 4)}


def world_rotations(rotations: np.ndarray, skeleton) -> np.ndarray:
    out = np.zeros_like(rotations)
    for index, joint in enumerate(skeleton.joints):
        if joint.parent == -1:
            out[:, index] = rotations[:, index]
        else:
            out[:, index] = _quaternion_multiply(out[:, joint.parent], rotations[:, index])
    return out


def relative_tilt(delivered_q: np.ndarray, truth_q: np.ndarray) -> dict:
    """`delivered * truth^-1` as an angle and as components in the TRUTH's own pelvis axes.

    The truth frame's columns are the hip line (X), up (Y) and forward (Z), so the three
    components are pitch about the hip line, yaw about up and roll about forward.
    """
    relative = Rotation.from_quat(delivered_q) * Rotation.from_quat(truth_q).inv()
    vector = relative.as_rotvec()
    frames = Rotation.from_quat(truth_q).as_matrix()
    angle = np.degrees(np.linalg.norm(vector, axis=1))
    pitch = np.degrees(np.einsum("ni,ni->n", vector, frames[:, :, 0]))
    yaw = np.degrees(np.einsum("ni,ni->n", vector, frames[:, :, 1]))
    roll = np.degrees(np.einsum("ni,ni->n", vector, frames[:, :, 2]))
    return {"angle": summary(angle), "pitch_about_hip_line": summary(np.abs(pitch)),
            "pitch_signed_median": round(float(np.median(pitch)), 4),
            "yaw": summary(np.abs(yaw)), "roll": summary(np.abs(roll)),
            "_angle": angle}


# --------------------------------------------------------------------------- the arms
def arm_geometry(name: str, rest_array: np.ndarray, skeleton) -> dict:
    """Each arm's own template and its own fit ORIGIN, for the unnormalised residual.

    The residual O1 bands is the residual of the fit the arm ACTUALLY TOOK -- its template
    carried onto its observed set about its own origin -- not a re-derivation under one
    common template. That is what makes it read 85-107 mm on the shipped SOMA template
    (whose 39 mm spine lever does not exist on this rig) and 84 mm on the wrong-origin
    control (whose lever is right and whose ORIGIN is 80 mm out).
    """
    index = skeleton.index
    rest = {n: rest_array[index(n)] for n in ("Hips", "Spine", "LeftUpperLeg",
                                              "RightUpperLeg")}
    mid = 0.5 * (rest["LeftUpperLeg"] + rest["RightUpperLeg"])
    # `src_default` is whatever `src/` currently ships, so its geometry must be RESOLVED
    # from `PELVIS_FRAME_SOURCE` and never assumed. Before D7c's src change that was mode C
    # and the SOMA template; after it, it is a rig mode and the rig's own triangle. Reading
    # the residual under the wrong template would score the shipping arm against geometry
    # it does not use -- 0.10 m instead of 1e-7 -- and O1's residual clause would fail for
    # a reporting reason.
    effective = cm.PELVIS_FRAME_SOURCE if name == "src_default" else name
    if effective in ("C_kabsch_pelvis", "C_soma_template"):
        return {"template": np.asarray(cm.SOMA77_REST_PELVIS_TEMPLATE_M, dtype=np.float64),
                "origin": "root_landmark",
                "note": ("SOMA-77's rest pelvis about the `root` landmark -- the fit D7c "
                         "replaces")}
    if effective == "wrong_origin":
        return {"template": np.stack((rest["Spine"], rest["LeftUpperLeg"],
                                      rest["RightUpperLeg"])),
                "origin": "root_landmark",
                "note": "the rig's rest offsets taken from `Hips` -- 80 mm above the "
                        "observed origin. Pre-registered WRONG; reads 0.000 deg of tilt."}
    return {"template": est.rest_template_about_hip_midpoint(rest),
            "origin": "hip_midpoint",
            "note": "the rig's own rest triangle about the leg-root midpoint"}


def install_arm(name: str, rest_array: np.ndarray, skeleton):
    """Substitute `_pelvis_world_frames` (or the template it reads) for one arm.

    Returns the restore callable. `src_default` installs NOTHING: it is whatever `src/`
    currently ships, which is the point of running this instrument before and after.
    """
    saved_pelvis = cm._pelvis_world_frames
    saved_template = cm.SOMA77_REST_PELVIS_TEMPLATE_M
    index = skeleton.index
    rest = {n: rest_array[index(n)] for n in skeleton.names}

    def restore():
        cm._pelvis_world_frames = saved_pelvis
        cm.SOMA77_REST_PELVIS_TEMPLATE_M = saved_template

    if name == "src_default":
        return restore
    if name == "C_soma_template":
        cm._pelvis_world_frames = lambda points, spine, **kw: saved_pelvis(
            points, spine, **{**kw, "mode": "C_kabsch_pelvis"})
        return restore
    if name == "wrong_origin":
        geometry = arm_geometry(name, rest_array, skeleton)
        cm.SOMA77_REST_PELVIS_TEMPLATE_M = tuple(map(tuple, geometry["template"]))
        cm._pelvis_world_frames = lambda points, spine, **kw: saved_pelvis(
            points, spine, **{**kw, "mode": "C_kabsch_pelvis"})
        return restore
    if name in est.RIG_MODES:
        # Since the src change these two are BRANCHES OF THE SHIPPED FUNCTION, so the arm
        # runs the delivery's own code path rather than the instrument's copy of it. The
        # two are pinned bit-for-bit in `tests/test_pelvis_rest.py`; if `src/` ever loses a
        # mode, this falls back to the instrument and says so in the report rather than
        # silently scoring a different estimator.
        if name in getattr(cm, "RIG_REST_PELVIS_MODES", ()):
            cm._pelvis_world_frames = (
                lambda points, spine, **kw: saved_pelvis(
                    points, spine, **{**kw, "rest": rest, "mode": name}))
        else:
            cm._pelvis_world_frames = (
                lambda points, spine, **kw: est.rig_rest_pelvis_frames(
                    points, spine, rest, mode=name, guard=True))
        return restore
    if name == "frozen_upright":
        cm._pelvis_world_frames = lambda points, spine, **kw: (
            est.world_vertical_frames(points),
            {"status": "solved", "mode": "frozen_upright_control",
             "resolved_fraction": 1.0, "smoothing_frames": 0, "interpolated_frames": 0})
        return restore
    restore()
    raise SystemExit(f"unknown arm {name!r}")


def run_arm(name: str, seed: int, rest_array: np.ndarray, skeleton, truth, truth_world,
            landmarks, spine_truth, toes, save: Path, export_glb: bool = False) -> dict:
    """One arm, one seed, through the REAL converter under a recording watcher."""

    captured: list = []
    saved_projection = cm.project_generated_foot_contacts

    def projection(track, **kwargs):
        projected, diagnostics = saved_projection(track, **kwargs)
        captured.append({"pre": track, "post": projected,
                         "diagnostics": diagnostics.as_dict()})
        return projected, diagnostics

    cm.project_generated_foot_contacts = projection
    restore = install_arm(name, rest_array, skeleton)
    pelvis_report: dict = {}
    try:
        track = cm.positions_to_body_track(
            rc.Z_UP_FROM_Y_UP(landmarks), sample_rate_hz=30, provenance_sha256="0" * 64,
            toe_world_z_up_m=rc.Z_UP_FROM_Y_UP(toes),
            spine_world_z_up_m=rc.Z_UP_FROM_Y_UP(spine_truth), skeleton=skeleton,
            pelvis_report_out=pelvis_report)
    finally:
        restore()
        cm.project_generated_foot_contacts = saved_projection

    assert len(captured) == 1, f"the projection ran {len(captured)} times, not once"
    record = captured[0]
    pre, post = record["pre"], record["post"]
    index = skeleton.index
    roots = np.asarray(track.root_translation_m, np.float64)
    rotations = np.asarray(track.local_rotations_xyzw, np.float64)
    hoist = roots - np.asarray(pre.root_translation_m, np.float64)
    magnitude = 1e3 * np.linalg.norm(hoist, axis=1)
    hoisted = magnitude > HOIST_REPORT_CUT_MM
    fk = forward_kinematics_positions(roots, rotations, skeleton=skeleton).astype(np.float64)
    fk_unhoisted = fk - hoist[:, None, :]
    delivered_world = world_rotations(rotations, skeleton)

    tilt = relative_tilt(delivered_world[:, index("Hips")], truth_world[:, index("Hips")])
    angle_series = tilt.pop("_angle")

    # The unnormalised three-point positional residual of the fit this arm took.
    geometry = arm_geometry(name, rest_array, skeleton)
    origin = (landmarks[:, cm.JOINT_INDEX["root"]] if geometry["origin"] == "root_landmark"
              else 0.5 * (landmarks[:, cm.JOINT_INDEX["left_hip"]]
                          + landmarks[:, cm.JOINT_INDEX["right_hip"]]))
    observed = np.stack([spine_truth - origin,
                         landmarks[:, cm.JOINT_INDEX["left_hip"]] - origin,
                         landmarks[:, cm.JOINT_INDEX["right_hip"]] - origin], axis=1)
    matrices = Rotation.from_quat(delivered_world[:, index("Hips")]).as_matrix()
    rotated = np.einsum("nij,kj->nki", matrices, geometry["template"])
    residual_m = np.linalg.norm(rotated - observed, axis=2).mean(axis=1)

    absolute = d9b.absolute_score(fk, landmarks, skeleton)
    absolute_unhoisted = d9b.absolute_score(fk_unhoisted, landmarks, skeleton)
    aligned = rc.score(fk, landmarks, skeleton)

    def joint_miss(name_, array):
        return 1e3 * np.linalg.norm(array[:, index(name_)] - truth[:, index(name_)], axis=1)

    if export_glb:
        # P2 on the oracle needs the EXPORTED file, not the track: the anchor lock is a
        # claim about what a viewer forward-kinematics from the GLB's own arrays, and a
        # code-path instrument cannot see what the exporter wrote (CLAUDE.md). `d3.export`
        # is the D3 gate's own call into the real `export_animated_body_glb`.
        d3.export(track, save / f"oracle-{name}-{seed}.glb")
    np.savez_compressed(
        save / f"oracle-{name}-{seed}.npz",
        root=np.asarray(track.root_translation_m), rotations=rotations,
        contacts=np.asarray(track.foot_contacts),
        pre_root=np.asarray(pre.root_translation_m),
        pre_rotations=np.asarray(pre.local_rotations_xyzw),
        post_root=np.asarray(post.root_translation_m),
        post_rotations=np.asarray(post.local_rotations_xyzw),
        post_contacts=np.asarray(post.foot_contacts),
        fk=fk, hoist=hoist, landmarks=landmarks, truth=truth,
        rest=np.asarray(skeleton.rest_translations_m))

    return {
        "arm": name,
        "code_path": ("src `_pelvis_world_frames`" if name in ("src_default", "C_soma_template",
                      "wrong_origin") or name in getattr(cm, "RIG_REST_PELVIS_MODES", ())
                      else "instrument-side"),
        "fit_geometry": {"origin": geometry["origin"], "note": geometry["note"],
                         "template_mm": (1e3 * geometry["template"]).round(4).tolist()},
        "pelvis_report": {k: v for k, v in pelvis_report.items()
                          if not isinstance(v, np.ndarray)},
        "pelvis_vs_truth_deg": tilt,
        "pelvis_vs_truth_worst_frame_deg": round(float(angle_series.max()), 4),
        "three_point_residual_m": {"median": float(np.median(residual_m)),
                                   "max": float(residual_m.max())},
        "spine_origin_miss_mm": {"hoist_subtracted": summary(joint_miss("Spine", fk_unhoisted)),
                                 "raw": summary(joint_miss("Spine", fk))},
        "hips_origin_miss_mm": {"hoist_subtracted": summary(joint_miss("Hips", fk_unhoisted))},
        "neck_miss_mm": {"hoist_subtracted": summary(joint_miss("Neck", fk_unhoisted))},
        "ABSOLUTE_groups_mm": {
            "hoist_subtracted_whole": d3.groups_mm(absolute_unhoisted),
            "whole": d3.groups_mm(absolute),
            "unhoisted_frames": d9b.groups_masked(absolute, ~hoisted),
            "hoisted_frames": (d9b.groups_masked(absolute, hoisted) if hoisted.any()
                               else None),
            "p95": d9b.groups_p95(absolute)},
        "ALIGNED_rc_score_groups_mm": d3.groups_mm(aligned),
        "hoist_mm": summary(magnitude),
        "hoisted_frames": int(hoisted.sum()),
        "contacts": [int(c) for c in np.asarray(track.foot_contacts).sum(axis=0)],
        "penetration_before_mm": round(
            1e3 * record["diagnostics"]["ground_penetration_before_m"], 4),
    }


def oracle_block(save: Path, arms: tuple[str, ...], baseline: Path | None,
                 export_glb: bool = False) -> dict:
    """The D3 gate's six exact-skeleton bodies, every arm, the ABSOLUTE row banded."""

    save.mkdir(parents=True, exist_ok=True)
    donor = d3.load_track(d3.DELIVERED, 0)
    rotations = np.asarray(donor.local_rotations_xyzw, np.float64)
    roots = np.asarray(donor.root_translation_m, np.float64)
    block: dict = {
        "reference": d3.REF_SYNTH,
        "gauge": ("every banded number is on the ABSOLUTE row. `retarget_cost.score` "
                  "subtracts the leg-root midpoint per frame and is blind to a root move "
                  "(CLAUDE.md, D9b); it is reported as O3 and no band reads it."),
        "truth_motion_blind_spot": (
            "the truth MOTION is D9b's delivered pelvis motion, produced by the fit being "
            "replaced. O1's exactness is a congruence and is unaffected; nothing here says "
            "how much real pelvic motion a performer has."),
        "arms": list(arms),
        "seeds": {},
    }
    for seed in d3.SEEDS:
        rng = np.random.default_rng(seed)
        rest_array, factors = d3.perturbed_rest(rng)
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
        spine_truth = truth[:, index("Spine")]
        toes = np.stack([truth[:, index("LeftToes")], truth[:, index("RightToes")]], axis=1)
        record: dict = {
            "factors": factors,
            "rig_rest_mm": {n: (1e3 * rest_array[index(n)]).round(3).tolist()
                            for n in ("Hips", "Spine", "LeftUpperLeg", "RightUpperLeg")},
            "root_landmark_is_the_hip_midpoint_to_mm": round(float(1e3 * np.abs(
                landmarks[:, cm.JOINT_INDEX["root"]]
                - 0.5 * (landmarks[:, cm.JOINT_INDEX["left_hip"]]
                         + landmarks[:, cm.JOINT_INDEX["right_hip"]])).max()), 9),
            "spine_landmark_is_the_rigs_Spine_joint": True,
            "arms": {},
        }
        for arm in arms:
            record["arms"][arm] = run_arm(arm, seed, rest_array, skeleton, truth,
                                          truth_world, landmarks, spine_truth, toes, save,
                                          export_glb=export_glb and arm == arms[0])
            row = record["arms"][arm]
            print(f"  seed {seed} {arm:20s} tilt med {row['pelvis_vs_truth_deg']['angle']['median']:8.4f} "
                  f"pitch {row['pelvis_vs_truth_deg']['pitch_signed_median']:+8.4f} "
                  f"resid {row['three_point_residual_m']['median']:.6f} m  "
                  f"Spine {row['spine_origin_miss_mm']['hoist_subtracted']['median']:7.3f} mm  "
                  f"Hips {row['hips_origin_miss_mm']['hoist_subtracted']['median']:7.3f} mm  "
                  f"ABS {row['ABSOLUTE_groups_mm']['hoist_subtracted_whole']}  "
                  f"ALIGNED {row['ALIGNED_rc_score_groups_mm']}  "
                  f"hoist p95 {row['hoist_mm']['p95']:6.3f} n={row['hoisted_frames']} "
                  f"contacts {row['contacts']}")
        if baseline is not None:
            record["O2_vs_baseline"] = o2_row(save, baseline, seed, arms, skeleton)
            print(f"  seed {seed} O2 {json.dumps(record['O2_vs_baseline'])}")
        block["seeds"][str(seed)] = record
    return block


def o2_row(save: Path, baseline: Path, seed: int, arms: tuple[str, ...], skeleton) -> dict:
    """O2: the legs, feet and toes against the BASELINE build's own forward kinematics.

    The baseline is this same instrument run under the previous `src/`, so candidate and
    baseline share every definition; bit-identity is NOT claimed (a pelvis frame is
    whole-take) and the band is the card's 0.1 mm.
    """
    shipping = arms[0]
    candidate = np.load(save / f"oracle-{shipping}-{seed}.npz")
    reference = np.load(baseline / f"oracle-src_default-{seed}.npz")
    index = skeleton.index
    leg = 1e3 * np.linalg.norm(
        candidate["fk"][:, [index(n) for n in LEG_JOINTS]]
        - reference["fk"][:, [index(n) for n in LEG_JOINTS]], axis=2)
    hoist = 1e3 * np.linalg.norm(candidate["hoist"] - reference["hoist"], axis=1)
    return {
        "arm": shipping,
        "leg_foot_toe_move_mm": summary(leg),
        "leg_foot_toe_max_mm": round(float(leg.max()), 5),
        "within_0_1_mm": bool(leg.max() <= O2_LEG_MM),
        "contacts_identical": bool(np.array_equal(candidate["contacts"],
                                                  reference["contacts"])),
        "hoist_change_mm": summary(hoist),
        "hoist_within_0_05_mm": bool(hoist.max() <= O2_HOIST_MM),
        "bit_identity_claimed": False,
    }


# ------------------------------------------------------------------------------ the take
def load_build(directory: Path, subject: int) -> dict:
    with np.load(directory / f"subject-{subject:02d}.body-track.npz") as archive:
        track = {k: archive[k] for k in archive.files}
    meta = json.loads((directory / f"subject-{subject:02d}.body-track.json").read_text())
    skeleton = skeleton_for_track_dict(meta)
    inputs = sorted((directory / "converter-inputs").glob("call-*.npz"))
    if len(inputs) <= subject:
        raise SystemExit(f"{directory} carries no converter-inputs dump for subject {subject}")
    with np.load(inputs[subject]) as archive:
        dumped = {k: archive[k] for k in archive.files}
    if not np.array_equal(dumped["positions_world_z_up_m"],
                          track["triangulated_world_positions_z_up_m"]):
        raise SystemExit(
            f"converter-inputs/{inputs[subject].name} is not subject {subject}'s call")
    return {"track": track, "skeleton": skeleton, "inputs": dumped}


def to_rig_y_up(z_up: np.ndarray) -> np.ndarray:
    """The converter's own change of basis, `positions_to_body_track` line for line."""
    points = np.asarray(z_up, dtype=np.float64)[..., (0, 2, 1)].copy()
    points[..., 2] *= -1.0
    return points


def take_subject(directory: Path, subject: int) -> dict:
    data = load_build(directory, subject)
    track, skeleton = data["track"], data["skeleton"]
    index = skeleton.index
    rest = np.asarray(skeleton.rest_translations_m, np.float64)
    points = to_rig_y_up(data["inputs"]["positions_world_z_up_m"])
    spine = to_rig_y_up(data["inputs"]["spine_world_z_up_m"])
    left = points[:, cm.JOINT_INDEX["left_hip"]]
    right = points[:, cm.JOINT_INDEX["right_hip"]]
    hip_mid = 0.5 * (left + right)
    root_landmark = points[:, cm.JOINT_INDEX["root"]]
    # `Root` is the identity in the converter, so `Hips`' LOCAL rotation is its world one.
    hips_local = np.asarray(track["local_rotations_xyzw"][:, index("Hips")], np.float64)
    assert np.allclose(track["local_rotations_xyzw"][:, index("Root")],
                       (0.0, 0.0, 0.0, 1.0)), "Root is not the identity on this build"
    matrices = Rotation.from_quat(hips_local).as_matrix()
    up = matrices[:, :, 1]
    valid = np.isfinite(spine).all(axis=1)

    def angle_between(a, b):
        a = a / np.linalg.norm(a, axis=1)[:, None]
        b = b / np.linalg.norm(b, axis=1)[:, None]
        return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0)))

    guard = est.pelvis_lever_guard(spine, hip_mid, cm.SEGMENT_LENGTH_CEILING_FRACTION)
    # the pelvis's own step, frame to frame, and the 800 deg/s physical ceiling
    steps = np.degrees(np.linalg.norm(
        (Rotation.from_quat(hips_local[1:]) * Rotation.from_quat(hips_local[:-1]).inv()
         ).as_rotvec(), axis=1))
    fk = forward_kinematics_positions(track["root_translation_m"],
                                      track["local_rotations_xyzw"],
                                      skeleton=skeleton).astype(np.float64)
    leg_root_mid = 0.5 * (fk[:, index("LeftUpperLeg")] + fk[:, index("RightUpperLeg")])
    hoist = hoist_by_converter_line(track, skeleton, points)
    return {
        "frames": int(len(points)),
        "spine_resolved_frames": int(valid.sum()),
        "delivered_up_vs_spine1_minus_hip_midpoint_deg": summary(
            angle_between(up[valid], (spine - hip_mid)[valid])),
        "delivered_up_vs_spine1_minus_root_landmark_deg": summary(
            angle_between(up[valid], (spine - root_landmark)[valid])),
        "root_landmark_above_hip_midpoint_mm": summary(
            1e3 * np.linalg.norm((root_landmark - hip_mid)[valid], axis=1)),
        "pelvis_lever": {
            "pre_guard_median_mm": round(1e3 * guard["median_m"], 4),
            "ceiling_fraction": cm.SEGMENT_LENGTH_CEILING_FRACTION,
            "demoted_count": int(guard["off"].sum()),
            "demoted_frames": [int(i) for i in np.flatnonzero(guard["off"])],
            "lever_mm": summary(1e3 * guard["lever_m"][guard["finite"]]),
            "lever_p5_p95_pct_of_median": [
                round(float(100 * (np.percentile(guard["lever_m"][guard["finite"]], q)
                                   / guard["median_m"] - 1)), 3) for q in (5, 95)],
            "resolved_fraction_after_the_guard": round(
                float((valid & ~guard["off"]).mean()), 4),
            "note": ("the median is frozen from the unchanged PRE-guard input, which is "
                     "byte-identical on every arm of this step (converter-only change)"),
        },
        "pelvis_step_deg_per_frame": summary(steps),
        "frames_over_800_deg_per_s": int((steps > 800.0 / 30.0).sum()),
        "leg_roots_on_captured_hip_midpoint_mm": summary(
            1e3 * np.linalg.norm(leg_root_mid - hoist - hip_mid, axis=1)),
        "hoist_mm": summary(1e3 * np.linalg.norm(hoist, axis=1)),
        "contacts": [int(c) for c in track["foot_contacts"].sum(axis=0)],
    }


def hoist_by_converter_line(track: dict, skeleton, points: np.ndarray) -> np.ndarray:
    """The converter's own root line, inverted: `root - (pelvis - rest[Hips] - R . mid)`."""
    index = skeleton.index
    rest = np.asarray(skeleton.rest_translations_m, np.float64)
    pelvis = 0.5 * (points[:, cm.JOINT_INDEX["left_hip"]]
                    + points[:, cm.JOINT_INDEX["right_hip"]])
    mid = 0.5 * (rest[index("LeftUpperLeg")] + rest[index("RightUpperLeg")])
    rotation = Rotation.from_quat(
        np.asarray(track["local_rotations_xyzw"][:, index("Hips")], np.float64))
    pre = pelvis - rest[index("Hips")] - rotation.apply(
        np.broadcast_to(mid, (len(pelvis), 3)))
    return np.asarray(track["root_translation_m"], np.float64) - pre


def take_block(directory: Path, label: str, baseline: Path | None) -> dict:
    block: dict = {"label": label, "directory": str(directory), "subjects": {}}
    for subject in (0, 1):
        row = take_subject(directory, subject)
        if baseline is not None:
            row["vs_baseline"] = take_pair(baseline, directory, subject)
        block["subjects"][f"subject_{subject:02d}"] = row
        print(f"  {label} subject {subject}: up vs Spine1-hipmid "
              f"{row['delivered_up_vs_spine1_minus_hip_midpoint_deg']['median']:.3f} deg, "
              f"lever median {row['pelvis_lever']['pre_guard_median_mm']:.4f} mm, "
              f"demoted {row['pelvis_lever']['demoted_count']} "
              f"{row['pelvis_lever']['demoted_frames']}, contacts {row['contacts']}")
    return block


def take_pair(baseline: Path, candidate: Path, subject: int) -> dict:
    """Candidate against baseline on one performer, every root figure hoist-subtracted."""
    base = load_build(baseline, subject)
    cand = load_build(candidate, subject)
    skeleton = base["skeleton"]
    index = skeleton.index
    if not np.array_equal(np.asarray(base["skeleton"].rest_translations_m),
                          np.asarray(cand["skeleton"].rest_translations_m)):
        rest_moved = True
    else:
        rest_moved = False
    landmarks_identical = bool(np.array_equal(
        base["track"]["triangulated_world_positions_z_up_m"],
        cand["track"]["triangulated_world_positions_z_up_m"]))
    points = to_rig_y_up(base["inputs"]["positions_world_z_up_m"])
    base_hips = np.asarray(base["track"]["local_rotations_xyzw"][:, index("Hips")], np.float64)
    cand_hips = np.asarray(cand["track"]["local_rotations_xyzw"][:, index("Hips")], np.float64)
    relative = Rotation.from_quat(cand_hips) * Rotation.from_quat(base_hips).inv()
    vector = relative.as_rotvec()
    angle = np.degrees(np.linalg.norm(vector, axis=1))
    hip_axis = Rotation.from_quat(base_hips).as_matrix()[:, :, 0]
    pitch = np.degrees(np.einsum("ni,ni->n", vector, hip_axis))
    base_hoist = hoist_by_converter_line(base["track"], skeleton, points)
    cand_hoist = hoist_by_converter_line(cand["track"], skeleton, points)
    fk_base = forward_kinematics_positions(base["track"]["root_translation_m"],
                                           base["track"]["local_rotations_xyzw"],
                                           skeleton=skeleton).astype(np.float64)
    fk_cand = forward_kinematics_positions(cand["track"]["root_translation_m"],
                                           cand["track"]["local_rotations_xyzw"],
                                           skeleton=skeleton).astype(np.float64)
    move = 1e3 * np.linalg.norm((fk_cand - cand_hoist[:, None])
                                - (fk_base - base_hoist[:, None]), axis=2)
    root_base = np.asarray(base["track"]["root_translation_m"], np.float64) - base_hoist
    root_cand = np.asarray(cand["track"]["root_translation_m"], np.float64) - cand_hoist
    forward = Rotation.from_quat(base_hips).as_matrix()[:, :, 2]
    # THE HIP RESIDUAL, and under (a) it is a REPORT and never a band. Under (b) the
    # observed hip line is an exact axis of the delivered frame, so the ANGULAR and
    # TRANSVERSE parts are zero by construction and only the rig-vs-performer WIDTH
    # mismatch survives. `E_rig_rest_kabsch` does not hold that axis exactly: the 197 mm
    # `Spine` lever pulls against it inside one un-centred SVD, which is the whole trade the
    # selector decided. The card is explicit that no hip-residual band may be manufactured
    # from these numbers.
    def hip_residual(track: dict, hips_quat: np.ndarray) -> dict:
        matrices = Rotation.from_quat(hips_quat).as_matrix()
        across = matrices[:, :, 0]
        left = points[:, cm.JOINT_INDEX["left_hip"]]
        right = points[:, cm.JOINT_INDEX["right_hip"]]
        observed = left - right
        unit = observed / np.linalg.norm(observed, axis=1)[:, None]
        angular = np.degrees(np.arccos(np.clip(np.sum(unit * across, axis=1), -1.0, 1.0)))
        hip_mid = 0.5 * (left + right)
        rest_local = np.asarray(skeleton.rest_translations_m, np.float64)
        mid = 0.5 * (rest_local[index("LeftUpperLeg")] + rest_local[index("RightUpperLeg")])
        transverse, full = [], []
        for side, sign in (("LeftUpperLeg", 1.0), ("RightUpperLeg", 1.0)):
            modelled = hip_mid + np.einsum(
                "nij,j->ni", matrices, rest_local[index(side)] - mid)
            captured = left if side == "LeftUpperLeg" else right
            delta = captured - modelled
            along = np.sum(delta * across, axis=1)[:, None] * across
            transverse.append(1e3 * np.linalg.norm(delta - along, axis=1))
            full.append(1e3 * np.linalg.norm(delta, axis=1))
        return {"angular_deg": summary(angular),
                "transverse_mm": summary(np.concatenate(transverse)),
                "full_positional_mm": summary(np.concatenate(full))}

    return {
        "landmarks_byte_identical_same_denominator": landmarks_identical,
        "rest_skeleton_moved": rest_moved,
        "pelvis_change_deg": {"angle": summary(angle),
                              "pitch_about_hip_line_signed_median": round(
                                  float(np.median(pitch)), 4)},
        "hip_residual_REPORT_never_a_band": {
            "note": ("under (b) the angular and transverse parts are zero by construction "
                     "and only the width mismatch survives; `E_rig_rest_kabsch` does not "
                     "hold the observed hip line exactly. A REPORT, per the card."),
            "baseline": hip_residual(base["track"], base_hips),
            "candidate": hip_residual(cand["track"], cand_hips)},
        "root_move_mm_hoist_subtracted": summary(
            1e3 * np.linalg.norm(root_cand - root_base, axis=1)),
        "root_move_fore_aft_signed_median_mm": round(float(np.median(
            1e3 * np.einsum("ni,ni->n", root_cand - root_base, forward))), 4),
        "joint_move_mm_hoist_subtracted": {
            n: summary(move[:, index(n)]) for n in REPORTED_JOINTS},
        "hoist_mm": {"baseline": summary(1e3 * np.linalg.norm(base_hoist, axis=1)),
                     "candidate": summary(1e3 * np.linalg.norm(cand_hoist, axis=1))},
        "contacts": {"baseline": [int(c) for c in base["track"]["foot_contacts"].sum(0)],
                     "candidate": [int(c) for c in cand["track"]["foot_contacts"].sum(0)],
                     "identical": bool(np.array_equal(base["track"]["foot_contacts"],
                                                      cand["track"]["foot_contacts"]))},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--oracle-save", type=Path)
    parser.add_argument("--oracle-baseline", type=Path)
    parser.add_argument("--oracle-export-glb", action="store_true",
                        help="export the SHIPPING arm's track to a GLB per seed, so P2's "
                             "anchor lock can be measured on the oracle bodies from the "
                             "exported file's own arrays rather than from the track")
    parser.add_argument("--arms", default="src_default,C_soma_template,wrong_origin,"
                                          "D_rig_rest_hipline,E_rig_rest_kabsch,"
                                          "frozen_upright")
    parser.add_argument("--take", type=Path)
    parser.add_argument("--take-label", default="build")
    parser.add_argument("--take-baseline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--src-stage", choices=("pre_change", "refactored"),
                        default="refactored",
                        help="which source stage this instrument run belongs to; stated by "
                             "the operator and checked by the gate against the converter's "
                             "hash and git")
    args = parser.parse_args()

    report: dict = {
        "title": "D7c -- the pelvis on the rig's own rest",
        "resolved_module": str(Path(cm.__file__).resolve()),
        # THE SAME BUILD-TIME CONTRACT THE BUILD REPORTS CARRY. This producer emitted none at
        # all until Astra's round 9, so every instrument report had to be stamped after the
        # fact; a run from here records its own.
        "source_fingerprint": source_fingerprint(cm.PELVIS_FRAME_SOURCE,
                                                 stage=args.src_stage),
        "pelvis_frame_source_in_src": cm.PELVIS_FRAME_SOURCE,
        "bands": {"O1_tilt_deg": O1_TILT_DEG, "O1_origin_mm": O1_ORIGIN_MM,
                  "O1_residual_m": O1_RESIDUAL_M, "O2_leg_mm": O2_LEG_MM,
                  "O2_hoist_mm": O2_HOIST_MM},
    }
    if args.oracle:
        save = args.oracle_save or (OUT_DIR / "oracle")
        save = save if save.is_absolute() else ROOT / save
        baseline = args.oracle_baseline
        if baseline is not None and not baseline.is_absolute():
            baseline = ROOT / baseline
        print(f"ORACLE -> {save}")
        report["oracle"] = oracle_block(save, tuple(args.arms.split(",")), baseline,
                                        export_glb=args.oracle_export_glb)
    if args.take is not None:
        take = args.take if args.take.is_absolute() else ROOT / args.take
        baseline = args.take_baseline
        if baseline is not None and not baseline.is_absolute():
            baseline = ROOT / baseline
        print(f"TAKE -> {take}")
        report["take"] = take_block(take, args.take_label, baseline)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
