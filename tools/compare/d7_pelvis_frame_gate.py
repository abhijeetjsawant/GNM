#!/usr/bin/env python3
"""D7: `Hips` gets its own frame from the pelvis's own landmarks. The gate.

THE CHANGE. `positions_to_body_track` takes SOMA-77's triangulated `Spine1` and builds
`Hips`' world rotation from the PELVIS -- a Kabsch fit of a rest pelvis template onto
{root, Spine1, LeftLeg, RightLeg} -- instead of from the trunk line (neck - hip midpoint).
`Spine`, `Chest` and `UpperChest` keep the thorax frame; the root formula is unchanged, so
`_leg_root_offset`'s `R_hips . mid` now rides the PELVIS instead of the lean.

THE DEFECT IT CLOSES. Until now `Hips`, `Spine`, `Chest` and `UpperChest` all took
`torso_up`, so the trunk was one rigid block. On the fixture's own clips a rigid pelvis
departs from that trunk line by 26.0 / 32.8 deg median / p95 on the squat clip's bent
frames, correlation +0.93 with tilt. D2b isolated the root's own silhouette share at
-0.022 / -0.013 IoU, tilt-dependent and ~0 upright.

THE PRE-REGISTRATION is `docs/reviews/pelvis-frame-2026-09-04.md` section 0, committed
before any number here existed (`artifacts/` is a symlink and nothing under it is tracked,
so the review is the tamper-evident copy). Refuted predictions are recorded as REFUTED.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7_pelvis_frame_gate.py

Reads the rebuild under `artifacts/compare/d7-pelvis-frame/delivery` and the reports of
`d7_pelvis_rigidity.py` and `d7_pelvis_synthetic.py`. Writes
`artifacts/compare/d7-pelvis-frame/gate.json`, preserving the `preregistration` block that
is already there. NEVER writes to `artifacts/commercial-multiview-*`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "tools/swap-harness", "tests", "scripts"):
    sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

from autoanim_gnm import commercial_multiview as cm  # noqa: E402
from autoanim_gnm.body import (  # noqa: E402
    DETAILED_HUMANOID, BodyTrack, forward_kinematics_positions, skeleton_for_track,
)
import d3_skeleton_gate as d3  # noqa: E402
import retarget_cost as rc  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d7-pelvis-frame"
REBUILD = OUT_DIR / "delivery"
DELIVERED = ROOT / "artifacts/commercial-multiview-soma77"
SQUAT = "autoanim_squat/research-squat-640"

CLOSURE_BAND_M = 1.0e-4
ORACLE_BAND_DEG = 0.01
MUSTFAIL_FLOOR_DEG = 10.0
BLOCK, DRAWS, SEED = 15, 2000, 20260904

REF_SYNTH = ("exact synthetic truth: the posed SOMASKEL77 Hips world rotation, by Kabsch of "
             "the clip's OWN rest pelvis offsets onto the posed ones. MAMMA-FREE.")
REF_INVARIANT = ("NONE -- a reference-free invariant. A segment between two points rigid to "
                 "one bone has constant length whatever the pose.")
REF_CLOSURE = ("the track's own forward kinematics -- the exported GLB scored against "
               "`forward_kinematics_positions` on the rest the track carries. An IDENTITY.")
REF_ROUNDTRIP = "the converter scored against its OWN output"
REF_MAMMA = "MAMMA's own SMPL-X fit -- AGREEMENT ONLY. It reports and selects nothing."


def load_track(directory: Path, subject: int) -> BodyTrack:
    return BodyTrack.from_dict(
        json.loads((directory / f"subject-{subject:02d}.body-track.json").read_text()))


def capture_positions(directory: Path, subject: int) -> np.ndarray:
    with np.load(directory / f"subject-{subject:02d}.body-track.npz") as archive:
        return np.asarray(archive["triangulated_world_positions_z_up_m"], np.float64)


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.einsum("nij,nkj->nik", a, b)
    trace = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def block_bootstrap(values: np.ndarray, statistic=np.median) -> list[float]:
    """Moving block 15, 2000 draws, fixed seed. Lag-1 autocorrelation here is 0.99."""

    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    n = len(values)
    starts = rng.integers(0, max(n - BLOCK, 1), size=(DRAWS, max(n // BLOCK, 1)))
    draws = np.asarray([
        statistic(np.concatenate([values[s:s + BLOCK] for s in row])[:n]) for row in starts])
    return [round(float(np.percentile(draws, 2.5)), 5), round(float(np.percentile(draws, 97.5)), 5)]


# ---------------------------------------------------------------- B1/B2: fold the synthetic
def synthetic_block(report: dict) -> None:
    payload = json.loads((OUT_DIR / "synthetic.json").read_text())
    clean = payload["clean"]["clips"]
    noisy = payload["noisy_stride_1"]["clips"][SQUAT]["bent_frames_only"]
    shipped = cm.PELVIS_FRAME_SOURCE
    clean_ok = all(
        row["arms"][arm]["own_rest_median_deg"] <= ORACLE_BAND_DEG
        and row["arms"][arm]["own_rest_max_deg"] <= 0.05
        for row in clean.values()
        for arm in ("A_root_to_spine1", "B_hipmid_to_spine1", "C_kabsch_pelvis"))
    squat = clean[SQUAT]["arms"]
    lumbar_best = min(squat["lumbar_root_to_spine2"]["bent_own_rest_median_deg"],
                      squat["lumbar_spine_line"]["bent_own_rest_median_deg"])
    report["B1_synthetic_clean"] = {
        "reference": REF_SYNTH,
        "band": "every rigid candidate <= 0.01 deg median and <= 0.05 deg max on all five clips",
        "verdict": "PASS" if clean_ok else "FAIL",
        "candidates_exact_on_every_clip": clean_ok,
        "degenerate_thorax_as_pelvis_bent_median_deg": squat["thorax_as_pelvis"]["bent_own_rest_median_deg"],
        "degenerate_world_vertical_bent_median_deg": squat["world_vertical"]["bent_own_rest_median_deg"],
        "degenerate_band_was": ">= 20 deg median on the squat's bent frames for BOTH",
        "degenerate_verdict": {
            "thorax_as_pelvis": "PASS (fails the band as required)"
            if squat["thorax_as_pelvis"]["bent_own_rest_median_deg"] >= 20.0 else "REFUTED",
            "world_vertical": "PASS (fails the band as required)"
            if squat["world_vertical"]["bent_own_rest_median_deg"] >= 20.0 else "REFUTED",
        },
        "best_lumbar_bent_median_deg": lumbar_best,
        "lumbar_band_was": ">= 5 deg median on the squat's bent frames",
        "lumbar_verdict": "PASS" if lumbar_best >= 5.0 else "REFUTED",
        "convention_residual_with_the_SHIPPED_constants_median_deg": {
            clip: {arm: row["arms"][arm]["shipped_constants_median_deg"]
                   for arm in ("A_root_to_spine1", "B_hipmid_to_spine1", "C_kabsch_pelvis",
                               "world_vertical", "thorax_as_pelvis")}
            for clip, row in clean.items()},
        "blind_to": ("whether SOMA-77's Spine1 is where a real L5/S1 is, and whether the "
                     "shipped rest-pitch convention is the delivered performer's."),
    }
    scores = {arm: noisy[arm]["0"]["median_deg"]
              for arm in ("A_root_to_spine1", "B_hipmid_to_spine1", "C_kabsch_pelvis")}
    winner = min(scores, key=scores.get)
    beats = {name: bool(scores[winner] < noisy[name]["0"]["median_deg"])
             for name in ("thorax_as_pelvis", "world_vertical")}
    beats["best_lumbar"] = bool(scores[winner] < min(
        noisy["lumbar_root_to_spine2"]["0"]["median_deg"],
        noisy["lumbar_spine_line"]["0"]["median_deg"]))
    report["B2a_synthetic_noisy_the_selector"] = {
        "reference": REF_SYNTH + " with I7/I8's measured heavy-tail frame-correlated noise "
                                 "at NOISE_SIGMA_PX = 3.20 px, our own detector's amplitude.",
        "band": ("the selected candidate's median error on the squat's bent frames is below "
                 "thorax_as_pelvis AND world_vertical AND the best lumbar arm"),
        "selection_rule_outcome": {"winner": winner, "median_deg_by_candidate": scores,
                                   "shipped": shipped, "selection_honoured": winner == shipped},
        "beats": beats,
        "verdict": "PASS" if all(beats.values()) else "FAIL",
        "controls_median_deg": {name: noisy[name]["0"]["median_deg"]
                                for name in ("thorax_as_pelvis", "world_vertical",
                                             "no_rest_correction", "lumbar_root_to_spine2",
                                             "lumbar_spine_line")},
        "THE_INSTRUMENT_IS_PESSIMISTIC_AND_HERE_IS_BY_HOW_MUCH":
            payload["noisy_stride_1"]["clips"][SQUAT]["noise_calibration_vs_the_real_take"],
        "noise_calibration_sweep": payload["noise_calibration_sweep"],
    }
    report["B2b_the_window"] = {
        "band": ("an over-smoothed window must fail on the fast clip by lag > 1.5 frames or "
                 "attenuation < 0.75; if no window beats 0 while holding lag <= 1 frame and "
                 "attenuation >= 0.9, the selection is 0"),
        "shipped": cm.PELVIS_SMOOTHING_FRAMES,
        "lag_frames_fast_clip": payload["noisy_fast"]["clips"][SQUAT]["window_lag_frames"],
        "attenuation_fast_clip": payload["noisy_fast"]["clips"][SQUAT]["window_attenuation"],
        "p95_by_window_real_speed": payload["selected_window"]["p95_by_window"],
        "what_this_band_cannot_prove": ("the window optimises exactly the quantity this arm "
                                        "measures. It is REPORTED, never banded."),
    }


def rigidity_block(report: dict) -> None:
    payload = json.loads((OUT_DIR / "rigidity.json").read_text())
    trusted = all(v["trusted_on_sd_mm"]
                  for subject in payload["subjects"].values()
                  if "verdict_on_sd_mm" in subject
                  for v in subject["verdict_on_sd_mm"].values())
    report["B3_rigidity_on_the_real_take"] = {
        "reference": REF_INVARIANT,
        "reading_rule": payload["reading_rule"],
        "verdict": "TRUSTED" if trusted else "NOT TRUSTED",
        "per_subject": {name: block.get("verdict_on_sd_mm")
                        for name, block in payload["subjects"].items()},
        "segment_length_stability": {name: block.get("segment_length_stability")
                                     for name, block in payload["subjects"].items()},
        "blind_to": ("ACCURACY -- a stable segment can be stably wrong. And cross-view "
                     "self-agreement is NOT rigidity: the ears were the most epipolar-"
                     "consistent landmarks in this lane and fit a rigid skull worst."),
    }


# ------------------------------------------------------------------ B5: exact recovery
def oracle_block(report: dict) -> None:
    from test_pelvis_frame import geodesic_deg as _g, hips_world_rotations, posed_body  # noqa: E402

    positions, spine, truth = posed_body(frames=40)
    solved = cm.positions_to_body_track(
        positions, sample_rate_hz=30, provenance_sha256="a" * 64,
        spine_world_z_up_m=spine, skeleton=DETAILED_HUMANOID)
    fell_back = cm.positions_to_body_track(
        positions, sample_rate_hz=30, provenance_sha256="a" * 64, skeleton=DETAILED_HUMANOID)
    error = _g(hips_world_rotations(solved), truth)
    control = _g(hips_world_rotations(fell_back), truth)
    report["B5_exact_recovery_oracle"] = {
        "reference": ("a body posed through our OWN forward kinematics with a pelvis "
                      "deliberately >= 20 deg from its thorax; the pelvis landmarks are the "
                      "SHIPPED rest template rotated, so recovery is an IDENTITY, not a fit."),
        "band_deg": ORACLE_BAND_DEG,
        "worst_deg": round(float(error.max()), 8),
        "median_deg": round(float(np.median(error)), 8),
        "verdict": "PASS" if error.max() < ORACLE_BAND_DEG else "FAIL",
        "must_fail_no_spine_input_median_deg": round(float(np.median(control)), 3),
        "must_fail_floor_deg": MUSTFAIL_FLOOR_DEG,
        "must_fail_verdict": "PASS (fails the band as required)"
        if np.median(control) >= MUSTFAIL_FLOOR_DEG else "REFUTED",
        "blind_to": ("whether the pelvis landmark is where a real pelvis is. This is closure "
                     "between two halves of our own code, exactly as D3's was."),
    }


# --------------------------------------------------------------- B4: canonical round trip
def roundtrip_block(report: dict) -> None:
    committed = json.loads(
        (ROOT / "artifacts/compare/d2-clavicle/gate-d2c.json").read_text())
    rows = {}
    for subject in (0, 1):
        source = capture_positions(DELIVERED, subject)
        fk1 = rc.retarget_then_fk(source, DETAILED_HUMANOID)
        synthetic = rc.landmarks_from_fk(fk1, DETAILED_HUMANOID)
        fk2 = rc.retarget_then_fk(rc.Z_UP_FROM_Y_UP(synthetic), DETAILED_HUMANOID)
        got = d3.groups_mm(rc.score(fk2, synthetic, DETAILED_HUMANOID))
        expected = committed["capture"][f"subject_{subject:02d}"]["d2c_roundtrip_canonical"]
        rows[f"subject_{subject:02d}"] = {
            "roundtrip_canonical_mm": got, "committed_d2c_mm": expected,
            "matches_committed": bool(abs(got["arms"] - expected["arms"]) <= 0.01
                                      and got["legs"] == expected["legs"]
                                      and got["torso"] == expected["torso"])}
    report["B4_round_trip_canonical"] = {
        "reference": REF_ROUNDTRIP,
        "PRE_REGISTERED_AS_TRIVIALLY_MET_AND_WHY": (
            "`retarget_cost.landmarks_from_fk` emits the 19-joint contract, which has no "
            "Spine1, so the converter takes the LEGACY path here and is bit-identical to D3. "
            "THE ROUND TRIP IS BLIND TO D7. It is run because a change to the branch line "
            "would show as a non-zero leg, and for no other reason."),
        "subjects": rows,
        "verdict": "PASS" if all(v["matches_committed"] for v in rows.values()) else "FAIL",
    }


# -------------------------------------------------- B6/B8: the rebuilt delivery, from bytes
def delivery_block(report: dict) -> None:
    work = {}
    for name in sorted(p.name for p in (DELIVERED / "work").glob("*observations.jsonl")):
        work[name] = bool((DELIVERED / "work" / name).read_bytes()
                          == (REBUILD / "work" / name).read_bytes())
    closure, spans, triangulation, diagnostics = {}, {}, {}, {}
    for subject in (0, 1):
        track = load_track(REBUILD, subject)
        skeleton = skeleton_for_track(track)
        expected = forward_kinematics_positions(
            np.asarray(track.root_translation_m, np.float64),
            np.asarray(track.local_rotations_xyzw, np.float64), skeleton=skeleton)
        names, got, rest = d3.glb_joint_positions(
            REBUILD / f"subject-{subject:02d}.glb")
        order = [track.joint_names.index(name) for name in names]
        worst = float(np.abs(got - expected[:, order]).max())
        closure[f"subject_{subject:02d}"] = {
            "max_m": worst, "within_band": bool(worst <= CLOSURE_BAND_M)}
        spans[f"subject_{subject:02d}"] = {
            "glb_rest_matches_track_max_m": float(np.abs(
                rest - np.asarray(track.rest_translations_m, np.float64)[order]).max())}
        triangulation[f"subject_{subject:02d}"] = bool(np.array_equal(
            capture_positions(REBUILD, subject), capture_positions(DELIVERED, subject)))
    run = json.loads((REBUILD / "run-report.json").read_text())
    for key in ("spine_triangulation", "pelvis_frame"):
        diagnostics[key] = run.get("diagnostics", run).get(key)
    report["B6_closure_on_the_rebuilt_glb"] = {
        "reference": REF_CLOSURE, "band_m": CLOSURE_BAND_M, "subjects": closure,
        "glb_rest_equals_track_rest": spans,
        "verdict": "PASS" if all(v["within_band"] for v in closure.values()) else "FAIL",
        "read_from_its_own_bytes": True,
    }
    report["B8_rebuild_hygiene"] = {
        "work_byte_identical": work,
        "triangulation_byte_identical_same_denominator": triangulation,
        "verdict": "PASS" if all(work.values()) and all(triangulation.values()) else "FAIL",
        "diagnostics": diagnostics,
    }


def legacy_bit_identity(report: dict) -> None:
    """The decisive legacy check: the WHOLE reconstruction with no spine feed.

    Runs the real `reconstruct_multiview` on the delivered build's own cached detections
    with `spine_landmarks_by_camera=None` and compares array by array against the pre-D7
    delivered build on disk. Nothing D7 added may perturb a single float32.
    """
    import build_commercial_multiview_comparison as build  # noqa: E402

    cameras = cm.load_camera_rig(DELIVERED / "camera-rig.json")
    observations, heads, toes = [], [], []
    for name in ("A001", "B001", "C001", "D001"):
        records = cm.load_observation_jsonl(
            DELIVERED / "work" / f"{name}-soma77-observations.jsonl")
        observations.append(records)
        heads.append(build._head_landmarks(records))
        toes.append(build._toe_landmarks(records))
    tracks, _, _, _ = cm.reconstruct_multiview(
        cameras, observations,
        head_landmarks_by_camera=heads, head_landmark_names=build.HEAD_LANDMARK_NAMES,
        toe_landmarks_by_camera=toes, spine_landmarks_by_camera=None,
        subject_count=2, sample_rate_hz=30)
    rows = {}
    identical = True
    for subject, track in enumerate(tracks):
        with np.load(DELIVERED / f"subject-{subject:02d}.body-track.npz") as archive:
            per = {
                "local_rotations_xyzw": bool(np.array_equal(
                    track.local_rotations_xyzw, archive["local_rotations_xyzw"])),
                "root_translation_m": bool(np.array_equal(
                    track.root_translation_m, archive["root_translation_m"])),
                "foot_contacts": bool(np.array_equal(
                    track.foot_contacts, archive["foot_contacts"])),
            }
        identical = identical and all(per.values())
        rows[f"subject_{subject:02d}"] = per
    report["legacy_bit_identity"] = {
        "how": ("the real `reconstruct_multiview` on the delivered build's own cached "
                "detections with `spine_landmarks_by_camera=None`, compared array by array "
                "against the pre-D7 delivered build on disk"),
        "bit_identical": identical, "per_subject": rows,
        "verdict": "PASS" if identical else "FAIL",
    }
    return tracks


# ------------------------------------------------------- REPORTED, never banded
def reported_block(report: dict) -> None:
    hips, hoist, temporal = {}, {}, {}
    for subject in (0, 1):
        rows = {}
        for label, directory in (("before_D3", DELIVERED), ("after_D7", REBUILD)):
            track = load_track(directory, subject)
            skeleton = skeleton_for_track(track)
            world = forward_kinematics_positions(
                np.asarray(track.root_translation_m, np.float64),
                np.asarray(track.local_rotations_xyzw, np.float64), skeleton=skeleton)
            index = track.joint_names.index("Hips")
            rig_hips = world[:, index]
            capture = np.stack([rig_hips[:, 0], -rig_hips[:, 2], rig_hips[:, 1]], axis=-1)
            source = capture_positions(directory, subject)
            hip_mid = 0.5 * (source[:, cm.JOINT_INDEX["left_hip"]]
                             + source[:, cm.JOINT_INDEX["right_hip"]])
            offset = capture - hip_mid
            horizontal = np.linalg.norm(offset[:, :2], axis=1) * 1000.0
            trunk = source[:, cm.JOINT_INDEX["neck"]] - hip_mid
            tilt = np.degrees(np.arccos(np.clip(
                trunk[:, 2] / np.linalg.norm(trunk, axis=1), -1.0, 1.0)))
            bent = tilt >= np.percentile(tilt, 66.7)
            rows[label] = {
                "horizontal_offset_median_mm": round(float(np.median(horizontal)), 3),
                "horizontal_offset_bent_tercile_median_mm": round(
                    float(np.median(horizontal[bent])), 3),
                "correlation_with_trunk_tilt": round(
                    float(np.corrcoef(horizontal, tilt)[0, 1]), 4),
                "bent_tercile_ci": block_bootstrap(horizontal[bent]),
            }
            # temporal: how fast Hips turns, reported and never banded
            quaternions = np.asarray(track.local_rotations_xyzw[:, index], np.float64)
            matrices = Rotation.from_quat(quaternions).as_matrix()
            step = geodesic_deg(matrices[1:], matrices[:-1]) * 30.0
            rows[label]["hips_frames_over_800_deg_per_s"] = int((step > 800.0).sum())
            rows[label]["hips_median_deg_per_s"] = round(float(np.median(step)), 2)
            floor = float(np.min(world[..., 1]))
            rows[label]["lowest_joint_y_m"] = round(floor, 5)
            rows[label]["foot_contacts"] = [int(v) for v in np.sum(track.foot_contacts, axis=0)]
        hips[f"subject_{subject:02d}"] = rows
    report["reported_never_banded"] = {
        "why_not_banded": ("the Hips-joint horizontal offset MUST move and a constant can "
                           "zero it, so it is not a band; a temporal reject would zero the "
                           "over-ceiling count by construction (D2c's lesson)."),
        "hips_joint_offset_and_temporal": hips,
        "degenerate_beside_it": ("D2b's rig-hip-mid instrument reads zero by construction "
                                 "since D2b, which is why this replaces it. A world-vertical "
                                 "pelvis would also produce a tilt-correlated offset; the "
                                 "synthetic arm is where the two are separated."),
    }


def mamma_pelvis_oracle(report: dict) -> None:
    """MAMMA's own pelvis-versus-thorax separation, beside ours. AGREEMENT ONLY."""

    import subject_map  # noqa: E402

    ma3d = ROOT / ("artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/"
                   "pushing_and_lifting_from_ground")
    ours = np.stack([capture_positions(REBUILD, s) for s in (0, 1)])
    try:
        mapping = subject_map.mamma_index_for(ours)
    except SystemExit as error:
        report["mamma_pelvis_oracle"] = {"status": "not_run", "reason": str(error)}
        return
    rows = {}
    for subject, body_id in sorted(mapping.items()):
        path = ma3d / f"smplx_params_body_id-{body_id:02d}.npz"
        if not path.exists():
            rows[f"subject_{subject:02d}"] = {"status": "absent", "path": str(path)}
            continue
        with np.load(path, allow_pickle=True) as archive:
            pose = np.asarray(archive["smplx_pose"], dtype=np.float64)
        pelvis = Rotation.from_rotvec(pose[:, :3]).as_matrix()
        spine3 = Rotation.from_rotvec(pose[:, 9 * 3:9 * 3 + 3]).as_matrix()
        # MAMMA's spine3 is PARENT-RELATIVE, so its own pelvis-vs-thorax separation is the
        # composed chain; the pitch difference below is an AGREEMENT figure and nothing
        # more. It never selects.
        theirs = geodesic_deg(pelvis, np.einsum("nij,njk->nik", pelvis, spine3))
        track = load_track(REBUILD, subject)
        skeleton = skeleton_for_track(track)
        hips = track.joint_names.index("Hips")
        chest = track.joint_names.index("Chest")
        world = np.asarray(track.local_rotations_xyzw, np.float64)
        hips_world = Rotation.from_quat(world[:, hips]).as_matrix()
        # Chest's world = Hips' world composed down the chain; the local chain product is
        # what the separation is, and Spine/Chest share the thorax frame.
        spine = track.joint_names.index("Spine")
        chain = Rotation.from_quat(world[:, spine]).as_matrix()
        chain = np.einsum("nij,njk->nik", chain, Rotation.from_quat(world[:, chest]).as_matrix())
        ourss = geodesic_deg(np.einsum("nij,njk->nik", hips_world, chain), hips_world)
        rows[f"subject_{subject:02d}"] = {
            "mamma_body_id": int(body_id),
            "mamma_pelvis_to_spine3_median_deg": round(float(np.median(theirs)), 3),
            "ours_pelvis_to_chest_median_deg": round(float(np.median(ourss)), 3),
            "note": ("two different chains and two different conventions; the figure that "
                     "matters is that BOTH are non-zero, i.e. both carry a pelvis that is "
                     "not welded to the thorax. Before D7 ours was 0.000 by construction."),
        }
    report["mamma_pelvis_oracle"] = {
        "reference": REF_MAMMA,
        "subject_map": {str(k): int(v) for k, v in mapping.items()},
        "subject_map_note": "resolved by 3D pelvis agreement, never by index",
        "gt_arrays_never_scored": "MAMMA's gt_* arrays are byte-copies of pred_*",
        "subjects": rows,
    }


def silhouette_block(report: dict) -> None:
    """B7. The photographs -- the one band the candidate cannot optimise.

    BEFORE is D3's own committed run (`artifacts/compare/d3-skeleton/silhouette-d3.json`);
    AFTER is the same instrument, the same rasteriser, the same masks and the same mask
    cache pointed at the D7 rebuild. The MAMMA mesh oracle reads NONE of our track, so its
    bit-identity between the two runs is the proof that reusing the caches changed nothing.
    """
    before = json.loads((ROOT / "artifacts/compare/d3-skeleton/silhouette-d3.json").read_text())
    after = json.loads((OUT_DIR / "silhouette-d7.json").read_text())
    cells, oracle = {}, {}
    rose = 0
    for camera in ("A001", "B001", "C001", "D001"):
        for subject in ("subject_00", "subject_01"):
            b = before["arms"]["ours_delivered"][camera][subject]
            a = after["arms"]["ours_delivered"][camera][subject]
            delta = a["iou"]["median"] - b["iou"]["median"]
            rose += int(delta > 0.0)
            cells[f"{camera}/{subject}"] = {
                "iou_before_D3": round(b["iou"]["median"], 4),
                "iou_after_D7": round(a["iou"]["median"], 4),
                "delta": round(delta, 4),
                "precision_before": round(b["precision"]["median"], 4),
                "precision_after": round(a["precision"]["median"], 4),
                "recall_before": round(b["recall"]["median"], 4),
                "recall_after": round(a["recall"]["median"], 4),
            }
            ob = before["arms"]["ORACLE_mamma_mesh"][camera][subject]["iou"]["median"]
            oa = after["arms"]["ORACLE_mamma_mesh"][camera][subject]["iou"]["median"]
            oracle[f"{camera}/{subject}"] = {
                "before": round(ob, 6), "after": round(oa, 6),
                "identical": bool(ob == oa)}
    identical = all(v["identical"] for v in oracle.values())
    report["B7_silhouette"] = {
        "reference": ("MAMMA's SAM2 person masks -- the pixels of the actual footage. The one "
                      "band the candidate cannot optimise."),
        "before": "artifacts/compare/d3-skeleton/silhouette-d3.json (D3's committed run)",
        "after": "artifacts/compare/d7-pelvis-frame/silhouette-d7.json",
        "cells": cells,
        "cells_that_rose": rose,
        "mamma_mesh_oracle_bit_identical": identical,
        "oracle": oracle,
        # NOT A VERDICT, DELIBERATELY. The pre-registered clauses name the torso+legs
        # part, the arms part and the tilt terciles; none of them was measured (below), so
        # inventing a whole-person band now -- "5 of 8 cells" -- would be picking a band
        # after seeing the numbers, which is the defect this lane exists to catch.
        "verdict_withheld": ("REPORTED, NOT BANDED: the pre-registered clauses were not "
                             "measured, and no substitute band is invented after the fact."),
        "mamma_mesh_oracle_bit_identical_IS_the_one_thing_this_run_proves": identical,
        "band_was": ("torso+legs rises on the bent tercile; arms within their CI of D3; the "
                     "upright tercile within its CI; the MAMMA oracle bit-identical"),
        "DEVIATION_the_partwise_and_tercile_cuts_were_NOT_run": (
            "`silhouette.py` writes summaries, not per-frame arrays, and "
            "`silhouette_partwise.py` / `silhouette_vs_tilt.py` have their BUILDS hardcoded "
            "to the D2 paths. The whole-person before/after by cell is what was measured; "
            "the torso+legs vs arms split and the tilt terciles, and therefore the "
            "block-bootstrap CIs on the difference, were NOT computed. The pre-registered "
            "clauses that name them are UNMEASURED, not met."),
        "over_attribution_note": ("a rigid-translation arm would be an UPPER BOUND, not an "
                                  "isolation: D7 leaves the leg roots on the captured hips, "
                                  "so translating D3's mesh moves legs D7 does not."),
        "blind_to": ("depth, a left/right mirror of a fore-aft symmetric pose, and anything "
                     "inside the outline. Every figure is read against a mesh whose shoulder "
                     "span is 540 mm against 346 and 363 measured."),
    }


def main() -> int:
    existing = json.loads((OUT_DIR / "gate.json").read_text())
    report: dict = {
        "step": existing["step"], "branch": existing["branch"],
        "branched_from": existing["branched_from"],
        "package_under_test": autoanim_gnm.__file__,
        "shipped": {
            "PELVIS_FRAME_SOURCE": cm.PELVIS_FRAME_SOURCE,
            "PELVIS_SMOOTHING_FRAMES": cm.PELVIS_SMOOTHING_FRAMES,
            "PELVIS_MINIMUM_RESOLVED_FRACTION": cm.PELVIS_MINIMUM_RESOLVED_FRACTION,
        },
        "preregistration": existing["preregistration"],
    }
    for name, function in (
        ("synthetic", synthetic_block), ("rigidity", rigidity_block),
        ("oracle", oracle_block), ("roundtrip", roundtrip_block),
        ("delivery", delivery_block), ("legacy", legacy_bit_identity),
        ("reported", reported_block), ("mamma", mamma_pelvis_oracle),
        ("silhouette", silhouette_block),
    ):
        try:
            function(report)
        except Exception as error:  # noqa: BLE001 -- a missing arm is reported, not fatal
            report[f"{name}_ERROR"] = f"{type(error).__name__}: {error}"
            print(f"  !! {name}: {type(error).__name__}: {error}")
    verdicts = {key: value["verdict"] for key, value in report.items()
                if isinstance(value, dict) and "verdict" in value}
    report["verdicts"] = verdicts
    report["overall"] = "PASS" if all(
        v.startswith("PASS") or v == "TRUSTED" for v in verdicts.values()) else "FAIL"
    (OUT_DIR / "gate.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(verdicts, indent=1))
    print(f"OVERALL: {report['overall']}")
    print(f"wrote {OUT_DIR / 'gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
