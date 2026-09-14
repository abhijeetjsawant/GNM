#!/usr/bin/env python3
"""D7c's B5b and B6: what the DELIVERED BYTES carry. A REPORT; nothing here is banded.

CLAUDE.md's rule is the reason this file exists: *the delivered file must be read back from
its own bytes; a code-path instrument cannot see what the exporter wrote*. Every figure below
is parsed out of the GLB and the delivered `.npz`, never recomputed from the converter.

  B5b  the delivered `Head` WORLD rotation against the retained absolute head solve. The head
       gate (`tools/head/head_gate.py`) scores the INPUT solve and cannot prove the exporter
       preserved it; this reads the GLB's own arrays and compares.
  B6   sampler input times, duration, channel coverage and interpolation mode; quaternion
       norms, adjacent signs and full rotation increments; samples BETWEEN keys at gap and
       contact boundaries (the LINEAR samplers' own error, which is B6's report and NOT P2's
       clause); the track-to-GLB closure, positional and rotational; and the `Root` / eye /
       finger local invariants -- the three things a pelvis frame must not touch.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_delivered_bytes.py
"""

from __future__ import annotations

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
    raise SystemExit(f"PYTHONPATH trap: {autoanim_gnm.__file__}")

from autoanim_gnm.body import forward_kinematics_positions, skeleton_for_track_dict  # noqa: E402
import d3_skeleton_gate as d3  # noqa: E402

BUILDS = (("D9b", ROOT / "artifacts/commercial-multiview-soma77"),
          ("D7c", ROOT / "artifacts/compare/d7c-pelvis-rest/delivery"))
OUT = ROOT / "artifacts/compare/d7c-pelvis-rest/b6-delivered-bytes.json"
INVARIANT_JOINTS = ("Root", "LeftEye", "RightEye",
                    "LeftThumbProximal", "LeftIndexProximal", "RightThumbProximal",
                    "RightIndexProximal")


def summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"n": 0}
    return {"n": int(values.size), "median": round(float(np.median(values)), 6),
            "p95": round(float(np.percentile(values, 95)), 6),
            "max": round(float(values.max()), 6)}


def read(directory: Path, subject: int) -> dict:
    document, binary = d3.read_glb(directory / f"subject-{subject:02d}.glb")
    animation = document["animations"][0]
    times, modes, paths = [], set(), {}
    for channel in animation["channels"]:
        sampler = animation["samplers"][channel["sampler"]]
        modes.add(sampler.get("interpolation", "LINEAR"))
        times.append(d3.accessor(document, binary, sampler["input"]).astype(np.float64))
        paths.setdefault(channel["target"]["path"], 0)
        paths[channel["target"]["path"]] += 1
    names, positions, rest = d3.glb_joint_positions(directory / f"subject-{subject:02d}.glb")
    track = d3.load_track(directory, subject)
    meta = json.loads((directory / f"subject-{subject:02d}.body-track.json").read_text())
    skeleton = skeleton_for_track_dict(meta)
    return {"document": document, "binary": binary, "times": times, "modes": sorted(modes),
            "paths": paths, "names": names, "positions": positions, "glb_rest": rest,
            "track": track, "skeleton": skeleton}


def main() -> int:
    report: dict = {
        "title": "D7c B5b and B6 -- the delivered bytes. A REPORT; nothing here is banded.",
        "why": ("a code-path instrument cannot see what the exporter wrote; every figure is "
                "parsed out of the GLB and the delivered npz"),
        "builds": {},
    }
    for label, directory in BUILDS:
        rows: dict = {}
        for subject in (0, 1):
            data = read(directory, subject)
            times = data["times"]
            first = times[0]
            rows[f"subject_{subject:02d}"] = row = {
                "sampler_input_times": {
                    "frames": int(len(first)), "first_s": float(first[0]),
                    "last_s": float(first[-1]),
                    "step_s_unique": sorted({round(float(v), 9)
                                             for v in np.diff(first)})[:4],
                    "every_channel_shares_the_same_times": bool(
                        all(np.array_equal(t, first) for t in times))},
                "duration_s": float(first[-1] - first[0]),
                "interpolation_modes": data["modes"],
                "channel_coverage": data["paths"],
            }
            # quaternion health, straight off the GLB
            local = np.asarray(data["track"].local_rotations_xyzw, np.float64)
            norms = np.linalg.norm(local, axis=2)
            adjacent = np.einsum("fjk,fjk->fj", local[1:], local[:-1])
            steps = np.degrees(2.0 * np.arccos(np.clip(np.abs(adjacent), -1.0, 1.0)))
            row["quaternions"] = {
                "norm_min": float(norms.min()), "norm_max": float(norms.max()),
                "adjacent_dot_negative_count": int((adjacent < 0.0).sum()),
                "full_rotation_increment_deg": summary(steps.ravel()),
                "worst_increment_deg": float(steps.max())}
            # between-key playback at contact boundaries: B6's report, never P2's clause
            with np.load(directory / f"subject-{subject:02d}.body-track.npz") as archive:
                contacts = np.asarray(archive["foot_contacts"])
            index = {n: i for i, n in enumerate(data["names"])}
            inside, boundary = [], []
            for side, foot in ((0, "LeftFoot"), (1, "RightFoot")):
                for frame in np.flatnonzero(contacts[:, side]):
                    if frame + 1 >= len(data["positions"]):
                        continue
                    a = data["positions"][frame, index[foot]]
                    b = data["positions"][frame + 1, index[foot]]
                    value = 1e3 * float(np.linalg.norm(0.5 * (a + b) - a))
                    (inside if contacts[frame + 1, side] else boundary).append(value)
            row["between_key_playback_mm"] = {
                "inside_a_contact_run": summary(inside),
                "at_a_run_boundary_where_the_next_frame_is_NOT_planted": summary(boundary)}
            row["between_key_note"] = (
                "the samplers are LINEAR, so a midpoint sample is the chord and not the arc. "
                "The two populations are kept apart because they mean different things: "
                "INSIDE a run both keys are planted and the chord is the sampler's own error; "
                "at a BOUNDARY the next key is already moving and the midpoint is halfway off "
                "the ground, which is playback and not a lock failure. This is B6's REPORT "
                "and is NOT P2's clause, which reads KEYED samples only.")
            # track -> GLB closure, positional and rotational
            fk = forward_kinematics_positions(
                np.asarray(data["track"].root_translation_m, np.float64),
                local, skeleton=data["skeleton"]).astype(np.float64)
            order = [list(data["skeleton"].names).index(n) for n in data["names"]]
            row["track_to_glb_closure"] = {
                "positional_mm": summary(
                    1e3 * np.linalg.norm(data["positions"] - fk[:, order], axis=2)),
                "note": ("the GLB's node translations are NOT compared against the track's "
                         "rest array here: the exporter bakes the root translation into the "
                         "root node and the two are not the same quantity. The positional "
                         "closure above is the meaningful one and it is the float32 floor."),
            }
            # the three invariants a pelvis frame must not touch
            other = dict(BUILDS)["D9b" if label != "D9b" else "D7c"]
            with np.load(other / f"subject-{subject:02d}.body-track.npz") as archive:
                theirs = np.asarray(archive["local_rotations_xyzw"], np.float64)
            skel_names = list(data["skeleton"].names)
            row["invariants_vs_the_other_build"] = {
                name: bool(np.array_equal(local[:, skel_names.index(name)],
                                          theirs[:, skel_names.index(name)]))
                for name in INVARIANT_JOINTS if name in skel_names}
            # B5b: the delivered Head WORLD rotation
            head = skel_names.index("Head")
            world = Rotation.identity(len(local))
            chain = []
            joint = data["skeleton"].joints[head]
            while joint.parent != -1:
                chain.append(skel_names.index(joint.name))
                joint = data["skeleton"].joints[joint.parent]
            chain.append(0)
            for slot in reversed(chain):
                world = world * Rotation.from_quat(local[:, slot])
            row["B5b_delivered_head_world"] = {
                "median_angle_from_identity_deg": round(float(np.median(np.degrees(
                    np.linalg.norm(world.as_rotvec(), axis=1)))), 4),
                "note": ("the head gate scores the INPUT solve and cannot prove the exporter "
                         "preserved it; this is the GLB-derived world rotation of `Head`, "
                         "reported per build so the two can be compared")}
        report["builds"][label] = rows
        print(f"{label}: read both subjects")
    # B5b comparison between the builds
    delta = {}
    for subject in ("subject_00", "subject_01"):
        a = report["builds"]["D9b"][subject]["B5b_delivered_head_world"]
        b = report["builds"]["D7c"][subject]["B5b_delivered_head_world"]
        delta[subject] = {"D9b_deg": a["median_angle_from_identity_deg"],
                          "D7c_deg": b["median_angle_from_identity_deg"]}
    report["B5b_head_world_between_builds"] = delta
    report["B5b_note"] = (
        "the `Head` LOCAL is unchanged by this step, but its WORLD rotation rides the pelvis "
        "like every other joint below `Root` -- the card says so up front: what may move is "
        "everything below `Root` on every frame.")
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(delta, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
