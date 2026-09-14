#!/usr/bin/env python3
"""D7c's B5b and B6: what the DELIVERED BYTES carry. A REPORT; nothing here is banded.

CLAUDE.md's rule is the reason this file exists: *the delivered file must be read back from
its own bytes; a code-path instrument cannot see what the exporter wrote*. Every figure below
is parsed out of the GLB and the delivered `.npz`, never recomputed from the converter.

  B5b  the delivered `Head` WORLD rotation, reconstructed FROM THE GLB'S OWN ROTATION
       CHANNELS, against the retained absolute head solve. An earlier version of this file
       read the body-track JSON and compared rounded medians, which is not a statement about
       the exporter at all; Astra's merge review caught it. What is established is stated
       with the number, and the number is small but NOT zero.
  B6   sampler input times, duration, channel coverage and interpolation mode; quaternion
       norms, adjacent signs and full rotation increments; samples BETWEEN keys at gap and
       contact boundaries -- computed by INTERPOLATING THE QUATERNIONS the LINEAR samplers
       carry and then running forward kinematics, which is what a viewer does; averaging
       already-composed world positions (the earlier version) under-reads it by three orders
       of magnitude. It is B6's report and NOT P2's clause, which reads keyed samples only.
       The track-to-GLB closure, positional AND ROTATIONAL. The GLB's rest, hierarchy and
       INVERSE BIND MATRICES against the sized skeleton -- the comparison the earlier version
       substituted a positional closure for. The `Root` / eye / finger invariants, stated as
       what they are: a TRACK-array claim. And the first MESH-DEFORMATION reading on the
       pelvis / hip / thigh region, skinned from the GLB's own POSITION, JOINTS and WEIGHTS:
       inverted or collapsed triangles, edge-length and area change against the rest. IoU can
       rise while the skin tears, and B1's three rising torso cells are exactly where that
       question bites. The figures go to D6; no band is invented for any of them.

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

from autoanim_gnm.body import (  # noqa: E402
    forward_kinematics_positions, skeleton_for_track_dict, _quaternion_multiply)
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



def glb_channels(path: Path) -> dict:
    """Every animated channel, the hierarchy and the skin, straight out of the GLB.

    `d3.glb_joint_positions` composes world POSITIONS and discards the rotations. B5b needs
    the world ROTATION and the between-key check needs the LOCAL quaternions to interpolate,
    so this returns the arrays themselves.
    """
    document, binary = d3.read_glb(path)
    nodes = document["nodes"]
    joints = document["skins"][0]["joints"]
    parent = {node: -1 for node in joints}
    for position, node in enumerate(nodes):
        for child in node.get("children", ()):
            parent[child] = position
    rotation, translation = {}, {}
    animation = document["animations"][0]
    for channel in animation["channels"]:
        sampler = animation["samplers"][channel["sampler"]]
        data = d3.accessor(document, binary, sampler["output"]).astype(np.float64)
        (rotation if channel["target"]["path"] == "rotation" else translation)[
            channel["target"]["node"]] = data
    rest = {j: np.asarray(nodes[j].get("translation", (0.0, 0.0, 0.0)), np.float64)
            for j in joints}
    inverse_bind = d3.accessor(
        document, binary, document["skins"][0]["inverseBindMatrices"]
    ).astype(np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)   # glTF is column-major
    primitive = document["meshes"][0]["primitives"][0]
    return {
        "document": document, "binary": binary, "nodes": nodes, "joints": joints,
        "parent": parent, "rest": rest, "rotation": rotation, "translation": translation,
        "names": [nodes[j]["name"] for j in joints],
        "inverse_bind": inverse_bind,
        "vertices": d3.accessor(document, binary,
                                primitive["attributes"]["POSITION"]).astype(np.float64),
        "triangles": d3.accessor(document, binary,
                                 primitive["indices"]).astype(np.int64).reshape(-1, 3),
        "skin_joints": np.stack([
            d3.accessor(document, binary, primitive["attributes"][f"JOINTS_{k}"])
            for k in (0, 1)], axis=1).astype(np.int64),
        "skin_weights": np.stack([
            d3.accessor(document, binary, primitive["attributes"][f"WEIGHTS_{k}"])
            for k in (0, 1)], axis=1).astype(np.float64),
    }


def world_from_channels(channels: dict, local_rotations: dict, frame: int | None = None,
                        local_translations: dict | None = None):
    """Forward kinematics on the GLB's own hierarchy, returning world rotations AND origins.

    `local_rotations` maps node -> [4] (one frame) so the between-key check can hand it
    INTERPOLATED quaternions rather than keyed ones.
    """
    order = {node: slot for slot, node in enumerate(channels["joints"])}
    count = len(channels["joints"])
    world_q = np.zeros((count, 4))
    world_p = np.zeros((count, 3))
    for node in channels["joints"]:
        slot = order[node]
        if local_translations is not None and node in local_translations:
            # THE TRANSLATION IS INTERPOLATED TOO. Reading it at the key while the rotations
            # are interpolated freezes the root at its key and turns a sub-millimetre
            # playback error into a millimetre-scale one.
            local_t = np.asarray(local_translations[node], np.float64)
        else:
            local_t = channels["translation"].get(node)
            local_t = (channels["rest"][node] if local_t is None
                       else np.asarray(local_t[frame], np.float64))
        # a node with no rotation CHANNEL keeps its own rest rotation, not the identity
        quaternion = np.asarray(
            local_rotations.get(node, channels["nodes"][node].get(
                "rotation", (0.0, 0.0, 0.0, 1.0))), np.float64)
        if channels["parent"][node] == -1:
            world_p[slot], world_q[slot] = local_t, quaternion
        else:
            up = order[channels["parent"][node]]
            world_p[slot] = world_p[up] + d3._qrot(world_q[up], local_t)
            world_q[slot] = d3._qmul(world_q[up], quaternion)
    return world_q, world_p


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """glTF's own LINEAR rotation sampler: normalised linear interpolation on the shorter arc.

    The glTF 2.0 specification defines LINEAR for rotations as spherical linear interpolation
    with the shorter-arc sign fix; for the increments here (median 0.03 deg) nlerp and slerp
    agree far below the quantity being measured, and the sign fix is the part that matters.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if float(np.dot(a, b)) < 0.0:
        b = -b
    out = (1.0 - t) * a + t * b
    return out / np.linalg.norm(out)


def skinned_vertices(channels: dict, world_q: np.ndarray, world_p: np.ndarray) -> np.ndarray:
    """Linear blend skinning, from the GLB's own POSITION, JOINTS, WEIGHTS and inverse binds.

    This is what a viewer draws. Nothing about AutoAnim is assumed.
    """
    vertices = channels["vertices"]
    out = np.zeros_like(vertices)
    joints = channels["skin_joints"].reshape(len(vertices), -1)
    weights = channels["skin_weights"].reshape(len(vertices), -1)
    for column in range(joints.shape[1]):
        weight = weights[:, column]
        active = weight > 0.0
        if not active.any():
            continue
        slot = joints[active, column]
        inverse = channels["inverse_bind"][slot]
        rest_space = np.einsum("nij,nj->ni", inverse[:, :3, :3], vertices[active]) \
            + inverse[:, :3, 3]
        posed = d3._qrot(world_q[slot], rest_space) + world_p[slot]
        out[active] += weight[active, None] * posed
    return out


def region_triangles(channels: dict, region: tuple[str, ...]) -> np.ndarray:
    """Triangles whose vertices are dominantly weighted to the named joints."""
    names = channels["names"]
    wanted = {names.index(n) for n in region if n in names}
    joints = channels["skin_joints"].reshape(len(channels["vertices"]), -1)
    weights = channels["skin_weights"].reshape(len(channels["vertices"]), -1)
    dominant = joints[np.arange(len(joints)), weights.argmax(axis=1)]
    member = np.isin(dominant, list(wanted))
    return channels["triangles"][member[channels["triangles"]].all(axis=1)]


def rest_world_matrices(channels: dict) -> np.ndarray:
    """The nodes' own TRS composed down the hierarchy, as 4x4 matrices."""
    order = {node: slot for slot, node in enumerate(channels["joints"])}
    out = np.zeros((len(channels["joints"]), 4, 4))
    for node in channels["joints"]:
        slot = order[node]
        local = np.eye(4)
        local[:3, :3] = quaternion_matrix(
            channels["nodes"][node].get("rotation", (0.0, 0.0, 0.0, 1.0)))
        local[:3, 3] = np.asarray(
            channels["nodes"][node].get("translation", (0.0, 0.0, 0.0)), np.float64)
        parent = channels["parent"][node]
        out[slot] = local if parent == -1 else out[order[parent]] @ local
    return out


def quaternion_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def animated_world_matrices(channels: dict, frame: int) -> np.ndarray:
    """The ANIMATED joint transforms at one frame, as a viewer composes them."""
    order = {node: slot for slot, node in enumerate(channels["joints"])}
    out = np.zeros((len(channels["joints"]), 4, 4))
    for node in channels["joints"]:
        slot = order[node]
        local = np.eye(4)
        rotation = channels["rotation"].get(node)
        local[:3, :3] = quaternion_matrix(
            rotation[frame] if rotation is not None
            else channels["nodes"][node].get("rotation", (0.0, 0.0, 0.0, 1.0)))
        translation = channels["translation"].get(node)
        local[:3, 3] = (np.asarray(translation[frame], np.float64)
                        if translation is not None else channels["rest"][node])
        parent = channels["parent"][node]
        out[slot] = local if parent == -1 else out[order[parent]] @ local
    return out


def skin(channels: dict, world: np.ndarray) -> np.ndarray:
    """Linear blend skinning: `sum_j w_j * (world_j @ inverseBind_j) * v`. What a viewer draws."""
    vertices = channels["vertices"]
    joints = channels["skin_joints"].reshape(len(vertices), -1)
    weights = channels["skin_weights"].reshape(len(vertices), -1)
    out = np.zeros_like(vertices)
    for column in range(joints.shape[1]):
        weight = weights[:, column]
        active = weight > 0.0
        if not active.any():
            continue
        slot = joints[active, column]
        matrix = world[slot] @ channels["inverse_bind"][slot]
        out[active] += weight[active, None] * (
            np.einsum("nij,nj->ni", matrix[:, :3, :3], vertices[active]) + matrix[:, :3, 3])
    return out


def dominant_rotation(channels: dict, frame: int, triangles: np.ndarray) -> np.ndarray:
    """Each triangle's own rigid motion: its dominant joint's skinning rotation at `frame`.

    A triangle's rest normal carried through this is where the normal SHOULD point; the
    triangle is inverted when its posed normal opposes it. Using the joint's transform rather
    than a fit to the triangle's own three points is what makes the test well posed -- three
    coplanar points cannot determine an out-of-plane sign.
    """
    world = animated_world_matrices(channels, frame)
    joints = channels["skin_joints"].reshape(len(channels["vertices"]), -1)
    weights = channels["skin_weights"].reshape(len(channels["vertices"]), -1)
    dominant = joints[np.arange(len(joints)), weights.argmax(axis=1)]
    slot = dominant[triangles[:, 0]]
    matrix = world[slot] @ channels["inverse_bind"][slot]
    return matrix[:, :3, :3]


def mesh_deformation(channels: dict, frames: list[int], region: tuple[str, ...]) -> dict:
    """The first mesh-deformation reading on the pelvis / hip / thigh region.

    REPORT ONLY and no band is invented. The reference is the mesh's OWN BIND POSE -- the
    `POSITION` attribute -- because that is the shape the inverse binds are defined against;
    an earlier draft used an identity-rotation pose, which is not the bind pose of this file
    (the nodes carry rest rotations) and reported half the region as inverted.

    Three quantities, because IoU can rise while the skin tears:
      * the area ratio against the bind pose, and its SPREAD -- how much the region stretches;
      * the edge-length ratio, whose collapse toward zero is a pinched seam;
      * triangles whose normal FLIPS relative to the bind pose -- an inverted triangle.
    """
    triangles = region_triangles(channels, region)
    if not len(triangles):
        return {"triangles": 0}

    def measure(vertices):
        a, b, c = (vertices[triangles[:, k]] for k in (0, 1, 2))
        normal = np.cross(b - a, c - a)
        area = 0.5 * np.linalg.norm(normal, axis=1)
        edges = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                          np.linalg.norm(a - c, axis=1)], axis=1)
        return area, edges, normal

    rest = channels["vertices"]
    rest_area, rest_edges, rest_normal = measure(rest)
    good = rest_area > 1e-12
    area_ratio, edge_ratio, inverted = [], [], []
    for frame in frames:
        posed = skin(channels, animated_world_matrices(channels, frame))
        area, edges, normal = measure(posed)
        area_ratio.append(area[good] / rest_area[good])
        edge_ratio.append((edges[good] / np.maximum(rest_edges[good], 1e-12)).ravel())
        # THE INVERSION TEST IS FRAME-RELATIVE, and this is its SECOND repair. Dotting the
        # posed normal against the triangle's BIND-SPACE normal is wrong -- a harmless RIGID
        # 180 deg rotation trips it, and Astra reproduced that false positive. Fitting a
        # rotation to the triangle's own three points is ALSO wrong, and quietly so: three
        # coplanar points give a rank-2 covariance, the third singular vector is arbitrary,
        # and the determinant's sign is then a coin toss -- it read 1242 of 2424 triangles
        # "inverted", i.e. exactly noise. What carries a triangle's rigid motion is the
        # SKINNING: the rest normal is rotated by the triangle's own dominant joint
        # transform, and the triangle is inverted when its posed normal opposes THAT.
        expected = np.einsum("tij,tj->ti", dominant_rotation(channels, frame, triangles),
                             rest_normal)
        inverted.append(int((np.einsum("ij,ij->i", normal[good],
                                       expected[good]) < 0.0).sum()))
    area_ratio = np.concatenate(area_ratio)
    edge_ratio = np.concatenate(edge_ratio)
    return {
        "triangles": int(len(triangles)), "frames_measured": len(frames),
        "region": list(region), "reference": "the mesh's own BIND POSE (POSITION)",
        "area_ratio": {"median": round(float(np.median(area_ratio)), 5),
                       "p5": round(float(np.percentile(area_ratio, 5)), 5),
                       "p95": round(float(np.percentile(area_ratio, 95)), 5),
                       "min": round(float(area_ratio.min()), 5),
                       "max": round(float(area_ratio.max()), 5)},
        "edge_length_ratio": {"median": round(float(np.median(edge_ratio)), 5),
                              "p5": round(float(np.percentile(edge_ratio, 5)), 5),
                              "p95": round(float(np.percentile(edge_ratio, 95)), 5),
                              "min": round(float(edge_ratio.min()), 5)},
        "inverted_triangles_frame_relative": {
            "per_frame_max": int(max(inverted)), "total": int(sum(inverted)),
            "test": ("the rest normal carried through the triangle's own dominant-joint "
                     "skinning rotation; inverted when the posed normal opposes THAT. A "
                     "rigid rotation of a limb cannot trip it, and unlike a Kabsch fit on "
                     "three coplanar points it is not rank-deficient.")},
        "collapsed_triangles_area_below_1e-4_of_bind": int((area_ratio < 1e-4).sum()),
        "reader_is_sound": (
            "Astra's merge review compared this reader's animated vertices against the "
            "retained BLENDER meshes on five frames per performer per build: maximum "
            "discrepancy 0.00518 mm. These figures are the mesh's, not the reader's."),
        "note": ("REPORT, handed to D6. NO DEFORMATION ACCEPTANCE BAND is invented and "
                 "nothing in the merge predicate reads any of it."),
    }


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
            channels = glb_channels(directory / f"subject-{subject:02d}.glb")
            # NOTE the two orders. `glb_local` is in the GLB's SKIN-JOINT order; `local`
            # below is the track's own array in SKELETON order, and `forward_kinematics_
            # positions` must be handed the latter. Conflating them silently produced a
            # 2351 mm "closure" in an earlier draft of this file.
            glb_local = np.stack([channels["rotation"][node]
                                  for node in channels["joints"]], axis=1)
            local = glb_local
            norms = np.linalg.norm(local, axis=2)
            unit = local / norms[..., None]
            adjacent = np.einsum("fjk,fjk->fj", unit[1:], unit[:-1])
            steps = np.degrees(2.0 * np.arccos(np.clip(np.abs(adjacent), -1.0, 1.0)))
            raw = np.einsum("fjk,fjk->fj", local[1:], local[:-1])
            raw_steps = np.degrees(2.0 * np.arccos(np.clip(np.abs(raw), -1.0, 1.0)))
            row["quaternions"] = {
                "read_from": "the GLB's own rotation channels",
                "norm_min": float(norms.min()), "norm_max": float(norms.max()),
                "adjacent_dot_negative_count": int((adjacent < 0.0).sum()),
                "full_rotation_increment_deg_NORMALISED": summary(steps.ravel()),
                "full_rotation_increment_deg_as_stored": summary(raw_steps.ravel()),
                "worst_increment_deg": float(steps.max()),
                "note": ("the increment is taken on NORMALISED quaternions. Taken on the "
                         "stored ones it also carries the float32 norm error, which is why "
                         "an earlier version of this file reported a nonzero median where "
                         "the true median increment on most joints is exactly zero.")}
            # between-key playback at contact boundaries: B6's report, never P2's clause
            with np.load(directory / f"subject-{subject:02d}.body-track.npz") as archive:
                contacts = np.asarray(archive["foot_contacts"])
            index = {n: i for i, n in enumerate(data["names"])}
            inside, boundary, gaps = [], [], []
            feet = {name: channels["names"].index(name) for name in
                    ("LeftFoot", "LeftToes", "RightFoot", "RightToes")}
            count = len(channels["rotation"][channels["joints"][0]])

            def midpoint_error(frame: int, joints: tuple[str, ...]) -> dict:
                """WHAT A VIEWER DRAWS halfway between two keys.

                BOTH channels are interpolated -- the rotations on the shorter arc and the
                translations linearly, which is what a LINEAR sampler does -- and forward
                kinematics is re-run. Interpolating only the rotations freezes the root at
                its key and inflates this by an order of magnitude; Astra's round 2 caught
                exactly that, and it is the second time this row has been wrong.
                """
                half_q = {node: slerp(channels["rotation"][node][frame],
                                      channels["rotation"][node][frame + 1], 0.5)
                          for node in channels["joints"]}
                half_t = {node: 0.5 * (np.asarray(array[frame], np.float64)
                                       + np.asarray(array[frame + 1], np.float64))
                          for node, array in channels["translation"].items()}
                _, mid_p = world_from_channels(channels, half_q, frame=frame,
                                               local_translations=half_t)
                _, key_a = world_from_channels(
                    channels, {n: channels["rotation"][n][frame]
                               for n in channels["joints"]}, frame=frame)
                _, key_b = world_from_channels(
                    channels, {n: channels["rotation"][n][frame + 1]
                               for n in channels["joints"]}, frame=frame + 1)
                return {joint: 1e3 * float(np.linalg.norm(
                    mid_p[feet[joint]]
                    - 0.5 * (key_a[feet[joint]] + key_b[feet[joint]])))
                    for joint in joints}

            for side, (foot, toes) in enumerate((("LeftFoot", "LeftToes"),
                                                 ("RightFoot", "RightToes"))):
                for frame in np.flatnonzero(contacts[:, side]):
                    if frame + 1 >= count:
                        continue
                    values = midpoint_error(int(frame), (foot, toes))
                    target = inside if contacts[frame + 1, side] else boundary
                    for joint, value in values.items():
                        target.append({"joint": joint, "mm": value})
            demoted = json.loads((directory.parent / "delivery-build.json").read_text()
                                 )["diagnostics"]["pelvis_frame"][subject][
                                     "lever_guard"]["demoted_frames"] \
                if (directory.parent / "delivery-build.json").exists() else []
            for frame in demoted:
                if frame + 1 < count:
                    gaps.append(midpoint_error(int(frame), ("LeftFoot",))["LeftFoot"])

            def by_joint(rows):
                return {name: summary([r["mm"] for r in rows if r["joint"] == name])
                        for name in ("LeftFoot", "LeftToes", "RightFoot", "RightToes")}

            row["between_key_playback_mm"] = {
                "method": ("BOTH channels interpolated -- rotations on the shorter arc, "
                           "translations linearly -- forward kinematics re-run, and compared "
                           "against the chord between the two keys' own positions"),
                "inside_a_contact_run": by_joint(inside),
                "at_a_run_boundary_where_the_next_frame_is_NOT_planted": by_joint(boundary),
                "at_a_GAP_boundary_LeftFoot_mm": summary(gaps),
                "maxima_mm": {name: round(max([r["mm"] for r in inside + boundary
                                               if r["joint"] == name], default=0.0), 6)
                              for name in ("LeftFoot", "LeftToes", "RightFoot",
                                           "RightToes")},
            }
            row["between_key_note"] = (
                "B6's REPORT and NOT P2's clause, which reads KEYED samples only. TWO earlier "
                "versions of this row were wrong and both are withdrawn: the first averaged "
                "already-composed world positions (0.0003 mm, three orders too small); the "
                "second interpolated the rotations but read the TRANSLATION AT THE KEY, "
                "freezing the root and inflating it to millimetres.")
            # track -> GLB closure, positional and rotational
            fk = forward_kinematics_positions(
                np.asarray(data["track"].root_translation_m, np.float64),
                np.asarray(data["track"].local_rotations_xyzw, np.float64),
                skeleton=data["skeleton"]).astype(np.float64)
            order = [list(data["skeleton"].names).index(n) for n in data["names"]]
            # ROTATIONAL CLOSURE, FRAME-CORRECTED. The raw comparison of the GLB's channel
            # against the track's local is ~32 deg and is NOT an error: the exporter builds
            # `animated_world[j] = track_world[j] * alignment[j] * rest_world[j]`
            # (`body_export.py:379`), so the GLB's world rotation differs from the track's by
            # a CONSTANT per-joint frame. Undo that constant -- estimate it on one frame, then
            # measure how CONSTANT it is across every other frame -- and the closure appears.
            track_local = np.asarray(data["track"].local_rotations_xyzw, np.float64)
            skel_names = list(data["skeleton"].names)
            track_world = np.zeros_like(track_local)
            for slot, joint in enumerate(data["skeleton"].joints):
                track_world[:, slot] = (
                    track_local[:, slot] if joint.parent == -1
                    else _quaternion_multiply(track_world[:, joint.parent],
                                              track_local[:, slot]))
            frames_count = track_local.shape[0]
            glb_world = np.zeros((frames_count, len(channels["joints"]), 4))
            for frame in range(frames_count):
                keyed = {node: channels["rotation"][node][frame]
                         for node in channels["joints"]}
                glb_world[frame], _ = world_from_channels(channels, keyed, frame=frame)
            residuals = []
            for slot, name in enumerate(channels["names"]):
                ours = Rotation.from_quat(track_world[:, skel_names.index(name)])
                theirs = Rotation.from_quat(glb_world[:, slot])
                constant = ours[0].inv() * theirs[0]
                residuals.append(np.degrees(np.linalg.norm(
                    ((ours * constant) * theirs.inv()).as_rotvec(), axis=1)))
            rotational = np.concatenate(residuals)
            skel_names = list(data["skeleton"].names)
            # THE GLB'S REST, HIERARCHY AND INVERSE BIND MATRICES against the sized
            # skeleton. The COMPONENTS of a node translation cannot be compared against the
            # skeleton's rest translation: every joint node also carries a rest ROTATION
            # (55 of 55 do), so a node's translation is expressed in its parent's BIND frame
            # and not in the rig's canonical axes. What is comparable is the LENGTH, which is
            # the bone, and that is what the D3 gate compares too.
            hierarchy_ok, lengths = True, []
            for slot, name in enumerate(channels["names"]):
                joint = data["skeleton"].joints[skel_names.index(name)]
                node = channels["joints"][slot]
                parent_node = channels["parent"][node]
                expected = (None if joint.parent == -1
                            else data["skeleton"].joints[joint.parent].name)
                actual = (None if parent_node == -1
                          else channels["nodes"][parent_node].get("name"))
                hierarchy_ok &= (expected == actual)
                if name != "Root":
                    lengths.append(1e3 * abs(
                        float(np.linalg.norm(channels["rest"][node]))
                        - float(np.linalg.norm(
                            np.asarray(joint.rest_translation_m, np.float64)))))
            # THE BIND POSE. Linear blend skinning at the NODE rest pose must return the
            # POSITION attribute if the node TRS is the bind pose. It does not.
            rest_world = rest_world_matrices(channels)
            bind_residual = float(np.abs(
                np.einsum("nij,njk->nik", channels["inverse_bind"], rest_world)
                - np.eye(4)).max())
            at_rest = skin(channels, rest_world)
            row["glb_rest_hierarchy_and_inverse_binds_vs_the_sized_skeleton"] = {
                "hierarchy_matches_joint_for_joint": bool(hierarchy_ok),
                "bone_length_error_mm": summary(lengths),
                "why_components_are_not_compared": (
                    "every joint node carries a rest ROTATION (55 of 55), so a node "
                    "translation is expressed in its parent's BIND frame, not in the rig's "
                    "canonical axes. The length is the bone and is what is comparable."),
                "inverse_bind_times_node_rest_world_minus_identity_max": round(
                    bind_residual, 6),
                "skinned_at_the_node_rest_pose_vs_the_POSITION_attribute_mm": summary(
                    1e3 * np.linalg.norm(at_rest - channels["vertices"], axis=1)),
                "VERDICT": "NOT ESTABLISHED -- and the reason is stated rather than hidden",
                "what_it_actually_shows": (
                    "`body_export.py:599` writes `animated_rotations[0, index]` as each "
                    "node's default rotation -- THE FIRST ANIMATED POSE, not the bind pose. "
                    "So skinning under the node defaults and comparing with `POSITION` was "
                    "never a test of the exporter or of this reader: it measures how far the "
                    "take's FIRST FRAME is from the asset's bind pose, which is a property "
                    "of the MOTION. The two builds differ there because their first frames "
                    "differ -- 594.005 -> 590.159 mm on performer 0 and 158.929 -> 156.052 "
                    "mm on performer 1, NOT identical as an earlier version said. A viewer "
                    "that disables the animation draws the take's first pose: correct "
                    "behaviour, not a defect. An earlier version called the mesh reading NOT "
                    "ESTABLISHED and suspected this reader; Astra's merge review settled it "
                    "against the retained Blender meshes at 0.00518 mm, so the reader is "
                    "sound and the reading is FINISHED rather than handed over."),
            }
            row["track_to_glb_closure"] = {
                "positional_mm": summary(
                    1e3 * np.linalg.norm(data["positions"] - fk[:, order], axis=2)),
                "rotational_deg_frame_corrected": summary(rotational),
                "rotational_samples": int(rotational.size),
                "rotational_note": (
                    "the exporter's documented per-joint constant frame change "
                    "(`body_export.py:379`) is undone first; this residual is how CONSTANT it "
                    "is over every joint-frame sample. The RAW comparison is ~32 deg and is "
                    "that change of frame, not an export error -- an earlier version of this "
                    "file reported the raw figure and refused to call it a closure, which was "
                    "the right caution and the wrong measurement."),
                "note": "the positional row is the float32 floor of the export, not a fit",
            }
            # the three invariants a pelvis frame must not touch
            other = dict(BUILDS)["D9b" if label != "D9b" else "D7c"]
            with np.load(other / f"subject-{subject:02d}.body-track.npz") as archive:
                theirs = np.asarray(archive["local_rotations_xyzw"], np.float64)
            row["invariants_vs_the_other_build_TRACK_ARRAYS"] = {
                name: bool(np.array_equal(
                    np.asarray(data["track"].local_rotations_xyzw,
                               np.float64)[:, skel_names.index(name)],
                    theirs[:, skel_names.index(name)]))
                for name in INVARIANT_JOINTS if name in skel_names}
            row["invariants_note"] = (
                "a TRACK-ARRAY claim: these are the delivered `.npz` local rotations, not "
                "every GLB proximal channel. The GLB carries one rotation channel per joint "
                "and the finger channels are written from these same arrays, but the claim "
                "as measured is about the track.")
            # B5b: the delivered Head WORLD rotation, FROM THE GLB'S OWN CHANNELS.
            head_slot = channels["names"].index("Head")
            head_world = np.zeros((local.shape[0], 4))
            for frame in range(local.shape[0]):
                keyed = {node: channels["rotation"][node][frame]
                         for node in channels["joints"]}
                world_q, _ = world_from_channels(channels, keyed, frame=frame)
                head_world[frame] = world_q[head_slot]
            row["B5b_delivered_head_world_quat"] = head_world.tolist()
            row["B5b_delivered_head_world"] = {
                "median_angle_from_identity_deg": round(float(np.median(np.degrees(
                    2.0 * np.arccos(np.clip(np.abs(head_world[:, 3]), -1.0, 1.0))))), 6),
                "read_from": "the GLB's own rotation channels, composed down the hierarchy",
                "note": ("an earlier version of this file read the body-track JSON and "
                         "compared rounded medians, which says nothing about the exporter. "
                         "The between-build comparison below is on the per-frame world "
                         "rotations themselves.")}
            row["mesh_deformation_pelvis_hip_thigh"] = mesh_deformation(
                channels, list(range(0, local.shape[0], 10)),
                ("Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg"))
        report["builds"][label] = rows
        print(f"{label}: read both subjects")
    # B5b comparison between the builds
    delta = {}
    for subject in ("subject_00", "subject_01"):
        a = np.asarray(report["builds"]["D9b"][subject].pop("B5b_delivered_head_world_quat"))
        b = np.asarray(report["builds"]["D7c"][subject].pop("B5b_delivered_head_world_quat"))
        dot = np.abs(np.einsum("fk,fk->f", a / np.linalg.norm(a, axis=1)[:, None],
                               b / np.linalg.norm(b, axis=1)[:, None]))
        angle = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
        delta[subject] = {
            "per_frame_difference_deg": summary(angle),
            "D9b_median_from_identity_deg": report["builds"]["D9b"][subject][
                "B5b_delivered_head_world"]["median_angle_from_identity_deg"],
            "D7c_median_from_identity_deg": report["builds"]["D7c"][subject][
                "B5b_delivered_head_world"]["median_angle_from_identity_deg"]}
    report["B5b_head_world_between_builds"] = delta
    report["B5b_what_is_established"] = (
        "NOT that the two are identical. The per-frame difference between the builds' "
        "delivered `Head` WORLD rotations is of the order of 1e-5 degrees -- far below "
        "anything that matters and consistent with float32 through a different chain, but "
        "NOT zero, and an earlier version of this file claimed identity from rounded medians "
        "read out of the body-track JSON. What IS established is that the converter places "
        "the head-on-torso rotation on `Head` as an ABSOLUTE target, so a ~9 degree pelvis "
        "change propagates to the head at the 1e-5 degree level rather than at 9 degrees, "
        "and the exporter carried that through to the delivered bytes.")
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(delta, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
