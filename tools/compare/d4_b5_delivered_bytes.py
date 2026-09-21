"""D4 B5 (REPORTED): the delivered bytes.

Everything here is read out of the delivered GLB itself and compared against MHR's own model,
dumped separately by `tools/fitter/mhr_delivery.py --dump-reference` (momentum lives on a
different interpreter). "The delivered file must be read back from its own bytes; a code-path
instrument cannot see what the exporter wrote" (CLAUDE.md) -- that lesson cost D3 two skeletons
81-195 mm apart, and at stage 2 of THIS step it caught a GLB that imported cleanly at the right
frame range and carried a body up to 0.76 m out of place.

Checked:
  * sampler times: one animation, 150 samples, exactly k/30 s, strictly increasing;
  * the skin: joint names, the parent hierarchy and the rest node translations against MHR's own,
    and the per-vertex skin indices and weights against MHR's `skin_weights`;
  * the mesh: vertex count constant over frames and every coordinate finite (read from the
    Blender-exported posed mesh, i.e. the file as a renderer sees it);
  * facing: the delivered eye midpoint must sit on the same side of the head joint as the
    CAPTURED eyes sit of the captured `nose` -- dot > 0, per frame. `nose` is SOMA-77 index 6,
    the Head skeletal JOINT inside the skull (CLAUDE.md), so both sides of this dot are
    head-joint-to-eye vectors and the comparison needs no external facing convention.

    .venv/bin/python tools/compare/d4_b5_delivered_bytes.py --delivery DIR \
        --reference artifacts/compare/d4-body/mhr-reference-lod2.npz \
        --mesh artifacts/compare/d4-body/work-delivery/delivered-mesh.npz --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/compare"))
# allow_pickle below: this repository's own build output under the gitignored artifacts/ tree.
import d3_skeleton_gate as d3  # noqa: E402
from d4_glb_closure import glb_joint_positions, to_capture  # noqa: E402

FPS = 30.0
FRAMES = 150


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    delivery = arguments.delivery.resolve()
    reference = np.load(arguments.reference, allow_pickle=True)
    mesh_npz = np.load(arguments.mesh, allow_pickle=True)
    report = {"delivery": str(delivery), "reference": str(arguments.reference), "subjects": {}}
    for subject in (0, 1):
        document, binary = d3.read_glb(delivery / f"subject-{subject:02d}.glb")
        track = np.load(delivery / f"subject-{subject:02d}.body-track.npz", allow_pickle=True)
        declared = json.loads((delivery / f"subject-{subject:02d}.body-track.json")
                              .read_text(encoding="utf-8"))["landmark_to_joint"]
        names_capture = [str(n) for n in track["consumed_joint_names"]]
        capture = np.asarray(track["triangulated_world_positions_z_up_m"], np.float64)

        animations = document["animations"]
        times = sorted({tuple(np.round(d3.accessor(document, binary, s["input"]).astype(np.float64),
                                       9)) for s in animations[0]["samplers"]})
        sample = np.asarray(times[0])
        expected = np.arange(FRAMES) / FPS

        nodes = document["nodes"]
        joints = document["skins"][0]["joints"]
        glb_names = [nodes[j]["name"] for j in joints]
        order = {node: i for i, node in enumerate(joints)}
        parent = {i: -1 for i in range(len(nodes))}
        for index, node in enumerate(nodes):
            for child in node.get("children", ()):
                parent[child] = index
        glb_parents = [order.get(parent[j], -1) for j in joints]
        glb_rest = np.asarray([nodes[j].get("translation", (0.0, 0.0, 0.0)) for j in joints],
                              np.float64)

        primitive = document["meshes"][0]["primitives"][0]
        glb_joint_idx = d3.accessor(document, binary, primitive["attributes"]["JOINTS_0"])
        glb_weights = d3.accessor(document, binary, primitive["attributes"]["WEIGHTS_0"])
        ref_index = np.asarray(reference["skin_index"])
        ref_weight = np.asarray(reference["skin_weight"])
        # glTF carries four influences per set; MHR keeps up to eight. Compare the sets.
        influences = glb_joint_idx.shape[1] if glb_joint_idx.ndim > 1 else 1
        dense_glb = np.zeros((glb_joint_idx.shape[0], len(joints)))
        for column in range(influences):
            np.add.at(dense_glb, (np.arange(dense_glb.shape[0]), glb_joint_idx[:, column]),
                      glb_weights[:, column])
        dense_ref = np.zeros((ref_index.shape[0], len(joints)))
        for column in range(ref_index.shape[1]):
            np.add.at(dense_ref, (np.arange(dense_ref.shape[0]), ref_index[:, column]),
                      ref_weight[:, column])

        verts = np.asarray(mesh_npz[f"verts_{subject:02d}"])
        joint_names, positions = glb_joint_positions(delivery / f"subject-{subject:02d}.glb")
        world = to_capture(positions)
        frames = min(world.shape[0], capture.shape[0])
        eye_mid = 0.5 * (world[:frames, joint_names.index(declared["left_eye"])]
                         + world[:frames, joint_names.index(declared["right_eye"])])
        head = world[:frames, joint_names.index(declared["nose"])]
        capture_eye = 0.5 * (capture[:frames, names_capture.index("left_eye")]
                             + capture[:frames, names_capture.index("right_eye")])
        capture_head = capture[:frames, names_capture.index("nose")]
        ours_axis = eye_mid - head
        capture_axis = capture_eye - capture_head
        dot = np.einsum("ij,ij->i", ours_axis, capture_axis)
        keep = np.isfinite(dot)
        norms = np.linalg.norm(ours_axis, axis=1) * np.linalg.norm(capture_axis, axis=1)
        angle = np.degrees(np.arccos(np.clip(dot / np.where(norms > 0, norms, np.nan), -1, 1)))

        checks = {
            "animations": len(animations),
            "sampler_time_sets": len(times),
            "sampler_samples": int(sample.size),
            "sampler_times_are_k_over_30_s": bool(sample.size == FRAMES
                                                  and np.allclose(sample, expected, atol=1e-6)),
            "sampler_times_strictly_increasing": bool(np.all(np.diff(sample) > 0)),
            "joint_names_match_MHR": glb_names == [str(n) for n in reference["joint_names"]],
            "hierarchy_matches_MHR": bool(np.array_equal(
                np.asarray(glb_parents), np.asarray(reference["joint_parents"]))),
            # the GLB is in metres (momentum writes the cm->m conversion as a root scale) and
            # MHR's offsets are in centimetres; the delivered REST is MHR's mean-body rest, with
            # the fitted identity carried by the animation channels, not baked into the nodes.
            "rest_translation_max_abs_diff_cm": float(np.abs(
                glb_rest * 100.0 - np.asarray(reference["joint_offsets"], np.float64)).max()),
            "mesh_vertices": int(verts.shape[1]),
            "mesh_vertices_match_MHR": int(verts.shape[1]) == int(reference["mesh_vertex_count"]),
            "mesh_vertex_count_constant_over_frames": bool(verts.ndim == 3),
            "mesh_all_finite": bool(np.isfinite(verts).all()),
            "mesh_frames": int(verts.shape[0]),
            "skin_weight_max_abs_diff": float(np.abs(dense_glb - dense_ref).max()),
            "skin_weights_sum_to_one_max_error": float(np.abs(dense_glb.sum(axis=1) - 1.0).max()),
            "facing_dot_positive_frames": int(np.count_nonzero(dot[keep] > 0)),
            "facing_dot_frames_scored": int(keep.sum()),
            "facing_dot_min": float(dot[keep].min()),
            "facing_angle_deg_median": round(float(np.nanmedian(angle)), 2),
            "facing_angle_deg_p95": round(float(np.nanpercentile(angle, 95)), 2),
            "facing_frames_over_90_deg": [int(f) for f in np.where(
                np.isfinite(angle) & (angle > 90.0))[0]],
            "facing_is_a_REPORTED_clause": True,
        }
        # The byte clauses and the facing clause are kept apart deliberately: the facing dot is
        # scored against the CAPTURED eyes and nose, which are detector output on that frame, so a
        # miss there is a statement about the reference as much as about the file.
        checks["facing_dot_positive_on_every_frame"] = (
            checks["facing_dot_positive_frames"] == checks["facing_dot_frames_scored"])
        checks["verdict"] = "PASS" if (
            checks["sampler_times_are_k_over_30_s"]
            and checks["sampler_times_strictly_increasing"]
            and checks["joint_names_match_MHR"] and checks["hierarchy_matches_MHR"]
            and checks["rest_translation_max_abs_diff_cm"] < 1e-4
            and checks["mesh_vertices_match_MHR"] and checks["mesh_all_finite"]
            and checks["mesh_frames"] == FRAMES
            and checks["skin_weight_max_abs_diff"] < 1e-4
        ) else "FAIL"
        report["subjects"][f"subject_{subject:02d}"] = checks
        print(f"subject {subject:02d}: {checks['verdict']}  times {checks['sampler_samples']}@"
              f"{FPS:g}fps, rest delta {checks['rest_translation_max_abs_diff_cm']:.2e} cm, "
              f"skin delta {checks['skin_weight_max_abs_diff']:.2e}; facing (reported) "
              f"{checks['facing_dot_positive_frames']}/{checks['facing_dot_frames_scored']} "
              f"frames, median angle {checks['facing_angle_deg_median']} deg")
    report["verdict_bytes"] = ("PASS" if all(row["verdict"] == "PASS"
                                             for row in report["subjects"].values()) else "FAIL")
    report["facing_positive_on_every_frame_both_performers"] = all(
        row["facing_dot_positive_on_every_frame"] for row in report["subjects"].values())
    Path(arguments.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("B5 bytes:", report["verdict_bytes"], "| facing positive on every frame:",
          report["facing_positive_on_every_frame_both_performers"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
