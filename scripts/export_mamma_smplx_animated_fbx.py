#!/usr/bin/env python3
"""Export native MAMMA SMPL-X skeletal motion as an animated FBX.

The exporter retains the original locked-head SMPL-X topology, fitted betas,
55-joint hierarchy, skinning weights, all 30 finger joints, jaw and eye joints.
Per-frame SMPL-X pose correctives are baked as morph targets so the FBX surface
matches MAMMA's saved vertices at the 30 Hz samples rather than degrading to
plain linear-blend skinning.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_mamma_smplx_neutral_fbx import (  # noqa: E402
    _build_character,
    _clear_scene,
    _load_sources,
)


def _arguments() -> tuple[Path, Path, Path, Path, Path, Path]:
    try:
        separator = sys.argv.index("--")
    except ValueError as exc:
        raise SystemExit(
            "Expected MODEL PARAMS VERTICES SKELETON OUTPUT.fbx REPORT.json after '--'"
        ) from exc
    values = sys.argv[separator + 1 :]
    if len(values) != 6:
        raise SystemExit(
            "Expected MODEL PARAMS VERTICES SKELETON OUTPUT.fbx REPORT.json after '--'"
        )
    model, params, vertices, skeleton, output, report = (
        Path(value).expanduser().resolve() for value in values
    )
    for source in (model, params, vertices, skeleton):
        if not source.is_file():
            raise SystemExit(f"Required source file is missing: {source}")
    if output.suffix.lower() != ".fbx":
        raise SystemExit(f"Output must use the .fbx extension: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return model, params, vertices, skeleton, output, report


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rodrigues(axis_angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(axis_angles, dtype=np.float64)
    theta = np.linalg.norm(angles, axis=-1, keepdims=True)
    axis = np.divide(
        angles, theta, out=np.zeros_like(angles), where=theta > 1e-12
    )
    x, y, z = np.moveaxis(axis, -1, 0)
    zero = np.zeros_like(x)
    skew = np.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], axis=-1
    ).reshape(angles.shape[:-1] + (3, 3))
    identity = np.broadcast_to(np.eye(3, dtype=np.float64), skew.shape)
    expanded_theta = theta[..., None]
    return (
        identity
        + np.sin(expanded_theta) * skew
        + (1.0 - np.cos(expanded_theta)) * (skew @ skew)
    )


def _motion(
    model_path: Path, params_path: Path, vertices_path: Path, data: dict[str, object]
) -> dict[str, np.ndarray]:
    with np.load(model_path, allow_pickle=True) as model:
        pose_directions = np.asarray(model["posedirs"], dtype=np.float64)
        left_hand_mean = np.asarray(model["hands_meanl"], dtype=np.float64).reshape(15, 3)
        right_hand_mean = np.asarray(model["hands_meanr"], dtype=np.float64).reshape(15, 3)
    with np.load(params_path, allow_pickle=True) as params:
        pose = np.asarray(params["smplx_pose"], dtype=np.float64)
        translation = np.asarray(params["smplx_translation"], dtype=np.float64)
    with np.load(vertices_path, allow_pickle=False) as vertices:
        expected_vertices = np.asarray(vertices["pred_vertices"], dtype=np.float64)
        expected_joints = np.asarray(vertices["pred_joints"], dtype=np.float64)[:, :55]
    frames = len(pose)
    if pose.shape != (frames, 165) or translation.shape != (frames, 3):
        raise RuntimeError("MAMMA motion arrays have invalid shapes")
    if expected_vertices.shape != (frames, 10475, 3):
        raise RuntimeError("MAMMA baked vertices differ from locked-head SMPL-X")

    effective_pose = pose.reshape(frames, 55, 3).copy()
    # This MAMMA run constructs the neutral model with flat_hand_mean=False;
    # SMPL-X therefore adds its native MANO means before Rodrigues conversion.
    effective_pose[:, 25:40] += left_hand_mean[None, ...]
    effective_pose[:, 40:55] += right_hand_mean[None, ...]
    local_rotations = _rodrigues(effective_pose)
    joints = np.asarray(data["joints"], dtype=np.float64)
    parents = np.asarray(data["parents"], dtype=np.int64)
    global_rotations = np.empty_like(local_rotations)
    global_positions = np.empty((frames, 55, 3), dtype=np.float64)
    for frame in range(frames):
        for joint in range(55):
            parent = int(parents[joint])
            if parent < 0:
                global_rotations[frame, joint] = local_rotations[frame, joint]
                global_positions[frame, joint] = joints[joint]
            else:
                global_rotations[frame, joint] = (
                    global_rotations[frame, parent] @ local_rotations[frame, joint]
                )
                global_positions[frame, joint] = (
                    global_positions[frame, parent]
                    + global_rotations[frame, parent] @ (joints[joint] - joints[parent])
                )
        global_positions[frame] += translation[frame]

    joint_error = np.linalg.norm(global_positions - expected_joints, axis=-1)
    if float(np.max(joint_error)) > 1e-5:
        raise RuntimeError(
            f"SMPL-X FK differs from MAMMA joints by {float(np.max(joint_error)):.6f} m"
        )

    source_to_y_up = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        dtype=np.float64,
    )
    root_origin = source_to_y_up @ global_positions[0, 0]
    canonical_positions = np.einsum(
        "ab,fjb->fja", source_to_y_up, global_positions
    ) - root_origin
    canonical_rotations = np.einsum(
        "ab,fjbc->fjac", source_to_y_up, global_rotations
    )
    canonical_vertices = np.einsum(
        "ab,fvb->fva", source_to_y_up, expected_vertices
    ) - root_origin
    pose_features = (local_rotations[:, 1:] - np.eye(3)).reshape(frames, -1)
    pose_offsets = np.einsum("vci,fi->fvc", pose_directions, pose_features)
    return {
        "effective_pose": effective_pose,
        "local_rotations": local_rotations,
        "canonical_positions": canonical_positions,
        "canonical_rotations": canonical_rotations,
        "canonical_vertices": canonical_vertices,
        "pose_offsets": pose_offsets.astype(np.float32),
        "joint_error": joint_error,
        "root_origin": root_origin,
    }


def _linearize_action(action: bpy.types.Action | None) -> None:
    if action is None:
        return
    if hasattr(action, "fcurves"):
        for curve in action.fcurves:
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "LINEAR"


def _animate(
    armature: bpy.types.Object,
    mesh: bpy.types.Object,
    data: dict[str, object],
    motion: dict[str, np.ndarray],
) -> None:
    names = data["names"]
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    frames = len(motion["canonical_positions"])
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.render.fps_base = 1.0
    # Blender's FBX round trip maps source frame 0 to imported frame 1. Author
    # at 0..149 so the delivered FBX opens on the conventional 1..150 range.
    scene.frame_start = 0
    scene.frame_end = frames - 1

    basis = mesh.shape_key_add(name="Basis", from_mix=False)
    basis.interpolation = "KEY_LINEAR"
    corrective_keys = []
    for frame in range(frames):
        key = mesh.shape_key_add(name=f"MAMMA_PoseCorrective_{frame:04d}", from_mix=False)
        key.interpolation = "KEY_LINEAR"
        coordinates = (vertices + motion["pose_offsets"][frame]).reshape(-1)
        key.data.foreach_set("co", coordinates)
        corrective_keys.append(key)
    for frame, key in enumerate(corrective_keys):
        if frame > 0:
            key.value = 0.0
            key.keyframe_insert(data_path="value", frame=frame - 1)
        key.value = 1.0
        key.keyframe_insert(data_path="value", frame=frame)
        if frame < frames - 1:
            key.value = 0.0
            key.keyframe_insert(data_path="value", frame=frame + 1)
        key.value = 0.0
    _linearize_action(mesh.data.shape_keys.animation_data.action)

    rest_matrices = {
        bone.name: bone.matrix_local.copy() for bone in armature.data.bones
    }
    armature.data.pose_position = "POSE"
    previous_quaternions: dict[str, object] = {}
    for frame in range(frames):
        scene.frame_set(frame)
        desired_matrices: dict[str, Matrix] = {}
        for joint, name in enumerate(names):
            rest_orientation = np.asarray(
                rest_matrices[name].to_3x3(), dtype=np.float64
            )
            rotation = (
                motion["canonical_rotations"][frame, joint] @ rest_orientation
            )
            desired_matrices[name] = Matrix.Translation(
                Vector(motion["canonical_positions"][frame, joint].tolist())
            ) @ Matrix(rotation.tolist()).to_4x4()
        for joint, name in enumerate(names):
            pose_bone = armature.pose.bones[name]
            pose_bone.rotation_mode = "QUATERNION"
            parent_index = int(data["parents"][joint])
            if parent_index < 0:
                basis = rest_matrices[name].inverted() @ desired_matrices[name]
            else:
                parent_name = names[parent_index]
                rest_relative = (
                    rest_matrices[parent_name].inverted() @ rest_matrices[name]
                )
                desired_relative = (
                    desired_matrices[parent_name].inverted() @ desired_matrices[name]
                )
                basis = rest_relative.inverted() @ desired_relative
            pose_bone.matrix_basis = basis
            quaternion = pose_bone.rotation_quaternion.copy()
            previous = previous_quaternions.get(name)
            if previous is not None and quaternion.dot(previous) < 0.0:
                pose_bone.rotation_quaternion = -quaternion
                quaternion = pose_bone.rotation_quaternion.copy()
            previous_quaternions[name] = quaternion
            pose_bone.keyframe_insert(data_path="location", frame=frame, group=name)
            pose_bone.keyframe_insert(
                data_path="rotation_quaternion", frame=frame, group=name
            )
            pose_bone.keyframe_insert(data_path="scale", frame=frame, group=name)
        bpy.context.view_layer.update()
        immediate_positions = np.asarray(
            [list(armature.pose.bones[name].matrix.translation) for name in names],
            dtype=np.float64,
        )
        immediate_error = float(
            np.max(
                np.linalg.norm(
                    immediate_positions - motion["canonical_positions"][frame], axis=1
                )
            )
        )
        if immediate_error > 1e-5:
            per_joint = np.linalg.norm(
                immediate_positions - motion["canonical_positions"][frame], axis=1
            )
            worst = sorted(
                (
                    {
                        "joint": names[index],
                        "error_m": float(per_joint[index]),
                        "actual": immediate_positions[index].tolist(),
                        "expected": motion["canonical_positions"][frame, index].tolist(),
                    }
                    for index in range(len(names))
                ),
                key=lambda value: value["error_m"],
                reverse=True,
            )[:8]
            raise RuntimeError(
                f"Blender pose assignment differs at frame {frame}: {json.dumps(worst)}"
            )
    _linearize_action(armature.animation_data.action)
    scene.frame_set(0)


def _find_character() -> tuple[bpy.types.Object, bpy.types.Object]:
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError(
            f"Expected one armature and one mesh, found {len(armatures)} and {len(meshes)}"
        )
    return armatures[0], meshes[0]


def _rig_joint_errors(
    armature: bpy.types.Object,
    names: tuple[str, ...],
    expected: np.ndarray,
    sample_frames: list[int],
    *,
    scene_offset: int,
) -> dict[str, object]:
    per_frame = []
    combined = []
    for frame in sample_frames:
        bpy.context.scene.frame_set(frame + scene_offset)
        bpy.context.view_layer.update()
        positions = np.asarray(
            [
                list(armature.matrix_world @ armature.pose.bones[name].matrix.translation)
                for name in names
            ],
            dtype=np.float64,
        )
        errors = np.linalg.norm(positions - expected[frame], axis=1)
        combined.append(errors)
        per_frame.append(
            {
                "source_frame": frame,
                "scene_frame": frame + scene_offset,
                "median_m": float(np.median(errors)),
                "p95_m": float(np.percentile(errors, 95)),
                "max_m": float(np.max(errors)),
            }
        )
    values = np.concatenate(combined)
    return {
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
        "max_m": float(np.max(values)),
        "per_frame": per_frame,
    }


def _action_fcurves(action: bpy.types.Action | None) -> list[object]:
    if action is None:
        return []
    return list(action.fcurves) if hasattr(action, "fcurves") else []


def _roundtrip_contract() -> dict[str, object]:
    armature, mesh = _find_character()
    bones = list(armature.data.bones)
    action = armature.animation_data.action if armature.animation_data else None
    shape_keys = mesh.data.shape_keys
    shape_action = shape_keys.animation_data.action if shape_keys and shape_keys.animation_data else None
    armature_curves = _action_fcurves(action)
    shape_curves = _action_fcurves(shape_action)
    animated_bones = {
        bone.name
        for bone in armature.pose.bones
        if any(f'pose.bones["{bone.name}"]' in curve.data_path for curve in armature_curves)
    }
    return {
        "bone_count": len(bones),
        "bone_names": [bone.name for bone in bones],
        "bone_parents": {
            bone.name: bone.parent.name if bone.parent else None for bone in bones
        },
        "vertices": len(mesh.data.vertices),
        "triangles": len(mesh.data.polygons),
        "uv_layers": len(mesh.data.uv_layers),
        "vertex_groups": len(mesh.vertex_groups),
        "shape_keys": len(shape_keys.key_blocks) if shape_keys else 0,
        "armature_action": action.name if action else None,
        "shape_action": shape_action.name if shape_action else None,
        "armature_action_range": list(action.frame_range) if action else None,
        "shape_action_range": list(shape_action.frame_range) if shape_action else None,
        "armature_fcurves": len(armature_curves),
        "shape_fcurves": len(shape_curves),
        "animated_bone_count": len(animated_bones),
    }


def _surface_errors(
    armature: bpy.types.Object,
    mesh: bpy.types.Object,
    expected: np.ndarray,
    expected_joints: np.ndarray,
    sample_frames: list[int],
) -> dict[str, object]:
    dependency_graph = bpy.context.evaluated_depsgraph_get()
    all_errors = []
    per_frame = []
    for frame in sample_frames:
        bpy.context.scene.frame_set(frame + 1)
        dependency_graph.update()
        evaluated_object = mesh.evaluated_get(dependency_graph)
        evaluated_mesh = evaluated_object.to_mesh()
        try:
            coordinates = np.empty(len(evaluated_mesh.vertices) * 3, dtype=np.float64)
            evaluated_mesh.vertices.foreach_get("co", coordinates)
            coordinates = coordinates.reshape(-1, 3)
            world = np.asarray(evaluated_object.matrix_world, dtype=np.float64)
            homogeneous = np.concatenate(
                [coordinates, np.ones((len(coordinates), 1), dtype=np.float64)], axis=1
            )
            world_coordinates = (homogeneous @ world.T)[:, :3]
            errors = np.linalg.norm(world_coordinates - expected[frame], axis=1)
            all_errors.append(errors)
            shape_keys = mesh.data.shape_keys
            active_keys = []
            if shape_keys is not None:
                active_keys = [
                    {"name": key.name, "value": float(key.value)}
                    for key in shape_keys.key_blocks[1:]
                    if abs(float(key.value)) > 1e-5
                ]
            posed_joints = np.asarray(
                [
                    list(
                        armature.matrix_world
                        @ armature.pose.bones[name].matrix.translation
                    )
                    for name in json.loads(
                        Path(__file__).resolve().parents[1]
                        .joinpath("src/autoanim_gnm/data/smplx55-v1.json")
                        .read_text(encoding="utf-8")
                    )["joint_names"]
                ],
                dtype=np.float64,
            )
            joint_errors = np.linalg.norm(posed_joints - expected_joints[frame], axis=1)
            per_frame.append(
                {
                    "source_frame": frame,
                    "scene_frame": frame + 1,
                    "median_m": float(np.median(errors)),
                    "p95_m": float(np.percentile(errors, 95)),
                    "max_m": float(np.max(errors)),
                    "evaluated_bounds_min": world_coordinates.min(axis=0).tolist(),
                    "evaluated_bounds_max": world_coordinates.max(axis=0).tolist(),
                    "expected_bounds_min": expected[frame].min(axis=0).tolist(),
                    "expected_bounds_max": expected[frame].max(axis=0).tolist(),
                    "active_shape_keys": active_keys,
                    "pelvis_position": list(
                        armature.matrix_world
                        @ armature.pose.bones["pelvis"].matrix.translation
                    ),
                    "expected_pelvis_position": expected_joints[frame, 0].tolist(),
                    "joint_median_m": float(np.median(joint_errors)),
                    "joint_p95_m": float(np.percentile(joint_errors, 95)),
                    "joint_max_m": float(np.max(joint_errors)),
                }
            )
        finally:
            evaluated_object.to_mesh_clear()
    errors = np.concatenate(all_errors)
    return {
        "median_m": float(np.median(errors)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
        "per_frame": per_frame,
    }


def main() -> None:
    model, params, vertices, skeleton, output, report_path = _arguments()
    _clear_scene()
    data = _load_sources(model, params, skeleton)
    armature, mesh = _build_character(data)
    motion = _motion(model, params, vertices, data)
    _animate(armature, mesh, data, motion)
    samples = [0, 37, 75, 112, 149]
    source_rig = _rig_joint_errors(
        armature,
        data["names"],
        motion["canonical_positions"],
        samples,
        scene_offset=0,
    )
    if source_rig["max_m"] > 1e-5:
        raise RuntimeError(
            "Authored Blender rig differs from MAMMA FK: "
            + json.dumps(source_rig, sort_keys=True)
        )
    source_bones = [bone.name for bone in armature.data.bones]
    source_parents = {
        bone.name: bone.parent.name if bone.parent else None for bone in armature.data.bones
    }
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=str(output),
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        use_mesh_modifiers=True,
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        axis_forward="-Z",
        axis_up="Y",
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode="AUTO",
        embed_textures=False,
    )
    if not output.is_file() or output.stat().st_size < 1_000_000:
        raise RuntimeError("Animated original MAMMA FBX is missing or unexpectedly small")

    _clear_scene()
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0
    bpy.ops.import_scene.fbx(filepath=str(output), automatic_bone_orientation=False)
    contract = _roundtrip_contract()
    armature, mesh = _find_character()
    roundtrip_rig = _rig_joint_errors(
        armature,
        data["names"],
        motion["canonical_positions"],
        samples,
        scene_offset=1,
    )
    surface = _surface_errors(
        armature,
        mesh,
        motion["canonical_vertices"],
        motion["canonical_positions"],
        samples,
    )
    failures = []
    if contract["bone_count"] != 55:
        failures.append("animated FBX does not contain 55 bones")
    if contract["bone_names"] != source_bones or contract["bone_parents"] != source_parents:
        failures.append("native SMPL-X hierarchy changed")
    if contract["animated_bone_count"] != 55:
        failures.append("not every native SMPL-X bone is animated")
    if contract["vertices"] != 10475 or contract["triangles"] != 20908:
        failures.append("native locked-head SMPL-X topology changed")
    if contract["uv_layers"] < 1 or contract["vertex_groups"] != 55:
        failures.append("UVs or native SMPL-X skinning groups are missing")
    if contract["shape_keys"] != 151 or contract["shape_fcurves"] != 150:
        failures.append("per-frame SMPL-X pose correctives were not preserved")
    if surface["max_m"] > 2e-4:
        failures.append(
            f"round-trip animated surface differs from MAMMA by {surface['max_m']:.6f} m"
        )

    report = {
        "schema_version": "autoanim.mamma-smplx-animated-fbx/1.0",
        "status": "passed" if not failures else "failed",
        "output_fbx": str(output),
        "identity": "MAMMA fitted subject-00, neutral gender locked-head SMPL-X",
        "gnm_included": False,
        "fps": 30,
        "frames": 150,
        "duration_seconds": 5.0,
        "root_space": "Y-up character-local, rebased from first MAMMA pelvis sample",
        "hand_mean": "native SMPL-X MANO means used by the pinned MAMMA run",
        "pose_correctives": "one baked SMPL-X corrective morph per source frame",
        "model_sha256": _sha256(model),
        "mamma_params_sha256": _sha256(params),
        "mamma_vertices_sha256": _sha256(vertices),
        "skeleton_contract_sha256": _sha256(skeleton),
        "source_fk_max_error_m": float(np.max(motion["joint_error"])),
        "authored_blender_rig_error": source_rig,
        "roundtrip_blender_rig_error": roundtrip_rig,
        "roundtrip": contract,
        "surface_samples": samples,
        "roundtrip_surface_error": surface,
        "failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("Animated MAMMA FBX verification failed: " + "; ".join(failures))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
