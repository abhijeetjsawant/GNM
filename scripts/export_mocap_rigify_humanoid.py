#!/usr/bin/env python3
"""Retarget one SMPL-X55 take onto a generic Rigify-compatible T-pose body.

The delivery rig deliberately differs from the fitted capture rig:

* its bind pose is a neutral 1.70 m T-pose;
* its bone names follow Blender Rigify's deform-chain conventions;
* its body is a generic rest mesh warped once into the target proportions;
* animation is skeletal only (no per-frame shapes or geometry cache).

Run with Blender:

    blender --background --python scripts/export_mocap_rigify_humanoid.py -- \
        MODEL.npz PARAMS.npz VERTS.npz SKELETON.json OUTPUT_DIRECTORY
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from autoanim_gnm.body import DETAILED_HUMANOID  # noqa: E402
from autoanim_gnm.mamma_motion import (  # noqa: E402
    load_mamma_body_track,
    rebase_mamma_body_track,
)
from export_mamma_smplx_animated_fbx import _linearize_action  # noqa: E402
from export_mamma_smplx_neutral_fbx import _clear_scene, _load_sources  # noqa: E402


OUTPUT_NEUTRAL = "mocap-blender-rigify-tpose-neutral.fbx"
OUTPUT_ANIMATED = "mocap-blender-rigify-animated.fbx"
OUTPUT_BLEND = "mocap-blender-rigify-animated.blend"
OUTPUT_REPORT = "mocap-blender-rigify-report.json"
FORBIDDEN_BINARY_LABEL = b"mamma"

# AutoAnim's generic humanoid uses +Y up, +Z forward and -X character-left.
# Blender uses +Z up, -Y forward and Rigify's .L side is +X.
AUTOANIM_TO_BLENDER = np.asarray(
    ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)

# The source model is +Y up, +Z back and +X character-left.
SOURCE_TO_BLENDER = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)

TARGET_TO_RIGIFY = {
    "Root": "root",
    "Hips": "spine",
    "Spine": "spine.001",
    "Chest": "spine.002",
    "UpperChest": "spine.003",
    "Neck": "spine.004",
    "Head": "spine.005",
    "LeftEye": "eye.L",
    "RightEye": "eye.R",
    "LeftShoulder": "shoulder.L",
    "LeftUpperArm": "upper_arm.L",
    "LeftLowerArm": "forearm.L",
    "LeftHand": "hand.L",
    "RightShoulder": "shoulder.R",
    "RightUpperArm": "upper_arm.R",
    "RightLowerArm": "forearm.R",
    "RightHand": "hand.R",
    "LeftUpperLeg": "thigh.L",
    "LeftLowerLeg": "shin.L",
    "LeftFoot": "foot.L",
    "LeftToes": "toe.L",
    "RightUpperLeg": "thigh.R",
    "RightLowerLeg": "shin.R",
    "RightFoot": "foot.R",
    "RightToes": "toe.R",
    "LeftThumbMetacarpal": "thumb.01.L",
    "LeftThumbProximal": "thumb.02.L",
    "LeftThumbDistal": "thumb.03.L",
    "LeftIndexProximal": "f_index.01.L",
    "LeftIndexIntermediate": "f_index.02.L",
    "LeftIndexDistal": "f_index.03.L",
    "LeftMiddleProximal": "f_middle.01.L",
    "LeftMiddleIntermediate": "f_middle.02.L",
    "LeftMiddleDistal": "f_middle.03.L",
    "LeftRingProximal": "f_ring.01.L",
    "LeftRingIntermediate": "f_ring.02.L",
    "LeftRingDistal": "f_ring.03.L",
    "LeftLittleProximal": "f_pinky.01.L",
    "LeftLittleIntermediate": "f_pinky.02.L",
    "LeftLittleDistal": "f_pinky.03.L",
    "RightThumbMetacarpal": "thumb.01.R",
    "RightThumbProximal": "thumb.02.R",
    "RightThumbDistal": "thumb.03.R",
    "RightIndexProximal": "f_index.01.R",
    "RightIndexIntermediate": "f_index.02.R",
    "RightIndexDistal": "f_index.03.R",
    "RightMiddleProximal": "f_middle.01.R",
    "RightMiddleIntermediate": "f_middle.02.R",
    "RightMiddleDistal": "f_middle.03.R",
    "RightRingProximal": "f_ring.01.R",
    "RightRingIntermediate": "f_ring.02.R",
    "RightRingDistal": "f_ring.03.R",
    "RightLittleProximal": "f_pinky.01.R",
    "RightLittleIntermediate": "f_pinky.02.R",
    "RightLittleDistal": "f_pinky.03.R",
}


def _arguments() -> tuple[Path, Path, Path, Path, Path]:
    try:
        separator = sys.argv.index("--")
    except ValueError as exc:
        raise SystemExit(
            "Expected MODEL PARAMS VERTICES SKELETON OUTPUT_DIRECTORY after '--'"
        ) from exc
    values = sys.argv[separator + 1 :]
    if len(values) != 5:
        raise SystemExit(
            "Expected MODEL PARAMS VERTICES SKELETON OUTPUT_DIRECTORY after '--'"
        )
    model, params, vertices, skeleton, output = (
        Path(value).expanduser().resolve() for value in values
    )
    for source in (model, params, vertices, skeleton):
        if not source.is_file():
            raise SystemExit(f"Required source is missing: {source}")
    output.mkdir(parents=True, exist_ok=True)
    return model, params, vertices, skeleton, output


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_positions() -> tuple[list[str], np.ndarray, np.ndarray]:
    names = [joint.name for joint in DETAILED_HUMANOID.joints]
    parents = np.asarray(
        [joint.parent for joint in DETAILED_HUMANOID.joints], dtype=np.int64
    )
    positions = np.zeros((len(names), 3), dtype=np.float64)
    for index, joint in enumerate(DETAILED_HUMANOID.joints):
        local = AUTOANIM_TO_BLENDER @ np.asarray(
            joint.rest_translation_m, dtype=np.float64
        )
        positions[index] = local if joint.parent < 0 else positions[joint.parent] + local
    return names, parents, positions


def _preferred_child(name: str, names: list[str], parents: np.ndarray) -> int | None:
    index = names.index(name)
    children = np.flatnonzero(parents == index).astype(int).tolist()
    preferred = {
        "Root": "Hips",
        "Hips": "Spine",
        "Spine": "Chest",
        "Chest": "UpperChest",
        "UpperChest": "Neck",
        "Neck": "Head",
        "LeftHand": "LeftMiddleProximal",
        "RightHand": "RightMiddleProximal",
        "LeftFoot": "LeftToes",
        "RightFoot": "RightToes",
    }.get(name)
    if preferred in names:
        candidate = names.index(preferred)
        if candidate in children:
            return candidate
    return children[0] if len(children) == 1 else None


def _tail(
    index: int, names: list[str], parents: np.ndarray, positions: np.ndarray
) -> Vector:
    head = Vector(positions[index].tolist())
    child = _preferred_child(names[index], names, parents)
    if child is not None:
        candidate = Vector(positions[child].tolist())
        if (candidate - head).length > 1e-5:
            return candidate
    if names[index] == "Root":
        return head + Vector((0.0, 0.0, 0.10))
    if names[index].endswith("Eye"):
        return head + Vector((0.0, -0.04, 0.0))
    parent = int(parents[index])
    direction = head - Vector(positions[parent].tolist()) if parent >= 0 else Vector()
    if direction.length < 1e-5:
        direction = Vector((0.0, 0.0, 0.04))
    else:
        direction.normalize()
        direction *= min(0.05, max(0.025, (head - Vector(positions[parent])).length * 0.5))
    return head + direction


def _build_armature(
    names: list[str], parents: np.ndarray, positions: np.ndarray
) -> bpy.types.Object:
    armature_data = bpy.data.armatures.new("Blender_Rigify_TPose")
    armature = bpy.data.objects.new("Blender_Rigify_TPose", armature_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones: list[object] = []
    for index, target_name in enumerate(names):
        bone = armature_data.edit_bones.new(TARGET_TO_RIGIFY[target_name])
        bone.head = Vector(positions[index].tolist())
        bone.tail = _tail(index, names, parents, positions)
        bone.use_connect = False
        edit_bones.append(bone)
    for index, parent in enumerate(parents):
        if parent >= 0:
            edit_bones[index].parent = edit_bones[int(parent)]
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.data.pose_position = "REST"
    armature["rig_family"] = "Blender Rigify-compatible humanoid"
    armature["bind_pose"] = "T-pose"
    armature["face_owner"] = "external face rig"
    for target_name in ("Root", "LeftEye", "RightEye"):
        armature.data.bones[TARGET_TO_RIGIFY[target_name]].use_deform = False
    return armature


def _source_positions(data: dict[str, object]) -> np.ndarray:
    return np.einsum(
        "ab,jb->ja", SOURCE_TO_BLENDER, np.asarray(data["joints"], dtype=np.float64)
    )


def _source_matrices(data: dict[str, object]) -> dict[str, Matrix]:
    names = list(data["names"])
    parents = np.asarray(data["parents"], dtype=np.int64)
    positions = _source_positions(data)
    armature_data = bpy.data.armatures.new("Source_Rest_Temporary")
    armature = bpy.data.objects.new("Source_Rest_Temporary", armature_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = []
    preferred = {
        "pelvis": "spine1",
        "spine1": "spine2",
        "spine2": "spine3",
        "spine3": "neck",
        "neck": "head",
        "left_wrist": "left_middle1",
        "right_wrist": "right_middle1",
    }
    for index, name in enumerate(names):
        bone = armature_data.edit_bones.new(name)
        head = Vector(positions[index].tolist())
        children = np.flatnonzero(parents == index).astype(int).tolist()
        child_name = preferred.get(name)
        child = names.index(child_name) if child_name in names else None
        if child not in children:
            child = children[0] if len(children) == 1 else None
        if child is not None:
            tail = Vector(positions[child].tolist())
        elif name == "head":
            tail = head + Vector((0.0, 0.0, 0.10))
        else:
            parent = int(parents[index])
            direction = head - Vector(positions[parent].tolist()) if parent >= 0 else Vector()
            if direction.length < 1e-5:
                direction = Vector((0.0, 0.0, 0.03))
            else:
                direction.normalize()
                direction *= 0.03
            tail = head + direction
        if (tail - head).length < 1e-5:
            tail = head + Vector((0.0, 0.0, 0.03))
        bone.head = head
        bone.tail = tail
        bone.use_connect = False
        edit_bones.append(bone)
    for index, parent in enumerate(parents):
        if parent >= 0:
            edit_bones[index].parent = edit_bones[int(parent)]
    bpy.ops.object.mode_set(mode="OBJECT")
    matrices = {bone.name: bone.matrix_local.copy() for bone in armature.data.bones}
    bpy.data.objects.remove(armature, do_unlink=True)
    bpy.data.armatures.remove(armature_data)
    return matrices


def _build_generic_mesh(
    armature: bpy.types.Object,
    data: dict[str, object],
    target_to_source: dict[str, str],
) -> bpy.types.Object:
    source_matrices = _source_matrices(data)
    target_matrices = {
        bone.name: bone.matrix_local.copy() for bone in armature.data.bones
    }
    source_to_target = {
        source: TARGET_TO_RIGIFY[target] for target, source in target_to_source.items()
    }
    for face_joint in ("jaw", "left_eye_smplhf", "right_eye_smplhf"):
        source_to_target[face_joint] = TARGET_TO_RIGIFY["Head"]

    source_names = list(data["names"])
    transforms: list[np.ndarray] = []
    for source_name in source_names:
        target_name = source_to_target[source_name]
        source_basis_name = "head" if source_name in {
            "jaw",
            "left_eye_smplhf",
            "right_eye_smplhf",
        } else source_name
        transform = target_matrices[target_name] @ source_matrices[source_basis_name].inverted()
        transforms.append(np.asarray(transform, dtype=np.float64))

    source_vertices = np.einsum(
        "ab,vb->va", SOURCE_TO_BLENDER, np.asarray(data["vertices"], dtype=np.float64)
    )
    homogeneous = np.concatenate(
        (source_vertices, np.ones((len(source_vertices), 1), dtype=np.float64)), axis=1
    )
    transformed = np.stack(
        [(matrix @ homogeneous.T).T[:, :3] for matrix in transforms], axis=1
    )
    weights = np.asarray(data["weights"], dtype=np.float64)
    target_vertices = np.einsum("vj,vjc->vc", weights, transformed)

    mesh_data = bpy.data.meshes.new("Generic_Rigify_Humanoid_Mesh")
    mesh_data.from_pydata(
        target_vertices.tolist(), [], np.asarray(data["faces"], dtype=np.int32).tolist()
    )
    mesh_data.update(calc_edges=True)
    mesh = bpy.data.objects.new("Generic_Rigify_Humanoid", mesh_data)
    bpy.context.collection.objects.link(mesh)

    uv_layer = mesh_data.uv_layers.new(name="Humanoid_UV")
    texture_vertices = np.asarray(data["texture_vertices"], dtype=np.float64)
    texture_faces = np.asarray(data["texture_faces"], dtype=np.int32)
    for polygon in mesh_data.polygons:
        for corner, loop_index in enumerate(polygon.loop_indices):
            uv = texture_vertices[texture_faces[polygon.index, corner]]
            uv_layer.data[loop_index].uv = (float(uv[0]), float(uv[1]))

    target_weights: dict[str, np.ndarray] = {}
    for source_index, source_name in enumerate(source_names):
        target_name = source_to_target[source_name]
        target_weights.setdefault(target_name, np.zeros(len(weights), dtype=np.float64))
        target_weights[target_name] += weights[:, source_index]
    for target_name, joint_weights in target_weights.items():
        group = mesh.vertex_groups.new(name=target_name)
        for vertex_index in np.flatnonzero(joint_weights > 1e-8):
            group.add(
                [int(vertex_index)], float(joint_weights[vertex_index]), "REPLACE"
            )

    material = bpy.data.materials.new("Generic_Humanoid_Material")
    material.diffuse_color = (0.42, 0.50, 0.58, 1.0)
    material.roughness = 0.78
    mesh_data.materials.append(material)
    modifier = mesh.modifiers.new(name="Rigify_Skin", type="ARMATURE")
    modifier.object = armature
    modifier.use_deform_preserve_volume = True
    mesh.parent = armature
    return mesh


def _quaternion_angle_degrees(left: object, right: object) -> float:
    dot = min(1.0, max(-1.0, abs(float(left.dot(right)))))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _clean_target_quaternions(
    values_xyzw: np.ndarray, names: list[str]
) -> tuple[np.ndarray, dict[str, object]]:
    """Repair isolated hand-fit spikes without smoothing valid body motion."""

    output = np.asarray(values_xyzw, dtype=np.float64).copy()
    repaired: list[dict[str, object]] = []
    finger_indices = [
        index
        for index, name in enumerate(names)
        if any(token in name for token in ("Thumb", "Index", "Middle", "Ring", "Little"))
    ]
    for joint in finger_indices:
        quaternions = [
            __import__("mathutils").Quaternion(
                (output[frame, joint, 3], *output[frame, joint, :3])
            )
            for frame in range(len(output))
        ]
        for frame in range(1, len(quaternions) - 1):
            before = _quaternion_angle_degrees(quaternions[frame - 1], quaternions[frame])
            after = _quaternion_angle_degrees(quaternions[frame], quaternions[frame + 1])
            across = _quaternion_angle_degrees(
                quaternions[frame - 1], quaternions[frame + 1]
            )
            if before > 30.0 and after > 30.0 and across < 12.0:
                quaternions[frame] = quaternions[frame - 1].slerp(
                    quaternions[frame + 1], 0.5
                )
                repaired.append(
                    {
                        "bone": TARGET_TO_RIGIFY[names[joint]],
                        "frame": frame,
                        "original_jump_degrees": before,
                        "method": "isolated_quaternion_spike_interpolation",
                    }
                )
        for frame in range(1, len(quaternions)):
            angle = _quaternion_angle_degrees(quaternions[frame - 1], quaternions[frame])
            if angle > 20.0:
                quaternions[frame] = quaternions[frame - 1].slerp(
                    quaternions[frame], 20.0 / angle
                )
                repaired.append(
                    {
                        "bone": TARGET_TO_RIGIFY[names[joint]],
                        "frame": frame,
                        "original_jump_degrees": angle,
                        "method": "20_degree_hand_step_limit",
                    }
                )
        for frame, quaternion in enumerate(quaternions):
            output[frame, joint] = (
                quaternion.x,
                quaternion.y,
                quaternion.z,
                quaternion.w,
            )
    return output, {"repair_count": len(repaired), "repairs": repaired}


def _animate(
    armature: bpy.types.Object,
    mesh: bpy.types.Object,
    params_path: Path,
) -> dict[str, object]:
    names = [joint.name for joint in DETAILED_HUMANOID.joints]
    parents = np.asarray(
        [joint.parent for joint in DETAILED_HUMANOID.joints], dtype=np.int64
    )
    rig_names = [TARGET_TO_RIGIFY[name] for name in names]
    rest = {bone.name: bone.matrix_local.copy() for bone in armature.data.bones}
    track = rebase_mamma_body_track(
        load_mamma_body_track(params_path, sample_rate_hz=30, source_up_axis="z")
    )
    local_xyzw, cleanup = _clean_target_quaternions(
        np.asarray(track.local_rotations_xyzw, dtype=np.float64), names
    )
    local_matrices = np.empty((len(local_xyzw), len(names), 3, 3), dtype=np.float64)
    for frame in range(len(local_xyzw)):
        for joint in range(len(names)):
            x, y, z, w = local_xyzw[frame, joint]
            local_matrices[frame, joint] = np.asarray(
                __import__("mathutils").Quaternion((w, x, y, z)).to_matrix(),
                dtype=np.float64,
            )
    world_autoanim = np.empty_like(local_matrices)
    for frame in range(len(local_matrices)):
        for joint, parent in enumerate(parents):
            world_autoanim[frame, joint] = (
                local_matrices[frame, joint]
                if parent < 0
                else world_autoanim[frame, int(parent)] @ local_matrices[frame, joint]
            )
    root_motion = np.einsum(
        "ab,fb->fa",
        AUTOANIM_TO_BLENDER,
        np.asarray(track.root_translation_m, dtype=np.float64),
    )

    frames = len(world_autoanim)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.render.fps_base = 1.0
    scene.frame_start = 0
    scene.frame_end = frames - 1
    armature.data.pose_position = "POSE"
    previous_quaternions: dict[str, object] = {}
    adjacent_angles: dict[str, list[float]] = {name: [] for name in rig_names}
    foot_targets = ("LeftFoot", "LeftToes", "RightFoot", "RightToes")

    def desired_without_ground(frame: int) -> dict[str, Matrix]:
        desired: dict[str, Matrix] = {}
        root_name = TARGET_TO_RIGIFY["Root"]
        desired[root_name] = Matrix.Translation(Vector(root_motion[frame].tolist())) @ rest[root_name]
        for index, target_name in enumerate(names[1:], start=1):
            rig_name = TARGET_TO_RIGIFY[target_name]
            parent_rig = rig_names[int(parents[index])]
            rest_relative = rest[parent_rig].inverted() @ rest[rig_name]
            inherited = desired[parent_rig] @ rest_relative
            delta = (
                AUTOANIM_TO_BLENDER
                @ world_autoanim[frame, index]
                @ AUTOANIM_TO_BLENDER.T
            )
            desired[rig_name] = Matrix.Translation(inherited.translation) @ Matrix(
                (delta @ np.asarray(rest[rig_name].to_3x3(), dtype=np.float64)).tolist()
            ).to_4x4()
        return desired

    for frame in range(frames):
        scene.frame_set(frame)
        desired = desired_without_ground(frame)

        for index, rig_name in enumerate(rig_names):
            pose_bone = armature.pose.bones[rig_name]
            pose_bone.rotation_mode = "QUATERNION"
            parent = int(parents[index])
            basis = (
                rest[rig_name].inverted() @ desired[rig_name]
                if parent < 0
                else (rest[rig_names[parent]].inverted() @ rest[rig_name]).inverted()
                @ (desired[rig_names[parent]].inverted() @ desired[rig_name])
            )
            pose_bone.matrix_basis = basis
            quaternion = pose_bone.rotation_quaternion.copy()
            previous = previous_quaternions.get(rig_name)
            if previous is not None:
                if quaternion.dot(previous) < 0.0:
                    quaternion = -quaternion
                    pose_bone.rotation_quaternion = quaternion
                dot = min(1.0, max(-1.0, abs(quaternion.dot(previous))))
                adjacent_angles[rig_name].append(float(np.degrees(2.0 * np.arccos(dot))))
            previous_quaternions[rig_name] = quaternion.copy()
            pose_bone.keyframe_insert(data_path="location", frame=frame, group=rig_name)
            pose_bone.keyframe_insert(
                data_path="rotation_quaternion", frame=frame, group=rig_name
            )
            pose_bone.keyframe_insert(data_path="scale", frame=frame, group=rig_name)
        bpy.context.view_layer.update()

    # Compute contact from the deformed generic body rather than assuming the
    # feet remain the support feature. This take moves from standing to floor
    # contact, so the support surface transfers through legs, hips and torso.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    raw_surface_floors: list[float] = []
    for frame in range(frames):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        evaluated = mesh.evaluated_get(depsgraph)
        coordinates = np.empty(len(evaluated.data.vertices) * 3, dtype=np.float64)
        evaluated.data.vertices.foreach_get("co", coordinates)
        coordinates = coordinates.reshape(-1, 3)
        world = np.asarray(mesh.matrix_world, dtype=np.float64)
        homogeneous = np.concatenate(
            (coordinates, np.ones((len(coordinates), 1), dtype=np.float64)), axis=1
        )
        world_coordinates = (world @ homogeneous.T).T[:, :3]
        raw_surface_floors.append(float(np.quantile(world_coordinates[:, 2], 0.001)))
    raw_corrections = -np.asarray(raw_surface_floors, dtype=np.float64)
    padded = np.pad(raw_corrections, (2, 2), mode="edge")
    ground_corrections = np.asarray(
        [np.median(padded[frame : frame + 5]) for frame in range(frames)],
        dtype=np.float64,
    )
    for frame in range(1, frames):
        delta = ground_corrections[frame] - ground_corrections[frame - 1]
        ground_corrections[frame] = ground_corrections[frame - 1] + np.clip(
            delta, -0.03, 0.03
        )
    lowest_after_smoothing = float(np.min(ground_corrections - raw_corrections))
    if lowest_after_smoothing < 0.0:
        ground_corrections -= lowest_after_smoothing

    root_pose = armature.pose.bones[TARGET_TO_RIGIFY["Root"]]
    for frame, correction in enumerate(ground_corrections):
        scene.frame_set(frame)
        root_pose.matrix = Matrix.Translation(
            Vector((0.0, 0.0, float(correction)))
        ) @ root_pose.matrix
        root_pose.keyframe_insert(data_path="location", frame=frame, group=root_pose.name)
    _linearize_action(armature.animation_data.action if armature.animation_data else None)

    foot_heights: list[float] = []
    surface_floors: list[float] = []
    for frame in range(frames):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        foot_heights.append(
            min(
                armature.pose.bones[TARGET_TO_RIGIFY[name]].matrix.translation.z
                for name in foot_targets
            )
        )
        evaluated = mesh.evaluated_get(depsgraph)
        coordinates = np.empty(len(evaluated.data.vertices) * 3, dtype=np.float64)
        evaluated.data.vertices.foreach_get("co", coordinates)
        coordinates = coordinates.reshape(-1, 3)
        world = np.asarray(mesh.matrix_world, dtype=np.float64)
        homogeneous = np.concatenate(
            (coordinates, np.ones((len(coordinates), 1), dtype=np.float64)), axis=1
        )
        surface_floors.append(
            float(np.quantile((world @ homogeneous.T).T[:, 2], 0.001))
        )
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        raise RuntimeError("Rigify-compatible humanoid did not create an action")
    action.name = "Mocap_Rigify_Humanoid_Animation"
    _linearize_action(action)
    scene.frame_set(0)
    return {
        "translation_scale": 1.0,
        "hand_cleanup": cleanup,
        "max_adjacent_rotation_degrees": max(
            max(values, default=0.0) for values in adjacent_angles.values()
        ),
        "max_adjacent_rotation_by_bone_degrees": {
            name: max(values, default=0.0) for name, values in adjacent_angles.items()
        },
        "foot_joint_height_m": {
            "min": float(np.min(foot_heights)),
            "median": float(np.median(foot_heights)),
            "max": float(np.max(foot_heights)),
        },
        "ground_contact": {
            "mode": "lowest_deformed_body_surface_to_ground",
            "surface_quantile": 0.001,
            "surface_height_min_m": float(np.min(surface_floors)),
            "surface_height_median_m": float(np.median(surface_floors)),
            "surface_height_max_m": float(np.max(surface_floors)),
            "correction_min_m": float(np.min(ground_corrections)),
            "correction_median_m": float(np.median(ground_corrections)),
            "correction_max_m": float(np.max(ground_corrections)),
            "max_adjacent_correction_m": float(
                np.max(np.abs(np.diff(ground_corrections)))
            ),
        },
    }


def _select_character(armature: bpy.types.Object, mesh: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature


def _export_fbx(
    path: Path, armature: bpy.types.Object, mesh: bpy.types.Object, *, animated: bool
) -> None:
    _select_character(armature, mesh)
    bpy.ops.export_scene.fbx(
        filepath=str(path),
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
        bake_anim=animated,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode="AUTO",
        embed_textures=False,
    )
    if not path.is_file() or path.stat().st_size < 100_000:
        raise RuntimeError(f"FBX is missing or unexpectedly small: {path}")
    if path.read_bytes().lower().count(FORBIDDEN_BINARY_LABEL):
        raise RuntimeError(f"Legacy provider label leaked into {path.name}")


def _roundtrip(path: Path, *, animated: bool) -> dict[str, object]:
    _clear_scene()
    bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError("Round trip did not produce one armature and one mesh")
    armature, mesh = armatures[0], meshes[0]
    action = armature.animation_data.action if armature.animation_data else None
    return {
        "bones": len(armature.data.bones),
        "bone_names": [bone.name for bone in armature.data.bones],
        "bone_parents": {
            bone.name: bone.parent.name if bone.parent else None
            for bone in armature.data.bones
        },
        "root_bones": [bone.name for bone in armature.data.bones if bone.parent is None],
        "vertices": len(mesh.data.vertices),
        "triangles": len(mesh.data.polygons),
        "uv_layers": len(mesh.data.uv_layers),
        "vertex_groups": len(mesh.vertex_groups),
        "shape_keys": len(mesh.data.shape_keys.key_blocks) if mesh.data.shape_keys else 0,
        "actions": len(bpy.data.actions),
        "action_range": list(action.frame_range) if action else None,
        "animated": animated,
    }


def main() -> None:
    model, params, vertices, skeleton, output_dir = _arguments()
    _clear_scene()
    data = _load_sources(model, params, skeleton)
    contract = json.loads(skeleton.read_text(encoding="utf-8"))
    target_to_source = {
        str(target): str(source)
        for target, source in contract["target_to_source"].items()
    }
    target_names, target_parents, target_positions = _target_positions()
    armature = _build_armature(target_names, target_parents, target_positions)
    mesh = _build_generic_mesh(armature, data, target_to_source)
    neutral_path = output_dir / OUTPUT_NEUTRAL
    animated_path = output_dir / OUTPUT_ANIMATED
    blend_path = output_dir / OUTPUT_BLEND
    report_path = output_dir / OUTPUT_REPORT

    neutral_bone_names = [bone.name for bone in armature.data.bones]
    neutral_bone_parents = {
        bone.name: bone.parent.name if bone.parent else None for bone in armature.data.bones
    }
    _export_fbx(neutral_path, armature, mesh, animated=False)

    animation = _animate(armature, mesh, params)
    _export_fbx(animated_path, armature, mesh, animated=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)

    neutral_roundtrip = _roundtrip(neutral_path, animated=False)
    animated_roundtrip = _roundtrip(animated_path, animated=True)
    expected_names = [TARGET_TO_RIGIFY[name] for name in target_names]
    expected_parents = {
        TARGET_TO_RIGIFY[name]: (
            TARGET_TO_RIGIFY[target_names[int(target_parents[index])]]
            if target_parents[index] >= 0
            else None
        )
        for index, name in enumerate(target_names)
    }
    failures: list[str] = []
    for label, value in (
        ("neutral", neutral_roundtrip),
        ("animated", animated_roundtrip),
    ):
        if value["bones"] != 55 or set(value["bone_names"]) != set(expected_names):
            failures.append(f"{label} bone contract changed")
        if value["bone_parents"] != expected_parents or value["root_bones"] != ["root"]:
            failures.append(f"{label} hierarchy changed")
        if value["vertices"] != 10475 or value["triangles"] != 20908:
            failures.append(f"{label} mesh topology changed")
        if value["uv_layers"] < 1 or value["shape_keys"] != 0:
            failures.append(f"{label} UV or skeletal-only contract failed")
    if neutral_roundtrip["actions"] != 0:
        failures.append("neutral T-pose unexpectedly contains animation")
    if animated_roundtrip["actions"] != 1:
        failures.append("animated humanoid does not contain exactly one action")
    if animated_roundtrip["action_range"] != [1.0, 150.0]:
        failures.append("animated humanoid is not exactly 150 frames")
    if animation["max_adjacent_rotation_degrees"] > 20.0:
        failures.append("retargeted animation contains a rotation discontinuity")
    if abs(animation["ground_contact"]["surface_height_min_m"]) > 1e-5:
        failures.append("retargeted body contact does not meet the ground plane")
    if animation["ground_contact"]["max_adjacent_correction_m"] > 0.05:
        failures.append("ground-contact correction changes too abruptly")

    report = {
        "schema_version": "autoanim.blender-rigify-tpose-retarget/1.0",
        "status": "passed" if not failures else "failed",
        "delivery": {
            "neutral_fbx": neutral_path.name,
            "animated_fbx": animated_path.name,
            "animated_blend": blend_path.name,
        },
        "rig": {
            "family": "Blender Rigify-compatible deform skeleton",
            "bind_pose": "T-pose",
            "bones": 55,
            "face_animation": "external/GNM-owned",
            "shape_keys": 0,
            "geometry_cache": False,
        },
        "animation": {
            "frames": 150,
            "fps": 30,
            "duration_seconds": 5.0,
            **animation,
        },
        "source_hashes": {
            "model": _sha256(model),
            "motion": _sha256(params),
            "surface_reference": _sha256(vertices),
            "skeleton_contract": _sha256(skeleton),
        },
        "neutral_roundtrip": neutral_roundtrip,
        "animated_roundtrip": animated_roundtrip,
        "authored_bone_names": neutral_bone_names,
        "authored_bone_parents": neutral_bone_parents,
        "output_hashes": {
            neutral_path.name: _sha256(neutral_path),
            animated_path.name: _sha256(animated_path),
            blend_path.name: _sha256(blend_path),
        },
        "failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("Rigify-compatible delivery failed: " + "; ".join(failures))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
