#!/usr/bin/env python3
"""Export a production-style skinned humanoid FBX with skeletal animation only.

The delivered file contains one neutral mesh, one 55-joint hierarchy, skinning
weights, and keyed joint transforms. It intentionally contains no per-frame
morph targets or geometry-cache substitutes.
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
from export_mamma_smplx_animated_fbx import (  # noqa: E402
    _find_character,
    _linearize_action,
    _motion,
    _rig_joint_errors,
    _surface_errors,
)


FORBIDDEN_BINARY_LABEL = b"mamma"
REQUIRED_HUMANOID_BONES = {
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "head",
    "left_collar",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_collar",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
}


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
    paths = tuple(Path(value).expanduser().resolve() for value in values)
    model, params, vertices, skeleton, output, report = paths
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


def _rename_character(armature: bpy.types.Object, mesh: bpy.types.Object) -> None:
    armature.name = "Mocap_Humanoid_Rig"
    armature.data.name = "Mocap_Humanoid_Rig"
    mesh.name = "Mocap_Humanoid_Body"
    mesh.data.name = "Mocap_Humanoid_Mesh"
    for index, material in enumerate(mesh.data.materials):
        if material is not None:
            material.name = f"Mocap_Humanoid_Material_{index:02d}"
    for modifier in mesh.modifiers:
        if modifier.type == "ARMATURE":
            modifier.name = "Mocap_Humanoid_Skin"


def _animate_skeleton(
    armature: bpy.types.Object,
    data: dict[str, object],
    motion: dict[str, np.ndarray],
) -> None:
    names = data["names"]
    frames = len(motion["canonical_positions"])
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.render.fps_base = 1.0
    scene.frame_start = 0
    scene.frame_end = frames - 1
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
            rotation = motion["canonical_rotations"][frame, joint] @ rest_orientation
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
                    desired_matrices[parent_name].inverted()
                    @ desired_matrices[name]
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
        actual = np.asarray(
            [list(armature.pose.bones[name].matrix.translation) for name in names],
            dtype=np.float64,
        )
        error = float(
            np.max(
                np.linalg.norm(
                    actual - motion["canonical_positions"][frame], axis=1
                )
            )
        )
        if error > 1e-5:
            raise RuntimeError(
                f"Authored humanoid pose differs at frame {frame}: {error:.9f} m"
            )
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        raise RuntimeError("Humanoid skeleton did not create an animation action")
    action.name = "Mocap_Humanoid_Animation"
    _linearize_action(action)
    scene.frame_set(0)


def _action_fcurves(action: bpy.types.Action | None) -> list[object]:
    if action is None or not hasattr(action, "fcurves"):
        return []
    return list(action.fcurves)


def _contract() -> dict[str, object]:
    armature, mesh = _find_character()
    bones = list(armature.data.bones)
    action = armature.animation_data.action if armature.animation_data else None
    curves = _action_fcurves(action)
    animated_bones = {
        bone.name
        for bone in armature.pose.bones
        if any(f'pose.bones["{bone.name}"]' in curve.data_path for curve in curves)
    }
    shape_keys = mesh.data.shape_keys
    armature_modifiers = [
        modifier
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE" and modifier.object == armature
    ]
    return {
        "armature": armature.name,
        "mesh": mesh.name,
        "bones": len(bones),
        "bone_names": [bone.name for bone in bones],
        "bone_parents": {
            bone.name: bone.parent.name if bone.parent else None for bone in bones
        },
        "root_bones": [bone.name for bone in bones if bone.parent is None],
        "vertices": len(mesh.data.vertices),
        "triangles": len(mesh.data.polygons),
        "uv_layers": len(mesh.data.uv_layers),
        "vertex_groups": len(mesh.vertex_groups),
        "armature_modifiers": len(armature_modifiers),
        "mesh_parented_to_armature": mesh.parent == armature,
        "shape_keys": len(shape_keys.key_blocks) if shape_keys else 0,
        "actions": len(bpy.data.actions),
        "action": action.name if action else None,
        "action_range": list(action.frame_range) if action else None,
        "armature_fcurves": len(curves),
        "animated_bones": len(animated_bones),
        "required_humanoid_bones_present": sorted(
            REQUIRED_HUMANOID_BONES.intersection({bone.name for bone in bones})
        ),
        "missing_required_humanoid_bones": sorted(
            REQUIRED_HUMANOID_BONES.difference({bone.name for bone in bones})
        ),
    }


def main() -> None:
    model, params, vertices, skeleton, output, report_path = _arguments()
    _clear_scene()
    data = _load_sources(model, params, skeleton)
    armature, mesh = _build_character(data)
    _rename_character(armature, mesh)
    motion = _motion(model, params, vertices, data)
    _animate_skeleton(armature, data, motion)
    if mesh.data.shape_keys is not None:
        raise RuntimeError("Skeleton-only humanoid unexpectedly contains shape keys")

    samples = [0, 37, 75, 112, 149]
    authored_rig = _rig_joint_errors(
        armature,
        data["names"],
        motion["canonical_positions"],
        samples,
        scene_offset=0,
    )
    if authored_rig["max_m"] > 1e-5:
        raise RuntimeError(
            f"Authored humanoid rig exceeds tolerance: {authored_rig['max_m']:.9f} m"
        )
    source_bone_names = [bone.name for bone in armature.data.bones]
    source_bone_parents = {
        bone.name: bone.parent.name if bone.parent else None
        for bone in armature.data.bones
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
    if not output.is_file() or output.stat().st_size < 100_000:
        raise RuntimeError("Humanoid FBX is missing or unexpectedly small")
    binary_label_hits = output.read_bytes().lower().count(FORBIDDEN_BINARY_LABEL)

    _clear_scene()
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0
    bpy.ops.import_scene.fbx(filepath=str(output), automatic_bone_orientation=False)
    contract = _contract()
    armature, mesh = _find_character()
    roundtrip_rig = _rig_joint_errors(
        armature,
        data["names"],
        motion["canonical_positions"],
        samples,
        scene_offset=1,
    )
    skin_surface_error = _surface_errors(
        armature,
        mesh,
        motion["canonical_vertices"],
        motion["canonical_positions"],
        samples,
    )

    failures = []
    if binary_label_hits:
        failures.append("legacy provider label remains in FBX binary")
    if contract["bones"] != 55:
        failures.append("humanoid does not contain 55 joints")
    if contract["bone_names"] != source_bone_names:
        failures.append("joint names changed during FBX round trip")
    if contract["bone_parents"] != source_bone_parents:
        failures.append("joint hierarchy changed during FBX round trip")
    if contract["root_bones"] != ["pelvis"]:
        failures.append("pelvis is not the sole humanoid root")
    if contract["animated_bones"] != 55:
        failures.append("not every humanoid joint is animated")
    if contract["vertices"] != 10475 or contract["triangles"] != 20908:
        failures.append("humanoid mesh topology changed")
    if contract["uv_layers"] < 1 or contract["vertex_groups"] != 55:
        failures.append("humanoid UVs or skinning groups are missing")
    if contract["armature_modifiers"] != 1 or not contract["mesh_parented_to_armature"]:
        failures.append("humanoid mesh is not bound to its skeleton")
    if contract["shape_keys"] != 0:
        failures.append("frame-specific shape keys remain")
    if contract["actions"] != 1 or contract["animated_bones"] != 55:
        failures.append("humanoid skeletal action is incomplete")
    if contract["missing_required_humanoid_bones"]:
        failures.append("required humanoid joints are missing")
    if contract["action_range"] != [1.0, 150.0]:
        failures.append("skeletal action is not exactly 150 frames")
    if roundtrip_rig["max_m"] > 1e-4:
        failures.append(
            f"round-trip joint error is {roundtrip_rig['max_m']:.9f} m"
        )

    report = {
        "schema_version": "autoanim.mocap-humanoid-fbx/1.0",
        "status": "passed" if not failures else "failed",
        "output": output.name,
        "output_sha256": _sha256(output),
        "source_hashes": {
            "model": _sha256(model),
            "motion": _sha256(params),
            "surface_reference": _sha256(vertices),
            "skeleton_contract": _sha256(skeleton),
        },
        "delivery_profile": "skinned_humanoid_skeletal_animation",
        "joint_profile": "55-joint body, hands, jaw, and eyes",
        "fps": 30,
        "frames": 150,
        "duration_seconds": 5.0,
        "shape_keys": 0,
        "geometry_cache": False,
        "authored_rig_error": authored_rig,
        "roundtrip_rig_error": roundtrip_rig,
        "linear_skinning_surface_difference": skin_surface_error,
        "roundtrip": contract,
        "binary_legacy_label_hits": binary_label_hits,
        "failures": failures,
    }
    report_text = json.dumps(report, indent=2) + "\n"
    if FORBIDDEN_BINARY_LABEL.decode("ascii") in report_text.lower():
        raise RuntimeError("Legacy provider label remains in delivery report")
    report_path.write_text(report_text, encoding="utf-8")
    if failures:
        raise RuntimeError("Humanoid FBX verification failed: " + "; ".join(failures))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
