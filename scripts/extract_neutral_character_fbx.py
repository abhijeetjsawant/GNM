#!/usr/bin/env python3
"""Extract a static bind/rest-pose character FBX from an animated FBX.

Run through Blender:

    blender --background --python scripts/extract_neutral_character_fbx.py -- \
        ANIMATED.fbx NEUTRAL.fbx REPORT.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix


def _arguments() -> tuple[Path, Path, Path]:
    try:
        separator = sys.argv.index("--")
    except ValueError as exc:
        raise SystemExit("Expected ANIMATED.fbx NEUTRAL.fbx REPORT.json after '--'") from exc
    values = sys.argv[separator + 1 :]
    if len(values) != 3:
        raise SystemExit("Expected ANIMATED.fbx NEUTRAL.fbx REPORT.json after '--'")
    source, output, report = (Path(value).expanduser().resolve() for value in values)
    if not source.is_file() or source.suffix.lower() != ".fbx":
        raise SystemExit(f"Input is not an existing FBX: {source}")
    if output.suffix.lower() != ".fbx":
        raise SystemExit(f"Output must use the .fbx extension: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return source, output, report


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.actions, bpy.data.armatures, bpy.data.meshes):
        for item in list(collection):
            collection.remove(item)


def _contract(*, require_static: bool) -> dict[str, object]:
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one armature, found {len(armatures)}")
    if not meshes:
        raise RuntimeError("Expected at least one mesh")
    armature = armatures[0]
    bones = list(armature.data.bones)
    parents = {bone.name: bone.parent.name if bone.parent else None for bone in bones}
    skinned = []
    for mesh in meshes:
        armature_modifiers = [
            modifier
            for modifier in mesh.modifiers
            if modifier.type == "ARMATURE" and modifier.object == armature
        ]
        if armature_modifiers:
            skinned.append(
                {
                    "name": mesh.name,
                    "vertices": len(mesh.data.vertices),
                    "polygons": len(mesh.data.polygons),
                    "vertex_groups": len(mesh.vertex_groups),
                }
            )
    if not skinned:
        raise RuntimeError("No mesh is skinned to the armature")
    action_count = len(bpy.data.actions)
    nla_strip_count = sum(
        len(track.strips)
        for obj in bpy.context.scene.objects
        if obj.animation_data
        for track in obj.animation_data.nla_tracks
    )
    identity = Matrix.Identity(4)
    maximum_pose_basis_error = max(
        max(abs(value) for row in (pose_bone.matrix_basis - identity) for value in row)
        for pose_bone in armature.pose.bones
    )
    if require_static and (action_count or nla_strip_count):
        raise RuntimeError("Neutral FBX unexpectedly contains animation")
    if require_static and maximum_pose_basis_error > 1e-6:
        raise RuntimeError("Neutral FBX is not in its bind/rest pose")
    return {
        "armature": armature.name,
        "bone_count": len(bones),
        "bone_names": [bone.name for bone in bones],
        "bone_parents": parents,
        "root_bones": [bone.name for bone in bones if bone.parent is None],
        "meshes": [
            {
                "name": mesh.name,
                "vertices": len(mesh.data.vertices),
                "polygons": len(mesh.data.polygons),
            }
            for mesh in meshes
        ],
        "skinned_meshes": skinned,
        "action_count": action_count,
        "nla_strip_count": nla_strip_count,
        "maximum_pose_basis_error": maximum_pose_basis_error,
    }


def _remove_animation_and_restore_bind_pose() -> None:
    for obj in bpy.context.scene.objects:
        obj.animation_data_clear()
        data = getattr(obj, "data", None)
        if data is not None and hasattr(data, "animation_data_clear"):
            data.animation_data_clear()
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    armature = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    armature.data.pose_position = "REST"
    for pose_bone in armature.pose.bones:
        pose_bone.matrix_basis.identity()
    bpy.context.view_layer.update()


def main() -> None:
    source, output, report_path = _arguments()
    _clear_scene()
    bpy.ops.import_scene.fbx(filepath=str(source), automatic_bone_orientation=False)
    source_contract = _contract(require_static=False)
    if source_contract["bone_count"] != 55:
        raise RuntimeError(
            f"Expected the AutoAnim-55 skeleton, found {source_contract['bone_count']} bones"
        )
    _remove_animation_and_restore_bind_pose()
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
        bake_anim=False,
        path_mode="AUTO",
        embed_textures=False,
    )
    if not output.is_file() or output.stat().st_size < 100_000:
        raise RuntimeError("Neutral FBX export is missing or unexpectedly small")

    _clear_scene()
    bpy.ops.import_scene.fbx(filepath=str(output), automatic_bone_orientation=False)
    neutral_contract = _contract(require_static=True)
    failures = []
    if neutral_contract["bone_count"] != 55:
        failures.append("neutral FBX does not contain 55 bones")
    if neutral_contract["bone_names"] != source_contract["bone_names"]:
        failures.append("bone names or order changed")
    if neutral_contract["bone_parents"] != source_contract["bone_parents"]:
        failures.append("bone hierarchy changed")
    if neutral_contract["skinned_meshes"] != source_contract["skinned_meshes"]:
        failures.append("skinned mesh contract changed")
    if neutral_contract["action_count"] or neutral_contract["nla_strip_count"]:
        failures.append("animation remains in the neutral FBX")
    if neutral_contract["maximum_pose_basis_error"] > 1e-6:
        failures.append("neutral character is not in bind/rest pose")

    report = {
        "schema_version": "autoanim.neutral-fbx-report/1.0",
        "source_animated_fbx": str(source),
        "output_neutral_fbx": str(output),
        "pose": "skeleton_bind_rest_pose",
        "source": source_contract,
        "neutral_roundtrip": neutral_contract,
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("Neutral FBX verification failed: " + "; ".join(failures))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
