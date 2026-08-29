#!/usr/bin/env python3
"""Convert an animated AutoAnim GLB to FBX and verify the exported skeleton.

Run through Blender:

    blender --background --python scripts/export_character_fbx.py -- \
        INPUT.glb OUTPUT.fbx REPORT.json

The verification pass clears the scene, imports the generated FBX, and checks
that the complete deform skeleton, hierarchy, skinned mesh, and animation have
survived the round trip.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import bpy


def _arguments() -> tuple[Path, Path, Path]:
    try:
        separator = sys.argv.index("--")
    except ValueError as exc:
        raise SystemExit("Expected INPUT.glb OUTPUT.fbx REPORT.json after '--'") from exc
    values = sys.argv[separator + 1 :]
    if len(values) != 3:
        raise SystemExit("Expected INPUT.glb OUTPUT.fbx REPORT.json after '--'")
    source, output, report = (Path(value).expanduser().resolve() for value in values)
    if not source.is_file() or source.suffix.lower() != ".glb":
        raise SystemExit(f"Input is not an existing GLB: {source}")
    if output.suffix.lower() != ".fbx":
        raise SystemExit(f"Output must use the .fbx extension: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return source, output, report


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.actions,
        bpy.data.armatures,
        bpy.data.meshes,
        bpy.data.materials,
    ):
        for item in list(collection):
            collection.remove(item)


def _scene_contract() -> dict[str, object]:
    armatures = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"),
        key=lambda obj: obj.name,
    )
    meshes = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "MESH"),
        key=lambda obj: obj.name,
    )
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one armature, found {len(armatures)}")
    if not meshes:
        raise RuntimeError("Expected at least one mesh")
    armature = armatures[0]
    bones = list(armature.data.bones)
    parents = {bone.name: bone.parent.name if bone.parent else None for bone in bones}
    skinned_meshes = []
    for mesh in meshes:
        modifiers = [
            modifier
            for modifier in mesh.modifiers
            if modifier.type == "ARMATURE" and modifier.object == armature
        ]
        groups = {group.name for group in mesh.vertex_groups}
        influenced_bones = sorted(name for name in parents if name in groups)
        if modifiers and influenced_bones:
            skinned_meshes.append(
                {
                    "name": mesh.name,
                    "vertices": len(mesh.data.vertices),
                    "polygons": len(mesh.data.polygons),
                    "influenced_bone_count": len(influenced_bones),
                }
            )
    if not skinned_meshes:
        raise RuntimeError("No mesh is skinned to the exported armature")

    actions = []
    animated_bones: set[str] = set()
    has_root_translation = False
    bone_path = re.compile(r'^pose\.bones\["(.+)"\]\.([a-z_]+)$')
    for action in bpy.data.actions:
        slots = getattr(action, "slots", ())
        channelbags = []
        if slots and hasattr(action, "layers"):
            for layer in action.layers:
                for strip in layer.strips:
                    for slot in slots:
                        bag = strip.channelbag(slot, ensure=False)
                        if bag is not None:
                            channelbags.append(bag)
        fcurves = list(action.fcurves) if hasattr(action, "fcurves") else []
        for bag in channelbags:
            fcurves.extend(bag.fcurves)
        paths = []
        keyframes = 0
        for curve in fcurves:
            paths.append(curve.data_path)
            keyframes += len(curve.keyframe_points)
            match = bone_path.match(curve.data_path)
            if match:
                animated_bones.add(match.group(1))
                if match.group(1) == "Root" and match.group(2) == "location":
                    has_root_translation = True
        actions.append(
            {
                "name": action.name,
                "frame_start": float(action.frame_range[0]),
                "frame_end": float(action.frame_range[1]),
                "fcurves": len(fcurves),
                "keyframes": keyframes,
                "paths": sorted(set(paths)),
            }
        )
    if not actions:
        raise RuntimeError("Export contains no animation action")
    if not animated_bones:
        raise RuntimeError("Export contains no animated skeleton channels")

    return {
        "armature": armature.name,
        "bone_names": [bone.name for bone in bones],
        "bone_parents": parents,
        "bone_count": len(bones),
        "root_bones": [bone.name for bone in bones if bone.parent is None],
        "meshes": [
            {
                "name": mesh.name,
                "vertices": len(mesh.data.vertices),
                "polygons": len(mesh.data.polygons),
            }
            for mesh in meshes
        ],
        "skinned_meshes": skinned_meshes,
        "actions": actions,
        "animated_bones": sorted(animated_bones),
        "animated_bone_count": len(animated_bones),
        "has_root_translation": has_root_translation,
    }


def main() -> None:
    source, output, report_path = _arguments()
    _clear_scene()
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0
    bpy.ops.import_scene.gltf(filepath=str(source), import_shading="NORMALS")
    source_contract = _scene_contract()
    if source_contract["bone_count"] != 55:
        raise RuntimeError(
            f"Expected the AutoAnim-55 skeleton, found {source_contract['bone_count']} bones"
        )

    action_start = min(item["frame_start"] for item in source_contract["actions"])
    action_end = max(item["frame_end"] for item in source_contract["actions"])
    bpy.context.scene.frame_start = int(round(action_start))
    bpy.context.scene.frame_end = int(round(action_end))
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
        raise RuntimeError("FBX export is missing or unexpectedly small")

    _clear_scene()
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0
    bpy.ops.import_scene.fbx(filepath=str(output), automatic_bone_orientation=False)
    roundtrip = _scene_contract()
    failures = []
    if roundtrip["bone_names"] != source_contract["bone_names"]:
        failures.append("bone names or order changed")
    if roundtrip["bone_parents"] != source_contract["bone_parents"]:
        failures.append("bone hierarchy changed")
    if roundtrip["bone_count"] != 55:
        failures.append("round-trip skeleton does not contain 55 bones")
    if not roundtrip["skinned_meshes"]:
        failures.append("round-trip mesh is not skinned")
    if roundtrip["animated_bone_count"] < 55:
        failures.append("round-trip FBX does not animate all 55 bones")
    if not roundtrip["has_root_translation"]:
        failures.append("round-trip FBX has no animated root translation")

    report = {
        "schema_version": "autoanim.fbx-export-report/1.0",
        "source_glb": str(source),
        "output_fbx": str(output),
        "fps": 30,
        "frame_start": bpy.context.scene.frame_start,
        "frame_end": bpy.context.scene.frame_end,
        "source": source_contract,
        "roundtrip": roundtrip,
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("FBX verification failed: " + "; ".join(failures))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
