#!/usr/bin/env python3
"""Create a provider-neutral mocap FBX and verify its public naming."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import bpy


FORBIDDEN = "mamma"


def _arguments() -> tuple[Path, Path, Path, str]:
    try:
        separator = sys.argv.index("--")
    except ValueError as exc:
        raise SystemExit("Expected INPUT.fbx OUTPUT.fbx REPORT.json MODE after '--'") from exc
    values = sys.argv[separator + 1 :]
    if len(values) != 4 or values[3] not in {"neutral", "animated"}:
        raise SystemExit("Expected INPUT.fbx OUTPUT.fbx REPORT.json neutral|animated")
    source, output, report = (Path(value).expanduser().resolve() for value in values[:3])
    if not source.is_file() or source.suffix.lower() != ".fbx":
        raise SystemExit(f"Input is not an existing FBX: {source}")
    if output.suffix.lower() != ".fbx":
        raise SystemExit(f"Output must use the .fbx extension: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return source, output, report, values[3]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _character() -> tuple[bpy.types.Object, bpy.types.Object]:
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError(
            f"Expected one armature and one mesh, found {len(armatures)} and {len(meshes)}"
        )
    return armatures[0], meshes[0]


def _rename(mode: str) -> None:
    armature, mesh = _character()
    armature.name = "Mocap_Rig_55"
    armature.data.name = "Mocap_Rig_55"
    mesh.name = "Mocap_Character"
    mesh.data.name = "Mocap_Character_Mesh"
    for index, material in enumerate(mesh.data.materials):
        if material is not None:
            material.name = f"Mocap_Material_{index:02d}"
    for modifier in mesh.modifiers:
        if modifier.type == "ARMATURE":
            modifier.name = "Mocap_Skin"
    if armature.animation_data and armature.animation_data.action:
        armature.animation_data.action.name = "Mocap_Animation"
    shape_keys = mesh.data.shape_keys
    if shape_keys is not None:
        shape_keys.name = "Mocap_Correctives"
        for index, key in enumerate(shape_keys.key_blocks[1:]):
            key.name = f"Mocap_Corrective_{index:04d}"
        if shape_keys.animation_data and shape_keys.animation_data.action:
            shape_keys.animation_data.action.name = "Mocap_Corrective_Animation"
    for collection in bpy.data.collections:
        collection.name = re.sub(FORBIDDEN, "Mocap", collection.name, flags=re.IGNORECASE)
    for obj in bpy.context.scene.objects:
        for key in list(obj.keys()):
            if key == "_RNA_UI":
                continue
            value = obj[key]
            if FORBIDDEN in key.lower() or (
                isinstance(value, str) and FORBIDDEN in value.lower()
            ):
                del obj[key]
    bpy.context.scene["asset_type"] = f"mocap_character_{mode}"


def _contract() -> dict[str, object]:
    armature, mesh = _character()
    bones = list(armature.data.bones)
    shape_keys = mesh.data.shape_keys
    actions = list(bpy.data.actions)
    names = [
        armature.name,
        armature.data.name,
        mesh.name,
        mesh.data.name,
        *[material.name for material in mesh.data.materials if material],
        *[action.name for action in actions],
    ]
    if shape_keys:
        names.extend(key.name for key in shape_keys.key_blocks)
    return {
        "armature": armature.name,
        "mesh": mesh.name,
        "bones": len(bones),
        "bone_names": [bone.name for bone in bones],
        "bone_parents": {
            bone.name: bone.parent.name if bone.parent else None for bone in bones
        },
        "vertices": len(mesh.data.vertices),
        "triangles": len(mesh.data.polygons),
        "uv_layers": len(mesh.data.uv_layers),
        "vertex_groups": len(mesh.vertex_groups),
        "shape_keys": len(shape_keys.key_blocks) if shape_keys else 0,
        "actions": len(actions),
        "action_names": [action.name for action in actions],
        "forbidden_name_hits": [name for name in names if FORBIDDEN in name.lower()],
    }


def main() -> None:
    source, output, report_path, mode = _arguments()
    _clear_scene()
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0
    bpy.ops.import_scene.fbx(filepath=str(source), automatic_bone_orientation=False)
    before = _contract()
    _rename(mode)
    expected = _contract()
    if expected["forbidden_name_hits"]:
        raise RuntimeError("Provider name remains in renamed Blender data")
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
        bake_anim=mode == "animated",
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
        raise RuntimeError("Renamed mocap FBX is missing or unexpectedly small")

    payload = output.read_bytes()
    binary_hits = payload.lower().count(FORBIDDEN.encode("ascii"))
    _clear_scene()
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0
    bpy.ops.import_scene.fbx(filepath=str(output), automatic_bone_orientation=False)
    roundtrip = _contract()
    failures = []
    if binary_hits:
        failures.append("provider name remains in the FBX binary")
    if roundtrip["forbidden_name_hits"]:
        failures.append("provider name remains after FBX re-import")
    for key in ("bones", "bone_names", "bone_parents", "vertices", "triangles"):
        if roundtrip[key] != expected[key]:
            failures.append(f"{key} changed during FBX round trip")
    if mode == "neutral":
        if roundtrip["actions"] != 0 or roundtrip["shape_keys"] != 0:
            failures.append("neutral mocap FBX contains animation")
    else:
        if roundtrip["actions"] != 2 or roundtrip["shape_keys"] != 151:
            failures.append("animated mocap tracks or correctives changed")

    report = {
        "schema_version": "autoanim.mocap-fbx/1.0",
        "status": "passed" if not failures else "failed",
        "mode": mode,
        "output": output.name,
        "source_asset_sha256": _sha256(source),
        "output_sha256": _sha256(output),
        "fps": 30 if mode == "animated" else None,
        "frames": 150 if mode == "animated" else 0,
        "roundtrip": roundtrip,
        "binary_forbidden_name_hits": binary_hits,
        "failures": failures,
    }
    report_text = json.dumps(report, indent=2) + "\n"
    if FORBIDDEN in report_text.lower():
        raise RuntimeError("Provider name remains in the delivery report")
    report_path.write_text(report_text, encoding="utf-8")
    if failures:
        raise RuntimeError("Mocap FBX relabel verification failed: " + "; ".join(failures))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
