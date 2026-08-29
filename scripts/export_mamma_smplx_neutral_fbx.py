#!/usr/bin/env python3
"""Export MAMMA's fitted locked-head SMPL-X identity as a neutral FBX.

Run through Blender:

    blender --background --python scripts/export_mamma_smplx_neutral_fbx.py -- \
        SMPLX_NEUTRAL.npz smplx_params_body_id-XX.npz smplx55-v1.json \
        OUTPUT.fbx REPORT.json
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np


def _arguments() -> tuple[Path, Path, Path, Path, Path]:
    try:
        separator = sys.argv.index("--")
    except ValueError as exc:
        raise SystemExit(
            "Expected MODEL.npz PARAMS.npz SKELETON.json OUTPUT.fbx REPORT.json after '--'"
        ) from exc
    values = sys.argv[separator + 1 :]
    if len(values) != 5:
        raise SystemExit(
            "Expected MODEL.npz PARAMS.npz SKELETON.json OUTPUT.fbx REPORT.json after '--'"
        )
    model, params, skeleton, output, report = (
        Path(value).expanduser().resolve() for value in values
    )
    for source in (model, params, skeleton):
        if not source.is_file():
            raise SystemExit(f"Required source file is missing: {source}")
    if output.suffix.lower() != ".fbx":
        raise SystemExit(f"Output must use the .fbx extension: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return model, params, skeleton, output, report


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


def _load_sources(
    model_path: Path, params_path: Path, skeleton_path: Path
) -> dict[str, object]:
    skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
    names = tuple(str(value) for value in skeleton["joint_names"])
    parents = np.asarray(skeleton["parents"], dtype=np.int64)
    if len(names) != 55 or parents.shape != (55,):
        raise RuntimeError("Expected the exact 55-joint SMPL-X contract")
    with np.load(model_path, allow_pickle=True) as model:
        v_template = np.asarray(model["v_template"], dtype=np.float64)
        shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
        regressor = np.asarray(model["J_regressor"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        faces = np.asarray(model["f"], dtype=np.int32)
        texture_vertices = np.asarray(model["vt"], dtype=np.float64)
        texture_faces = np.asarray(model["ft"], dtype=np.int32)
    with np.load(params_path, allow_pickle=True) as params:
        betas = np.asarray(params["smplx_betas"], dtype=np.float64).reshape(-1)
    if betas.shape != (16,):
        raise RuntimeError(f"Expected 16 fitted MAMMA betas, found {betas.shape}")
    if shapedirs.shape[:2] != v_template.shape or shapedirs.shape[2] < len(betas):
        raise RuntimeError("SMPL-X shape basis is incompatible with MAMMA betas")
    vertices = v_template + np.einsum(
        "vci,i->vc", shapedirs[:, :, : len(betas)], betas
    )
    joints = regressor @ vertices
    if vertices.shape != (10475, 3) or faces.shape != (20908, 3):
        raise RuntimeError("Locked-head SMPL-X topology differs from the expected model")
    if joints.shape != (55, 3) or weights.shape != (10475, 55):
        raise RuntimeError("Locked-head SMPL-X skeleton or weights differ")
    if texture_faces.shape != faces.shape:
        raise RuntimeError("SMPL-X UV topology differs from the geometry")
    if not all(
        np.isfinite(value).all()
        for value in (vertices, joints, weights, texture_vertices)
    ):
        raise RuntimeError("SMPL-X neutral asset contains non-finite values")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("SMPL-X skinning weights are not normalized")
    return {
        "names": names,
        "parents": parents,
        "vertices": vertices.astype(np.float32),
        "faces": faces,
        "joints": joints.astype(np.float32),
        "weights": weights.astype(np.float32),
        "texture_vertices": texture_vertices.astype(np.float32),
        "texture_faces": texture_faces,
        "betas": betas.astype(np.float32),
    }


def _preferred_child(index: int, children: list[int], joints: np.ndarray) -> int:
    if len(children) == 1:
        return children[0]
    # Branching torso joints should point along the centre chain rather than a
    # hip, clavicle, jaw, or eye branch. The most vertical child is stable for
    # the native SMPL-X T-pose.
    return max(children, key=lambda child: float(joints[child, 1]))


def _tail(index: int, parents: np.ndarray, joints: np.ndarray) -> Vector:
    children = np.flatnonzero(parents == index).astype(int).tolist()
    head = Vector(joints[index].tolist())
    if children:
        tail = Vector(joints[_preferred_child(index, children, joints)].tolist())
        if (tail - head).length > 1e-5:
            return tail
    parent = int(parents[index])
    if parent >= 0:
        direction = head - Vector(joints[parent].tolist())
    else:
        direction = Vector((0.0, 0.1, 0.0))
    if direction.length < 1e-5:
        direction = Vector((0.0, 0.03, 0.0))
    else:
        direction.normalize()
        direction *= 0.03
    return head + direction


def _build_character(data: dict[str, object]) -> tuple[bpy.types.Object, bpy.types.Object]:
    names = data["names"]
    parents = data["parents"]
    vertices = data["vertices"]
    faces = data["faces"]
    joints = data["joints"]
    weights = data["weights"]

    mesh_data = bpy.data.meshes.new("MAMMA_SMPLX_Original_Mesh")
    mesh_data.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh_data.update(calc_edges=True)
    mesh_object = bpy.data.objects.new("MAMMA_SMPLX_Original", mesh_data)
    bpy.context.collection.objects.link(mesh_object)

    uv_layer = mesh_data.uv_layers.new(name="SMPLX_UV")
    texture_vertices = data["texture_vertices"]
    texture_faces = data["texture_faces"]
    for polygon in mesh_data.polygons:
        for corner, loop_index in enumerate(polygon.loop_indices):
            uv = texture_vertices[texture_faces[polygon.index, corner]]
            uv_layer.data[loop_index].uv = (float(uv[0]), float(uv[1]))

    material = bpy.data.materials.new("MAMMA SMPL-X neutral preview")
    material.diffuse_color = (0.55, 0.30, 0.23, 1.0)
    material.roughness = 0.82
    mesh_data.materials.append(material)

    armature_data = bpy.data.armatures.new("MAMMA SMPL-X 55")
    armature = bpy.data.objects.new("MAMMA_SMPLX_55", armature_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = []
    for index, name in enumerate(names):
        bone = armature_data.edit_bones.new(name)
        bone.head = Vector(joints[index].tolist())
        bone.tail = _tail(index, parents, joints)
        bone.use_connect = False
        edit_bones.append(bone)
    for index, parent in enumerate(parents):
        if parent >= 0:
            edit_bones[index].parent = edit_bones[int(parent)]
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.data.pose_position = "REST"

    for joint_index, name in enumerate(names):
        group = mesh_object.vertex_groups.new(name=name)
        joint_weights = weights[:, joint_index]
        indices = np.flatnonzero(joint_weights > 1e-8)
        for vertex_index in indices:
            group.add([int(vertex_index)], float(joint_weights[vertex_index]), "REPLACE")
    modifier = mesh_object.modifiers.new(name="MAMMA SMPL-X Skin", type="ARMATURE")
    modifier.object = armature
    mesh_object.parent = armature
    bpy.context.view_layer.update()
    return armature, mesh_object


def _scene_contract() -> dict[str, object]:
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError(
            f"Expected one armature and one mesh, found {len(armatures)} and {len(meshes)}"
        )
    armature, mesh = armatures[0], meshes[0]
    bones = list(armature.data.bones)
    modifiers = [
        modifier
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE" and modifier.object == armature
    ]
    identity = Matrix.Identity(4)
    pose_error = max(
        max(abs(value) for row in (bone.matrix_basis - identity) for value in row)
        for bone in armature.pose.bones
    )
    return {
        "armature": armature.name,
        "mesh": mesh.name,
        "bone_count": len(bones),
        "bone_names": [bone.name for bone in bones],
        "bone_parents": {
            bone.name: bone.parent.name if bone.parent else None for bone in bones
        },
        "root_bones": [bone.name for bone in bones if bone.parent is None],
        "vertices": len(mesh.data.vertices),
        "triangles": len(mesh.data.polygons),
        "uv_layers": len(mesh.data.uv_layers),
        "vertex_groups": len(mesh.vertex_groups),
        "armature_modifiers": len(modifiers),
        "actions": len(bpy.data.actions),
        "pose_basis_error": pose_error,
    }


def main() -> None:
    model_path, params_path, skeleton_path, output, report_path = _arguments()
    _clear_scene()
    data = _load_sources(model_path, params_path, skeleton_path)
    _build_character(data)
    source_contract = _scene_contract()
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
        raise RuntimeError("Original MAMMA SMPL-X FBX is missing or unexpectedly small")

    _clear_scene()
    bpy.ops.import_scene.fbx(filepath=str(output), automatic_bone_orientation=False)
    roundtrip = _scene_contract()
    failures = []
    for key in ("bone_count", "bone_names", "bone_parents", "root_bones"):
        if roundtrip[key] != source_contract[key]:
            failures.append(f"{key} changed during FBX round trip")
    if roundtrip["vertices"] != 10475 or roundtrip["triangles"] != 20908:
        failures.append("locked-head SMPL-X topology changed")
    if roundtrip["uv_layers"] < 1:
        failures.append("SMPL-X UV map is absent")
    if roundtrip["vertex_groups"] != 55 or roundtrip["armature_modifiers"] != 1:
        failures.append("SMPL-X skinning contract changed")
    if roundtrip["actions"] != 0 or roundtrip["pose_basis_error"] > 1e-6:
        failures.append("original SMPL-X neutral FBX is not static in rest pose")

    report = {
        "schema_version": "autoanim.mamma-smplx-neutral-fbx/1.0",
        "status": "passed" if not failures else "failed",
        "output_fbx": str(output),
        "identity": "MAMMA fitted subject-00, neutral gender locked-head SMPL-X",
        "pose": "native SMPL-X T-pose with flat-hand mean",
        "gnm_included": False,
        "model_sha256": _sha256(model_path),
        "mamma_params_sha256": _sha256(params_path),
        "skeleton_contract_sha256": _sha256(skeleton_path),
        "smplx_betas": data["betas"].astype(float).tolist(),
        "source": source_contract,
        "roundtrip": roundtrip,
        "failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("MAMMA SMPL-X FBX verification failed: " + "; ".join(failures))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
