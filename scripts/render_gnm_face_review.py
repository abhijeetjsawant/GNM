#!/usr/bin/env python3
"""Render per-frame GNM vertices as a locked face close-up in Blender."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector


def _arguments() -> tuple[Path, Path, Path, int]:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(values) != 4:
        raise ValueError("Expected neutral GLB, render NPZ, output MP4, and FPS")
    glb, arrays, output = (Path(value).resolve() for value in values[:3])
    fps = int(values[3])
    if not glb.is_file() or not arrays.is_file() or fps <= 0:
        raise ValueError("Invalid face-review input")
    output.parent.mkdir(parents=True, exist_ok=True)
    return glb, arrays, output, fps


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _area_light(name: str, location: Vector, target: Vector, energy: float, size: float) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(light)
    light.location = location
    _look_at(light, target)


def main() -> int:
    glb, arrays_path, output, fps = _arguments()
    with np.load(arrays_path, allow_pickle=False) as values:
        vertices = np.asarray(values["vertices"], dtype=np.float32)
        colors = (
            np.asarray(values["colors"], dtype=np.float32)
            if "colors" in values.files
            else None
        )
    if vertices.ndim != 3 or vertices.shape[2] != 3 or len(vertices) < 1:
        raise ValueError("render vertices must have shape [frames,vertices,3]")
    if colors is not None and (
        colors.shape != (vertices.shape[1], 3) or not np.isfinite(colors).all()
    ):
        raise ValueError("render colors must have shape [vertices,3]")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.render.fps_base = 1.0
    bpy.ops.import_scene.gltf(filepath=str(glb))
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    candidates = [obj for obj in meshes if len(obj.data.vertices) == vertices.shape[1]]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one {vertices.shape[1]}-vertex GNM mesh; found {len(candidates)}"
        )
    face = candidates[0]
    if not face.data.materials:
        if colors is None:
            raise ValueError("untextured review meshes require per-vertex colors")
        color_attribute = face.data.color_attributes.new(
            name="GNMColor", type="FLOAT_COLOR", domain="POINT"
        )
        rgba = np.column_stack((np.clip(colors, 0.0, 1.0), np.ones(len(colors))))
        color_attribute.data.foreach_set("color", rgba.ravel())
        material = bpy.data.materials.new("GNMAnatomy")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled = nodes.get("Principled BSDF")
        vertex_color = nodes.new("ShaderNodeVertexColor")
        vertex_color.layer_name = "GNMColor"
        links.new(vertex_color.outputs["Color"], principled.inputs["Base Color"])
        principled.inputs["Roughness"].default_value = 0.62
        face.data.materials.append(material)

    def update_mesh(current_scene: bpy.types.Scene) -> None:
        # Blender's animation renderer is reliably inclusive for 1-based frame
        # ranges.  Keep the source array zero-based by subtracting one here.
        frame = min(max(current_scene.frame_current - 1, 0), len(vertices) - 1)
        face.data.vertices.foreach_set("co", vertices[frame].ravel())
        face.data.update()

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(update_mesh)
    scene.frame_start = 1
    scene.frame_end = len(vertices)
    update_mesh(scene)

    all_points = vertices[0]
    lower = Vector(tuple(float(np.min(all_points[:, axis])) for axis in range(3)))
    upper = Vector(tuple(float(np.max(all_points[:, axis])) for axis in range(3)))
    center = (lower + upper) * 0.5
    extent = upper - lower
    face_height = max(float(extent.y), 1.0e-3)

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = str(output)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("GNMFaceReviewWorld")
    scene.world.color = (0.008, 0.010, 0.015)

    camera_data = bpy.data.cameras.new("GNMFaceReviewCamera")
    camera_data.lens = 72.0
    camera = bpy.data.objects.new("GNMFaceReviewCamera", camera_data)
    scene.collection.objects.link(camera)
    # This renderer validates facial performance, so frame the measured face
    # rather than the unmeasured neck/rear scalp that dominate a full-head shot.
    target = Vector((center.x, center.y + 0.08 * face_height, center.z))
    camera.location = target + Vector((0.0, 0.0, 1.25 * face_height))
    _look_at(camera, target)
    scene.camera = camera

    _area_light(
        "Key",
        target + Vector((-0.65 * face_height, 0.75 * face_height, 1.25 * face_height)),
        target,
        8.0,
        0.8 * face_height,
    )
    _area_light(
        "Fill",
        target + Vector((0.75 * face_height, 0.25 * face_height, 1.0 * face_height)),
        target,
        3.0,
        1.0 * face_height,
    )
    _area_light(
        "Rim",
        target + Vector((0.0, 0.6 * face_height, -1.0 * face_height)),
        target,
        5.0,
        0.7 * face_height,
    )

    bpy.ops.render.render(animation=True)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("Blender did not produce the face review")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
