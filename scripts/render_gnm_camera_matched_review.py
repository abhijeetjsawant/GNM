#!/usr/bin/env python3
"""Render GNM vertices through the exact solved weak-perspective camera."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    sy, cy = math.sin(yaw), math.cos(yaw)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sr, cr = math.sin(roll), math.cos(roll)
    ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rx = np.asarray(((1, 0, 0), (0, cp, -sp), (0, sp, cp)))
    rz = np.asarray(((cr, -sr, 0), (sr, cr, 0), (0, 0, 1)))
    return rz @ rx @ ry


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def area_light(name: str, location: Vector, target: Vector, energy: float, size: float) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(light)
    light.location = location
    look_at(light, target)


def main() -> int:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(values) != 7:
        raise ValueError("Expected GLB, mesh NPZ, camera NPZ, output, FPS, width, height")
    glb, mesh_path, camera_path, output = (Path(value).resolve() for value in values[:4])
    fps, width, height = (int(value) for value in values[4:])
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(mesh_path, allow_pickle=False) as payload:
        vertices = np.asarray(payload["vertices"], dtype=np.float32)
    with np.load(camera_path, allow_pickle=False) as payload:
        cameras = np.asarray(payload["cameras"], dtype=np.float64)
    if cameras.shape != (len(vertices), 6):
        raise ValueError("camera track must have shape [frames,6]")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    bpy.ops.import_scene.gltf(filepath=str(glb))
    candidates = [
        obj for obj in scene.objects
        if obj.type == "MESH" and len(obj.data.vertices) == vertices.shape[1]
    ]
    if len(candidates) != 1:
        raise RuntimeError("Could not identify the animated GNM mesh")
    face = candidates[0]

    camera_data = bpy.data.cameras.new("GNMMeasuredCamera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("GNMMeasuredCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    z_location = float(np.max(vertices[..., 2])) + 1.0

    def update(current_scene: bpy.types.Scene) -> None:
        frame = min(max(current_scene.frame_current - 1, 0), len(vertices) - 1)
        parameters = cameras[frame]
        rotation = rotation_matrix(*parameters[:3])
        posed = vertices[frame].astype(np.float64) @ rotation.T
        face.data.vertices.foreach_set("co", posed.astype(np.float32).ravel())
        face.data.update()
        scale = float(np.exp(parameters[3]))
        cx = (0.5 * width - float(parameters[4])) / scale
        cy = (float(parameters[5]) - 0.5 * height) / scale
        camera_data.ortho_scale = height / scale
        camera.location = (cx, cy, z_location)
        look_at(camera, Vector((cx, cy, 0.0)))

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(update)
    scene.frame_start = 1
    scene.frame_end = len(vertices)
    update(scene)

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.fps = fps
    scene.render.fps_base = 1.0
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = str(output)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("GNMMeasuredWorld")
    scene.world.color = (0.008, 0.010, 0.015)
    target = Vector((0.0, 0.25, 0.08))
    area_light("Key", Vector((-0.25, 0.50, 0.50)), target, 8.0, 0.30)
    area_light("Fill", Vector((0.30, 0.32, 0.45)), target, 3.0, 0.35)
    area_light("Rim", Vector((0.0, 0.45, -0.30)), target, 5.0, 0.25)

    bpy.ops.render.render(animation=True)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("Blender did not produce the camera-matched review")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
