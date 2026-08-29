#!/usr/bin/env python3
"""Render the connected audio-acting GLB to a silent review MP4 in Blender.

Usage:
  blender --background --python scripts/render_audio_acting_shot.py -- input.glb output.mp4

Audio is deliberately muxed by the caller after the render, so this Blender
worker is reusable for any verified audio source and does not invent timing.
"""

from __future__ import annotations

from pathlib import Path
import sys

import bpy
from mathutils import Vector


def _arguments() -> tuple[Path, Path]:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(values) != 2:
        raise ValueError("Expected input GLB and output silent MP4 after '--'")
    source = Path(values[0]).resolve()
    output = Path(values[1]).resolve()
    if not source.is_file() or source.suffix.lower() != ".glb":
        raise ValueError(f"Input must be an existing GLB: {source}")
    if output.suffix.lower() != ".mp4":
        raise ValueError(f"Output must be an MP4: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return source, output


def _world_bounds() -> tuple[Vector, Vector]:
    corners: list[Vector] = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not corners:
        raise RuntimeError("Imported GLB contains no mesh")
    values = tuple(corners)
    lower = Vector(tuple(min(point[axis] for point in values) for axis in range(3)))
    upper = Vector(tuple(max(point[axis] for point in values) for axis in range(3)))
    return lower, upper


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _area_light(name: str, location: Vector, target: Vector, energy: float, size: float) -> None:
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = energy
    light_data.shape = "DISK"
    light_data.size = size
    light = bpy.data.objects.new(name, light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = location
    _look_at(light, target)


def main() -> int:
    source, output = _arguments()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = 210
    scene.render.fps = 30
    scene.render.fps_base = 1.0
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = str(output)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("ReviewWorld")
    scene.world.color = (0.012, 0.016, 0.024)

    lower, upper = _world_bounds()
    center = (lower + upper) * 0.5
    extent = upper - lower
    scale = max(float(extent.x), float(extent.y), float(extent.z), 1.0)

    camera_data = bpy.data.cameras.new("ReviewCamera")
    # A portrait-review focal length keeps the face legible while retaining
    # hands and feet for body-performance assessment.
    camera_data.lens = 85.0
    camera = bpy.data.objects.new("ReviewCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector((0.0, -2.55 * scale, 0.03 * scale))
    _look_at(camera, center + Vector((0.0, 0.0, -0.03 * scale)))
    scene.camera = camera

    _area_light(
        "Key",
        center + Vector((-1.0 * scale, -1.4 * scale, 1.3 * scale)),
        center,
        energy=950.0,
        size=1.5 * scale,
    )
    _area_light(
        "Fill",
        center + Vector((1.3 * scale, -0.8 * scale, 0.5 * scale)),
        center,
        energy=380.0,
        size=1.8 * scale,
    )
    _area_light(
        "Rim",
        center + Vector((0.0, 1.3 * scale, 1.0 * scale)),
        center,
        energy=720.0,
        size=1.2 * scale,
    )

    bpy.ops.render.render(animation=True)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("Blender did not produce the requested review MP4")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
