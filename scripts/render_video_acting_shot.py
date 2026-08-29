#!/usr/bin/env python3
"""Render a video-captured connected character to a silent review MP4.

Usage:
  blender --background --python scripts/render_video_acting_shot.py -- \
    input.glb output.mp4 30 70 [full|upper]

The caller supplies the exact source FPS and decoded frame count so the render
cannot silently drift from the video capture clock.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def _arguments() -> tuple[Path, Path, int, int, str]:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(values) not in (4, 5):
        raise ValueError(
            "Expected input GLB, output MP4, FPS, frame count, and optional framing"
        )
    source = Path(values[0]).resolve()
    output = Path(values[1]).resolve()
    fps = int(values[2])
    frame_count = int(values[3])
    framing = values[4] if len(values) == 5 else "full"
    if not source.is_file() or source.suffix.lower() != ".glb":
        raise ValueError(f"Input must be an existing GLB: {source}")
    if output.suffix.lower() != ".mp4":
        raise ValueError(f"Output must be an MP4: {output}")
    if fps <= 0 or fps > 240 or frame_count <= 0 or frame_count > 100_000:
        raise ValueError("FPS or frame count is outside the review-render limits")
    if framing not in {"full", "upper"}:
        raise ValueError("Framing must be 'full' or 'upper'")
    output.parent.mkdir(parents=True, exist_ok=True)
    return source, output, fps, frame_count, framing


def _world_bounds() -> tuple[Vector, Vector]:
    corners: list[Vector] = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not corners:
        raise RuntimeError("Imported GLB contains no mesh")
    lower = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
    upper = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
    return lower, upper


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _area_light(
    name: str,
    location: Vector,
    target: Vector,
    energy: float,
    size: float,
) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(light)
    light.location = location
    _look_at(light, target)


def _remove_unobserved_lower_body() -> int:
    """Clip leg-skinned vertices for an explicitly waist-up source review."""

    leg_groups = {
        "LeftUpperLeg",
        "LeftLowerLeg",
        "LeftFoot",
        "LeftToes",
        "RightUpperLeg",
        "RightLowerLeg",
        "RightFoot",
        "RightToes",
    }
    removed = 0
    for obj in tuple(bpy.context.scene.objects):
        if obj.type != "MESH":
            continue
        indices = {
            group.index for group in obj.vertex_groups if group.name in leg_groups
        }
        if not indices:
            continue
        selected = [
            vertex.index
            for vertex in obj.data.vertices
            if sum(
                influence.weight
                for influence in vertex.groups
                if influence.group in indices
            )
            >= 0.5
        ]
        if not selected:
            continue
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        for index in selected:
            obj.data.vertices[index].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")
        removed += len(selected)
    if removed == 0:
        raise RuntimeError("Upper-body framing found no leg-skinned vertices to clip")
    return removed


def main() -> int:
    source, output, fps, frame_count, framing = _arguments()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    # glTF animation times are imported against the active scene FPS.
    scene.render.fps = fps
    scene.render.fps_base = 1.0
    bpy.ops.import_scene.gltf(filepath=str(source))

    if framing == "upper":
        removed = _remove_unobserved_lower_body()
        print(f"Clipped {removed} unobserved lower-body vertices")

    if not bpy.data.actions:
        raise RuntimeError("Imported GLB has no animation")
    expected_last_frame = frame_count - 1
    imported_last_frame = max(float(action.frame_range[1]) for action in bpy.data.actions)
    if not math.isclose(imported_last_frame, expected_last_frame, abs_tol=1.01):
        raise RuntimeError(
            "Imported animation does not match the source clock: "
            f"last={imported_last_frame:.3f}, expected={expected_last_frame}"
        )

    scene.frame_start = 0
    scene.frame_end = expected_last_frame
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
        scene.world = bpy.data.worlds.new("VideoActingReviewWorld")
    scene.world.color = (0.012, 0.016, 0.024)

    lower, upper = _world_bounds()
    center = (lower + upper) * 0.5
    extent = upper - lower
    scale = max(float(extent.x), float(extent.y), float(extent.z), 1.0)

    camera_data = bpy.data.cameras.new("VideoActingReviewCamera")
    camera_data.lens = 72.0
    camera = bpy.data.objects.new("VideoActingReviewCamera", camera_data)
    scene.collection.objects.link(camera)
    # AutoAnim's connected character faces +Y in its exported rest frame.
    if framing == "upper":
        # Match waist-up dialogue footage. The source contains no observable
        # leg evidence, so the review should not foreground hallucinated legs.
        upper_scale = max(float(extent.z), 1.0)
        target = Vector((center.x, center.y, lower.z + 0.73 * extent.z))
        camera.location = target + Vector((0.0, 1.15 * upper_scale, 0.01 * upper_scale))
        _look_at(camera, target)
    else:
        camera.location = center + Vector((0.0, 2.7 * scale, 0.03 * scale))
        _look_at(camera, center + Vector((0.0, 0.0, -0.03 * scale)))
    scene.camera = camera

    _area_light(
        "Key",
        center + Vector((-1.0 * scale, 1.4 * scale, 1.3 * scale)),
        center,
        energy=950.0,
        size=1.5 * scale,
    )
    _area_light(
        "Fill",
        center + Vector((1.3 * scale, 0.8 * scale, 0.5 * scale)),
        center,
        energy=380.0,
        size=1.8 * scale,
    )
    _area_light(
        "Rim",
        center + Vector((0.0, -1.3 * scale, 1.0 * scale)),
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
