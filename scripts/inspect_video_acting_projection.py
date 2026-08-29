#!/usr/bin/env python3
"""Print rendered arm-joint pixel positions for a connected-character GLB."""

from __future__ import annotations

from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.render_video_acting_shot import (
    _look_at,
    _remove_unobserved_lower_body,
    _world_bounds,
)


def main() -> int:
    values = sys.argv[sys.argv.index("--") + 1 :]
    if len(values) != 2:
        raise ValueError("Expected connected GLB and frame index")
    source = Path(values[0]).resolve()
    frame = int(values[1])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 25
    bpy.ops.import_scene.gltf(filepath=str(source))
    _remove_unobserved_lower_body()
    lower, upper = _world_bounds()
    center = (lower + upper) * 0.5
    extent = upper - lower
    upper_scale = max(float(extent.z), 1.0)
    target = Vector((center.x, center.y, lower.z + 0.73 * extent.z))
    camera_data = bpy.data.cameras.new("ProjectionCamera")
    camera_data.lens = 72.0
    camera = bpy.data.objects.new("ProjectionCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = target + Vector((0.0, 1.15 * upper_scale, 0.01 * upper_scale))
    _look_at(camera, target)
    scene.camera = camera
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.frame_set(frame)
    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one armature, found {len(armatures)}")
    armature = armatures[0]
    for name in (
        "LeftUpperArm",
        "LeftLowerArm",
        "LeftHand",
        "RightUpperArm",
        "RightLowerArm",
        "RightHand",
    ):
        bone = armature.pose.bones[name]
        world = armature.matrix_world @ bone.head
        ndc = world_to_camera_view(scene, camera, world)
        print(f"{name} {ndc.x * 640.0:.3f} {(1.0 - ndc.y) * 640.0:.3f} depth={ndc.z:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
