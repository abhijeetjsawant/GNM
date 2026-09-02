"""D1 (fix): render one hand, close enough to read the fingers. Before and after.

WHY. `FINGER_REST_CURL_DEG` is a POSE, not a measurement, and its side sign moved with the
skeleton in this repair. No gate in this lane can see a pose: the curl produces ZERO joint
displacement (`tests/test_facing_fix.py` measures it), so a joint score is blind, and the
hands are a few pixels across in a full-body camera overlay, so a silhouette is blind too.
The only instrument for a pose is a picture, and it has to be close enough to read.

The import path is `camera_overlay.py`'s, deliberately and for its one repaired property:
THE SCENE FPS IS SET BEFORE THE IMPORT. glTF stores keyframe times in seconds and Blender
converts them with the scene fps, so a 30 fps motion in the 24 fps factory scene shows the
wrong frame -- and a wrong frame is a wrong hand pose, which is exactly what this is
looking at. The camera is placed from the hand's own bone frame rather than from a rig
camera, because a rig camera cannot see a hand.

    blender --background --python tools/compare/hand_closeup.py -- OUT.jpg GLB_DIR SUBJECT SIDE FRAME
"""
import bpy, sys, json, math
import numpy as np
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
OUT, GLB_DIR, SUBJECT, SIDE = argv[0], argv[1], int(argv[2]), argv[3]
FRAME = int(argv[4]) if len(argv) > 4 else 75

bpy.ops.wm.read_factory_settings(use_empty=True)
FPS = int(json.load(open(f"{GLB_DIR}/subject-00.body-track.json"))["timebase"]["sample_rate_hz"])
bpy.context.scene.render.fps = FPS
print(f"SCENE FPS set to {FPS} before import")

bpy.ops.import_scene.gltf(filepath=f"{GLB_DIR}/subject-{SUBJECT:02d}.glb")
for o in [o for o in bpy.data.objects if o.type == "MESH" and "MPFB" not in o.name]:
    bpy.data.objects.remove(o, do_unlink=True)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")

scene = bpy.context.scene
scene.frame_set(FRAME)
bpy.context.view_layer.update()
evaluated = armature.evaluated_get(bpy.context.evaluated_depsgraph_get())

# The hand's own frame, read off the posed bone rather than guessed. `matrix` columns are
# the bone's local axes in armature space; Blender's bone Y is its length axis, so the rig
# axes are recovered from the CHILD bone directions instead: the middle finger gives the
# bone's long axis and the index-to-little spread gives the axis across the knuckles.
pose = evaluated.pose.bones
hand = pose[f"{SIDE}Hand"]
world = evaluated.matrix_world
origin = world @ hand.head
along = (world @ pose[f"{SIDE}MiddleProximal"].head) - origin        # out along the fingers
across = ((world @ pose[f"{SIDE}IndexProximal"].head)
          - (world @ pose[f"{SIDE}LittleProximal"].head))            # across the knuckles
along.normalize()
across = (across - along * across.dot(along)); across.normalize()
palm = along.cross(across)                                           # the palm normal
tip = world @ pose[f"{SIDE}MiddleDistal"].head
centre = (origin + tip) * 0.5
span = max((origin - tip).length, 0.06)

# Look along the hand's own palm normal, from ~4 hand-spans away -- but from whichever
# END of that normal is OUTSIDE the body. The normal's sign is `along x across`, and the
# repair swaps which physical knuckle answers to `Index`, so that sign flips across the
# repair: aiming at a fixed sign put the AFTER camera inside the torso and rendered a flat
# grey field. Choosing the end farther from the chest keeps the hand visible on both arms,
# and the line below records WHICH anatomical face was seen so the pair is not read as if
# it were the same one.
DISTANCE = span * 4.0
outward = (world @ hand.head) - (world @ pose["UpperChest"].head)
face = "palm" if palm.dot(outward) > 0 else "back-of-hand"
normal = palm if palm.dot(outward) > 0 else -palm
eye = centre + normal * DISTANCE
forward = (centre - eye).normalized()
up_hint = across
right = forward.cross(up_hint); right.normalize()
up = right.cross(forward)
basis = Matrix((right, up, -forward)).transposed().to_4x4()
basis.translation = eye

data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", data)
scene.collection.objects.link(cam); scene.camera = cam
cam.matrix_world = basis
data.lens = 85.0
data.sensor_width = 36.0
data.clip_start = 0.001

body = bpy.data.materials.new("body"); body.use_nodes = False
body.diffuse_color = (0.36, 0.47, 0.58, 1.0)
for o in meshes:
    o.data.materials.clear(); o.data.materials.append(body)

scene.render.engine = "BLENDER_WORKBENCH"
shading = scene.display.shading
shading.light = "STUDIO"; shading.color_type = "MATERIAL"; shading.show_cavity = True
scene.display.render_aa = "32"
scene.render.resolution_x, scene.render.resolution_y = 900, 900
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("w"); scene.world.color = (0.94, 0.95, 0.96)
scene.render.image_settings.file_format = "JPEG"
scene.render.image_settings.quality = 92

# POSE CHECK, the same idea as camera_overlay's: state the geometry that is being looked
# at, so a wrong frame or a wrong hand cannot pass silently.
print(f"HAND CHECK subject-{SUBJECT:02d} {SIDE} frame {FRAME}: seeing the {face}, "
      f"hand->middle-tip span {span * 1000:.1f} mm, camera {DISTANCE * 1000:.0f} mm away "
      f"along {tuple(round(v, 3) for v in normal)}")
scene.render.filepath = OUT
bpy.ops.render.render(write_still=True)
print("WROTE", OUT)
