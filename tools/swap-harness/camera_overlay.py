"""Render the delivered character from a REAL camera pose, to compare with the footage.

The validation this pipeline never had, and the one that would have caught the
facing bug on day one: put a Blender camera at a rig camera's exact extrinsics and
intrinsics, render the exported GLB, and hold it beside the source frame. Anything
that is geometrically wrong -- facing, scale, position, limb assignment -- shows up
immediately, and none of it is visible when the character is viewed in isolation,
because a wrong-but-self-consistent character still looks like a person.

    blender --background --python tools/swap-harness/camera_overlay.py -- OUT.jpg A001 0 [flip]

The optional `flip` argument rotates each character 180 degrees about its own
vertical axis, which is how the 2026-08-30 facing bug was confirmed.

TIMEBASE. glTF stores keyframe times in SECONDS and Blender's importer converts them
with the *scene* fps, so the scene fps must be set BEFORE the import. Until 2026-09-02
this script never set it: a 30 fps delivered motion landed in the 24 fps factory scene,
every overlay asked for frame F actually showed take frame 60 + 1.25*F, and anything
past F = 119 showed a frozen last pose. It is the momentum/glTF gotcha in CLAUDE.md,
sitting inside the instrument that was supposed to catch geometry errors -- and it
reads as a placement error, not a timing one. Every overlay rendered before that date,
including the ones behind the 2026-08-30 facing diagnosis, is off-frame.

The POSE CHECK printed below exists so that can never happen silently again: it holds
the posed Hips against the triangulated pelvis of the frame that was asked for, and a
timebase that is wrong by even one frame shows up in millimetres.
"""
import bpy, sys, json, math
import numpy as np
from mathutils import Vector, Quaternion, Matrix

argv = sys.argv[sys.argv.index("--")+1:]
OUT, CAMNAME, FRAME = argv[0], argv[1], int(argv[2])

bpy.ops.wm.read_factory_settings(use_empty=True)
GLB_DIR = argv[4] if len(argv) > 4 else "artifacts/commercial-multiview-soma77"

# The delivered sample rate, read from the track beside the GLB rather than assumed.
# `FPS_OVERRIDE=24` reproduces the pre-2026-09-02 behaviour for the before/after proof.
FPS = 30
try:
    FPS = int(json.load(open(f"{GLB_DIR}/subject-00.body-track.json"))["timebase"]["sample_rate_hz"])
except Exception as error:                                        # noqa: BLE001
    print("POSE CHECK: no track beside the GLB, assuming 30 fps --", error)
import os
FPS = int(os.environ.get("FPS_OVERRIDE", FPS))
bpy.context.scene.render.fps = FPS
print(f"SCENE FPS set to {FPS} before import")

for p in (f"{GLB_DIR}/subject-00.glb", f"{GLB_DIR}/subject-01.glb"):
    bpy.ops.import_scene.gltf(filepath=p)
for o in [o for o in bpy.data.objects if o.type == "MESH" and "MPFB" not in o.name]:
    bpy.data.objects.remove(o, do_unlink=True)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]

FLIP = len(argv) > 3 and argv[3] == "flip"
if FLIP:
    bpy.context.scene.frame_set(FRAME)
    bpy.context.view_layer.update()
    for a in [o for o in bpy.data.objects if o.type == "ARMATURE"]:
        dg = bpy.context.evaluated_depsgraph_get()
        ev = a.evaluated_get(dg)
        pb = ev.pose.bones.get("Hips") or ev.pose.bones[0]
        pivot = (ev.matrix_world @ pb.matrix).translation
        T = Matrix.Translation(pivot) @ Matrix.Rotation(math.pi, 4, 'Z') @ Matrix.Translation(-pivot)
        a.matrix_world = T @ a.matrix_world
    print("FLIPPED both characters 180 degrees about their own vertical axis")

rig = json.load(open("artifacts/soma77-full/camera-rig.json"))["cameras"][CAMNAME]
W, H = rig["resolution"]
fx, fy, cx, cy = rig["intrinsics"]
w, x, y, z = rig["camera_to_world_quaternion_wxyz"]

scene = bpy.context.scene
cd = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cd)
scene.collection.objects.link(cam); scene.camera = cam

# glTF import puts the character back in the capture's own Z-up world, so the rig's
# extrinsics can be used directly. The only conversion needed is the camera basis:
# ours is computer-vision (+X right, +Y DOWN, +Z forward), Blender's is (+X right,
# +Y up, -Z forward) -- a 180 degree turn about X.
R = Quaternion((w, x, y, z)).to_matrix().to_4x4()
flip = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
M = R @ flip
M.translation = Vector(rig["camera_center_world_m"])
cam.matrix_world = M

cd.sensor_fit = "HORIZONTAL"; cd.sensor_width = 36.0
cd.lens = fx * cd.sensor_width / W
cd.shift_x = (W / 2.0 - cx) / W
cd.shift_y = (cy - H / 2.0) / W

body = bpy.data.materials.new("body"); body.use_nodes = False
body.diffuse_color = (0.36, 0.47, 0.58, 1.0)
for o in meshes:
    o.data.materials.clear(); o.data.materials.append(body)

scene.render.engine = "BLENDER_WORKBENCH"
sh = scene.display.shading
sh.light = "STUDIO"; sh.color_type = "MATERIAL"; sh.show_cavity = True
scene.display.render_aa = "8"
scene.render.resolution_x, scene.render.resolution_y = 960, 540
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("w"); scene.world.color = (0.94, 0.95, 0.96)
scene.render.image_settings.file_format = "JPEG"
scene.render.image_settings.quality = 88
scene.frame_set(FRAME)
bpy.context.view_layer.update()

# POSE CHECK -- is the pose on screen the pose that was asked for?
# The glTF import puts the character back in the capture's own Z-up world, so the posed
# Hips must sit on the triangulated pelvis of frame FRAME. Under the old 24 fps scene
# this was tens of millimetres to a third of a metre out, and it looked like a
# retarget error. It is the one check that separates "wrong geometry" from "wrong frame".
def _project(point_world):
    relative = cam.matrix_world.inverted() @ Vector(point_world)
    x, y, z = relative.x, relative.y, -relative.z            # Blender camera looks down -Z
    if z <= 1e-9:
        return None
    return (fx * x / z + cx, -fy * y / z + cy)

for index, armature in enumerate(sorted(
    [o for o in bpy.data.objects if o.type == "ARMATURE"], key=lambda o: o.name
)):
    try:
        track = np.load(f"{GLB_DIR}/subject-{index:02d}.body-track.npz")
        # `root` is index 8 of the 19-joint contract, not index 0. Read it from the
        # run report rather than repeating the literal.
        joints = json.load(open(f"{GLB_DIR}/run-report.json"))["joint_names"]
        pelvis = track["triangulated_world_positions_z_up_m"][FRAME, joints.index("root")]
    except Exception as error:                                    # noqa: BLE001
        print("POSE CHECK: skipped --", error)
        break
    evaluated = armature.evaluated_get(bpy.context.evaluated_depsgraph_get())
    bone = evaluated.pose.bones.get("Hips") or evaluated.pose.bones[0]
    posed = evaluated.matrix_world @ bone.head
    error_mm = (Vector(tuple(float(v) for v in pelvis)) - posed).length * 1000.0
    a, b = _project(posed), _project(tuple(float(v) for v in pelvis))
    px = math.hypot(a[0] - b[0], a[1] - b[1]) if a and b else float("nan")
    print(f"POSE CHECK subject-{index:02d} frame {FRAME}: posed Hips vs triangulated "
          f"pelvis  {error_mm:8.1f} mm  {px:7.1f} px")

scene.render.filepath = OUT
bpy.ops.render.render(write_still=True)
print("CAMERA", CAMNAME, "lens", round(cd.lens,2), "mm  at", tuple(round(v,3) for v in cam.location))
print("WROTE", OUT)
