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
"""
import bpy, sys, json, math
from mathutils import Vector, Quaternion, Matrix

argv = sys.argv[sys.argv.index("--")+1:]
OUT, CAMNAME, FRAME = argv[0], argv[1], int(argv[2])

bpy.ops.wm.read_factory_settings(use_empty=True)
for p in ("artifacts/commercial-multiview-soma77/subject-00.glb",
          "artifacts/commercial-multiview-soma77/subject-01.glb"):
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
scene.render.filepath = OUT
bpy.ops.render.render(write_still=True)
print("CAMERA", CAMNAME, "lens", round(cd.lens,2), "mm  at", tuple(round(v,3) for v in cam.location))
print("WROTE", OUT)
