import bpy, sys, json
from mathutils import Vector, Quaternion, Matrix
argv=sys.argv[sys.argv.index("--")+1:]
OUTDIR, STEP = argv[0], int(argv[1])
import os
S = os.environ.get('FITTER_WORK', os.path.expanduser('~/.cache/autoanim-fitter'))
bpy.ops.wm.read_factory_settings(use_empty=True)
# momentum writes glTF keyframe times in SECONDS at 30 fps. The importer converts
# them to frames using the SCENE fps, so a default 24 fps scene compresses 150
# frames into 119.2 and then holds the last pose -- the animation runs 25% fast
# and the character shows a later moment than the footage under it. Set fps first.
bpy.context.scene.render.fps = 30
for i in (0,1):
    bpy.ops.import_scene.gltf(filepath=f"{S}/fitted_{i}.gltf")
# momentum's glTF export is in METRES (root rest translation reads 0.92, a hip
# height), Y-up. Blender's importer maps Y-up (X,Y,Z) to Z-up (X,-Z,Y), which is
# exactly the capture frame -- so no scale and no rotation are needed. An earlier
# 0.01 scale, assuming centimetres, shrank the character to nothing.
bpy.context.view_layer.update()
# momentum exports a marker sphere per locator; keep only the body meshes.
for o in [o for o in bpy.data.objects if o.type=="MESH" and not o.name.startswith("mesh")]:
    bpy.data.objects.remove(o, do_unlink=True)
meshes=[o for o in bpy.data.objects if o.type=="MESH"]
print("meshes:", [o.name for o in meshes])
rig=json.load(open("artifacts/soma77-full/camera-rig.json"))["cameras"]["A001"]
W,H=rig["resolution"]; fx,fy,cx,cy=rig["intrinsics"]; w,x,y,z=rig["camera_to_world_quaternion_wxyz"]
sc=bpy.context.scene
cd=bpy.data.cameras.new("cam"); cam=bpy.data.objects.new("cam",cd)
sc.collection.objects.link(cam); sc.camera=cam
M=Quaternion((w,x,y,z)).to_matrix().to_4x4() @ Matrix.Diagonal((1.0,-1.0,-1.0,1.0))
M.translation=Vector(rig["camera_center_world_m"]); cam.matrix_world=M
cd.sensor_fit="HORIZONTAL"; cd.sensor_width=36.0; cd.lens=fx*36.0/W
cd.shift_x=(W/2.0-cx)/W; cd.shift_y=(cy-H/2.0)/W
mat=bpy.data.materials.new("body"); mat.use_nodes=False; mat.diffuse_color=(0.36,0.47,0.58,1.0)
for o in meshes: o.data.materials.clear(); o.data.materials.append(mat)
sc.render.engine="BLENDER_WORKBENCH"
sh=sc.display.shading; sh.light="STUDIO"; sh.color_type="MATERIAL"; sh.show_cavity=True; sh.show_shadows=False
sc.display.render_aa="8"; sc.render.resolution_x, sc.render.resolution_y = 1280,720
sc.render.film_transparent=True
sc.world=bpy.data.worlds.new("w"); sc.world.color=(0.94,0.95,0.96)
sc.render.image_settings.file_format="PNG"; sc.render.image_settings.color_mode="RGBA"
n=0
for f in range(0,150,STEP):
    sc.frame_set(f); sc.render.filepath=f"{OUTDIR}/g_{n:03d}.png"
    bpy.ops.render.render(write_still=True); n+=1
print("RENDERED", n)
