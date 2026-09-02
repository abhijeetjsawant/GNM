"""Export the DELIVERED GLB's posed mesh, per frame, in the capture's world frame.

Companion to `tools/compare/silhouette.py` (step I6). Blender's Python has no cv2 and
no scipy, so the render path and the scoring path cannot live in one process: this
script runs inside Blender and writes an npz that `silhouette.py` rasterises.

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python tools/compare/blender_export_mesh.py -- OUT.npz GLB_DIR [FPS]

**`scene.render.fps` is set BEFORE the import, and that is not optional.** momentum
writes glTF keyframe times in seconds and Blender converts them with the *scene* fps.
The delivered motion is 30 fps; imported into the factory-default 24 fps scene the
action arrives as frames 0..119.2 -- 150 frames of motion crushed into 120, running
25 % fast and frozen after the last converted frame. `tools/swap-harness/camera_overlay.py`
does not set it, so every overlay it has ever rendered for frame F is showing take
frame 60 + 1.25*F. Verified here: 24 fps gives a 0.0..119.2 action range, 30 fps gives
0.0..149.0 for the same file.

The mesh selection follows `camera_overlay.py`: keep the MPFB body, drop everything
else (the GLB also carries an Icosphere). Vertices come from the evaluated depsgraph
(armature deform applied) and are pushed through the object's world matrix, so the
output is the capture's own Z-up metric world -- the frame the camera rig's extrinsics
are expressed in.
"""

import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
GLB_DIR = argv[1] if len(argv) > 1 else "artifacts/commercial-multiview-soma77"
FPS = int(argv[2]) if len(argv) > 2 else 30

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = FPS  # BEFORE the import. See the module docstring.

out: dict[str, np.ndarray] = {}
ranges = []
for subject in (0, 1):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = FPS
    bpy.ops.import_scene.gltf(filepath=f"{GLB_DIR}/subject-{subject:02d}.glb")
    for o in [o for o in bpy.data.objects if o.type == "MESH" and "MPFB" not in o.name]:
        bpy.data.objects.remove(o, do_unlink=True)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if len(meshes) != 1:
        raise SystemExit(f"expected one MPFB mesh, found {[m.name for m in meshes]}")
    mesh = meshes[0]
    action_range = [tuple(a.frame_range) for a in bpy.data.actions]
    ranges.append(action_range)
    first, last = action_range[0]
    frames = int(round(last - first)) + 1
    sc = bpy.context.scene

    faces = None
    verts = None
    for i, f in enumerate(range(int(first), int(first) + frames)):
        sc.frame_set(f)
        evaluated = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
        data = evaluated.to_mesh()
        if faces is None:
            sizes = {len(p.vertices) for p in data.polygons}
            if sizes != {3}:
                raise SystemExit(f"mesh is not triangulated: polygon sizes {sizes}")
            faces = np.array([tuple(p.vertices) for p in data.polygons], dtype=np.int32)
            verts = np.empty((frames, len(data.vertices), 3), dtype=np.float32)
        flat = np.empty(len(data.vertices) * 3, dtype=np.float64)
        data.vertices.foreach_get("co", flat)
        local = flat.reshape(-1, 3)
        matrix = np.array(evaluated.matrix_world, dtype=np.float64)
        verts[i] = (local @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)
        evaluated.to_mesh_clear()

    out[f"verts_{subject:02d}"] = verts
    out[f"faces_{subject:02d}"] = faces
    print(f"subject {subject:02d}: {verts.shape} verts, {faces.shape} faces, action {action_range}")

out["fps"] = np.asarray(FPS)
np.savez_compressed(OUT, **out)
print("WROTE", OUT, "action ranges", ranges)
