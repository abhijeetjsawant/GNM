"""Where does the DELIVERED MESH's face point? Measured on the shipped GLB, in Blender.

Step D1 needs one reading that no joint name can fake. Every by-name score -- the
scoreboard, I1's retarget split, I4's feet bar -- pairs `LeftFoot` with a left foot
because both are called left. If the rig's naming is mirrored relative to the mesh it
drives, all of them agree with each other and every one of them is wrong about the
surface. So this probe never reads a joint name. It picks vertex sets out of the BIND
POSE by geometry alone -- the most anterior head vertex is the nose, whatever bone it
happens to be weighted to -- carries them through the real skinning to world space, and
reports where they ended up.

    blender --background --python tools/compare/facing_surface_probe.py -- OUT.json [GLB_DIR]

Writes JSON for `tools/compare/facing_location.py` to consume. Two guards it asserts
rather than trusts: the imported vertex order must match the source asset (checked
against `neutral-body.npz` in the bind pose, sub-millimetre), and the scene fps must be
the delivered sample rate before the import, or every frame is the wrong frame
(`camera_overlay.py`'s 2026-09-02 bug).
"""
import bpy, sys, json
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
GLB_DIR = argv[1] if len(argv) > 1 else "artifacts/commercial-multiview-soma77"
# The asset the GLBs were bound to. D1 (fix) rebuilds against a relabelled asset under
# `artifacts/compare/d1-fix/body-run/`, and the vertex-order guard below compares the
# imported bind pose against it, so it must be the asset that build actually used.
ASSET = argv[2] if len(argv) > 2 else (
    ".cache/autoanim_gnm/body-provider/run/detailed-hands-fbd9784b/neutral-body.npz")  # regenerated 2026-09-02

bpy.ops.wm.read_factory_settings(use_empty=True)
FPS = int(json.load(open(f"{GLB_DIR}/subject-00.body-track.json"))["timebase"]["sample_rate_hz"])
bpy.context.scene.render.fps = FPS

asset = np.load(ASSET)
rest = asset["vertices_m"].astype(np.float64)          # rig space, Y up, +Z is the face
names = [str(v) for v in asset["joint_names"].tolist()]
dominant = asset["joint_indices"][
    np.arange(len(rest)), asset["joint_weights"].argmax(1)
]

# --- the vertex sets, chosen from the bind pose by GEOMETRY, never by name -------------
# `head` is the set of vertices whose heaviest bone is the head; within it the extreme
# +Z vertex is the nose and the extreme -Z the back of the skull. Whether that +Z end is
# "anatomically forward" is exactly the question, so nothing here assumes it.
head = np.flatnonzero(dominant == names.index("Head"))
top = rest[:, 1] > np.percentile(rest[:, 1], 55)       # upper body, for the across axis
arms = np.flatnonzero(top & (np.abs(rest[:, 0]) > 0.20))
SETS = {
    # 40 vertices each, so a single stray vertex cannot carry a direction.
    "head_plus_z": head[np.argsort(rest[head, 2])[-40:]],
    "head_minus_z": head[np.argsort(rest[head, 2])[:40]],
    "upper_plus_x": arms[np.argsort(rest[arms, 0])[-40:]],
    "upper_minus_x": arms[np.argsort(rest[arms, 0])[:40]],
}

report = {
    "instrument": "tools/compare/facing_surface_probe.py",
    "glb_dir": GLB_DIR,
    "scene_fps": FPS,
    "rest_sets_rig_space": {
        key: {"vertices": len(idx), "centroid_xyz_m": [float(v) for v in rest[idx].mean(0)]}
        for key, idx in SETS.items()
    },
    "asset": ASSET,
    "rest_reading": (
        "In the BIND pose the nose set (head_plus_z) sits at rig +Z. A body facing +Z "
        "with up +Y is right-handed only if its anatomical left is +X. Until 2026-09-02 "
        "the bones named Left sat at rig -X, so the asset's Left-named bones drove the "
        "mesh's anatomical RIGHT: that was the mirror, read off the shipped asset. "
        "`left_named_mean_x_m` below states which side this asset is on."
    ),
    "left_named_mean_x_m": float(
        rest[dominant == names.index("LeftUpperArm")][:, 0].mean()),
    "right_named_mean_x_m": float(
        rest[dominant == names.index("RightUpperArm")][:, 0].mean()),
    "subjects": {},
}

for p in (f"{GLB_DIR}/subject-00.glb", f"{GLB_DIR}/subject-01.glb"):
    bpy.ops.import_scene.gltf(filepath=p)
# The GLB also carries icosphere eye proxies; the body is the MPFB mesh, as in
# camera_overlay.py.
meshes = sorted(
    [o for o in bpy.data.objects if o.type == "MESH" and "MPFB" in o.name],
    key=lambda o: o.name,
)
if len(meshes) != 2:
    raise SystemExit(f"expected 2 delivered meshes, found {[o.name for o in meshes]}")

# Every frame: the block bootstrap needs more than two blocks to mean anything.
FRAMES = list(range(150))
for subject, obj in enumerate(meshes):
    # Vertex order guard. Blender's glTF importer converts Y-up to Z-up as
    # (x, y, z) -> (x, -z, y); undo it and the bind pose must be the source asset.
    bpy.context.scene.frame_set(0)
    bind = np.array([tuple(obj.matrix_world @ v.co) for v in obj.data.vertices])
    back = np.stack([bind[:, 0], bind[:, 2], -bind[:, 1]], axis=-1)
    if len(bind) != len(rest):
        raise SystemExit(f"vertex count {len(bind)} != asset {len(rest)}")
    drift = float(np.abs(back - rest).max())
    per_frame = {key: [] for key in SETS}
    for frame in FRAMES:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        world = np.array([tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices])
        evaluated.to_mesh_clear()
        for key, idx in SETS.items():
            per_frame[key].append([float(v) for v in world[idx].mean(0)])
    report["subjects"][f"subject_{subject:02d}"] = {
        "object": obj.name,
        "vertex_order_matches_asset_max_abs_m": drift,
        "vertex_order_guard": (
            "the delivered mesh in its bind pose, converted back to rig space, against "
            "neutral-body.npz -- if this is not ~0 the vertex sets are not the ones chosen"
        ),
        "frames": FRAMES,
        # Centroids in the capture's own Z-up world metres, one per sampled frame.
        "centroids_world_z_up_m": per_frame,
    }
    print(f"subject {subject}: {len(bind)} vertices, bind drift {drift * 1000:.4f} mm")

with open(OUT, "w") as handle:
    json.dump(report, handle, indent=2)
print("WROTE", OUT)
