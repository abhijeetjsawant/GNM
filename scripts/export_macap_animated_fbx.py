#!/usr/bin/env python3
"""Apply one MAMMA SMPL-X take to the symmetric macap base skeleton.

The rig is the mean-shape symmetrized base model, not the per-subject fitted
body, so limb lengths are generic while the motion is the captured performance.
Joint rotations come from MAMMA's 165-value SMPL-X pose; jaw and eye slots are
zeroed because Google GNM owns facial motion in the assembled character.

Pose correctives are not baked, matching the base model's declared contract, so
the delivered surface is plain linear-blend skinning.

Run through Blender:

    blender --background --python scripts/export_macap_animated_fbx.py -- \
        SMPLX_NEUTRAL.npz smplx55-v1.json PARAMS.npz VERTICES.npz \
        OUTPUT.fbx REPORT.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_mamma_smplx_animated_fbx import _linearize_action, _rodrigues  # noqa: E402
from export_mamma_smplx_neutral_fbx import _clear_scene, _scene_contract, _sha256  # noqa: E402
from export_macap_base_model_fbx import (  # noqa: E402
    CHARACTER_NAME,
    FORBIDDEN_BINARY_LABELS,
    MAX_INFLUENCES,
    _build_character,
    _load_sources,
)


SAMPLE_RATE_HZ = 30
# An occluded joint can leave MAMMA's fit unconstrained, and the solve then
# walks the joint around a full loop that returns almost to where it started.
# A run of elevated angular velocity containing at least one frame above the
# trigger is replaced by a slerp between the good frames bracketing it, which
# keeps the endpoints the capture actually observed and deletes the loop.
SPIN_TRIGGER_DEG_PER_FRAME = 30.0
SPIN_SETTLE_DEG_PER_FRAME = 5.0
# A genuine fast action would not hold elevated velocity across most of a take;
# refusing beyond this fraction stops the repair silently flattening real motion.
SPIN_MAX_WINDOW_FRACTION = 0.5
# SMPL-X pose slots owned by the facial performance rather than by body capture.
FACE_POSE_SLICE = slice(22, 25)
# MAMMA's calibrated capture world is Z-up; AutoAnim runtime space is Y-up.
SOURCE_TO_Y_UP = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)), dtype=np.float64
)


def _arguments() -> tuple[Path, Path, Path, Path, Path, Path]:
    try:
        separator = sys.argv.index("--")
    except ValueError as exc:
        raise SystemExit(
            "Expected MODEL SKELETON PARAMS VERTICES OUTPUT.fbx REPORT.json after '--'"
        ) from exc
    values = sys.argv[separator + 1 :]
    if len(values) != 6:
        raise SystemExit(
            "Expected MODEL SKELETON PARAMS VERTICES OUTPUT.fbx REPORT.json after '--'"
        )
    model, skeleton, params, vertices, output, report = (
        Path(value).expanduser().resolve() for value in values
    )
    for source in (model, skeleton, params, vertices):
        if not source.is_file():
            raise SystemExit(f"Required source file is missing: {source}")
    if output.suffix.lower() != ".fbx":
        raise SystemExit(f"Output must use the .fbx extension: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return model, skeleton, params, vertices, output, report


def _axis_angle_to_quaternion(values: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(values, axis=-1, keepdims=True)
    scale = np.where(angle < 1e-12, 0.5, np.sin(0.5 * angle) / np.where(angle < 1e-12, 1.0, angle))
    return np.concatenate((values * scale, np.cos(0.5 * angle)), axis=-1)


def _quaternion_to_axis_angle(values: np.ndarray) -> np.ndarray:
    quaternion = values / np.linalg.norm(values, axis=-1, keepdims=True)
    vector = quaternion[..., :3]
    sine = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sine, np.abs(quaternion[..., 3:4]))
    sign = np.where(quaternion[..., 3:4] < 0.0, -1.0, 1.0)
    scale = np.where(sine < 1e-12, 2.0, angle / np.where(sine < 1e-12, 1.0, sine))
    return vector * sign * scale


def _slerp(start: np.ndarray, end: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    first, second = start.copy(), end.copy()
    if float(first @ second) < 0.0:
        second = -second
    dot = float(np.clip(first @ second, -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        result = first[None, :] + fractions[:, None] * (second - first)[None, :]
    else:
        omega = np.arccos(dot)
        sine = np.sin(omega)
        result = (
            np.sin((1.0 - fractions)[:, None] * omega) * first[None, :]
            + np.sin(fractions[:, None] * omega) * second[None, :]
        ) / sine
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _relative_angles_deg(rotations: np.ndarray) -> np.ndarray:
    """Frame-to-frame rotation magnitude per joint, in degrees."""

    relative = np.einsum("fjab,fjcb->fjac", rotations[1:], rotations[:-1])
    trace = np.clip((np.trace(relative, axis1=2, axis2=3) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def _repair_spins(
    effective_pose: np.ndarray, names: tuple[str, ...]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Slerp across runs where a joint's fit loops instead of holding still."""

    repaired = np.array(effective_pose, dtype=np.float64, copy=True)
    deltas = _relative_angles_deg(_rodrigues(repaired))
    frames = len(repaired)
    report: list[dict[str, object]] = []
    for joint in range(repaired.shape[1]):
        series = deltas[:, joint]
        elevated = series > SPIN_SETTLE_DEG_PER_FRAME
        start = 0
        while start < len(series):
            if not elevated[start]:
                start += 1
                continue
            end = start
            while end + 1 < len(series) and elevated[end + 1]:
                end += 1
            run = series[start : end + 1]
            if run.max() > SPIN_TRIGGER_DEG_PER_FRAME:
                # Deltas start..end connect frames start..end+1; both ends are
                # the last frames the solve tracked before it drifted.
                low, high = start, end + 1
                span = high - low + 1
                if span > SPIN_MAX_WINDOW_FRACTION * frames:
                    raise RuntimeError(
                        f"{names[joint]} spin window spans {span} of {frames} frames; "
                        "refusing to overwrite most of the take"
                    )
                anchor_low = _axis_angle_to_quaternion(repaired[low, joint])
                anchor_high = _axis_angle_to_quaternion(repaired[high, joint])
                fractions = np.arange(1, high - low) / float(high - low)
                repaired[low + 1 : high, joint] = _quaternion_to_axis_angle(
                    _slerp(anchor_low, anchor_high, fractions)
                )
                after = _relative_angles_deg(_rodrigues(repaired[:, joint : joint + 1]))[:, 0]
                net = 2.0 * np.degrees(
                    np.arccos(min(1.0, abs(float(anchor_low @ anchor_high))))
                )
                report.append(
                    {
                        "joint": names[joint],
                        "frames_replaced": [low + 2, high],
                        "window_frames": [low + 1, high + 1],
                        "path_length_before_deg": float(series[low:high].sum()),
                        "path_length_after_deg": float(after[low:high].sum()),
                        "net_displacement_deg": float(net),
                        "peak_delta_before_deg": float(run.max()),
                        "peak_delta_after_deg": float(after[low:high].max()),
                    }
                )
            start = end + 1
    return repaired, report


def _motion(
    model_path: Path,
    params_path: Path,
    vertices_path: Path,
    data: dict[str, object],
) -> dict[str, object]:
    """Forward-kinematic the MAMMA take onto the symmetric rest skeleton."""

    # allow_pickle is required because both released archives store an object
    # array: the SMPL-X model's ``joint2num`` and MAMMA's ``v_template_pred``.
    # Both inputs are local pipeline artifacts whose SHA-256 the report records;
    # the vertex archive below needs no pickling and is loaded without it.
    with np.load(model_path, allow_pickle=True) as model:
        left_hand_mean = np.asarray(model["hands_meanl"], dtype=np.float64).reshape(15, 3)
        right_hand_mean = np.asarray(model["hands_meanr"], dtype=np.float64).reshape(15, 3)
    with np.load(params_path, allow_pickle=True) as params:
        pose = np.asarray(params["smplx_pose"], dtype=np.float64)
        translation = np.asarray(params["smplx_translation"], dtype=np.float64)
    with np.load(vertices_path, allow_pickle=False) as vertices:
        fitted_joints = np.asarray(vertices["pred_joints"], dtype=np.float64)[:, :55]

    frames = len(pose)
    if pose.shape != (frames, 165) or translation.shape != (frames, 3):
        raise RuntimeError("MAMMA motion arrays have invalid shapes")
    if fitted_joints.shape != (frames, 55, 3):
        raise RuntimeError("MAMMA joint archive does not match the pose archive")
    if not (np.isfinite(pose).all() and np.isfinite(translation).all()):
        raise RuntimeError("MAMMA motion arrays contain non-finite values")

    effective_pose = pose.reshape(frames, 55, 3).copy()
    # This MAMMA run constructs the neutral model with flat_hand_mean=False;
    # SMPL-X therefore adds its native MANO means before Rodrigues conversion.
    effective_pose[:, 25:40] += left_hand_mean[None, ...]
    effective_pose[:, 40:55] += right_hand_mean[None, ...]
    face_pose_magnitude = float(np.abs(effective_pose[:, FACE_POSE_SLICE]).max())
    effective_pose[:, FACE_POSE_SLICE] = 0.0
    effective_pose, spin_repairs = _repair_spins(effective_pose, data["names"])

    local_rotations = _rodrigues(effective_pose)
    joints = np.asarray(data["joints"], dtype=np.float64)
    parents = np.asarray(data["parents"], dtype=np.int64)
    global_rotations = np.empty_like(local_rotations)
    global_positions = np.empty((frames, 55, 3), dtype=np.float64)
    for frame in range(frames):
        for joint in range(55):
            parent = int(parents[joint])
            if parent < 0:
                global_rotations[frame, joint] = local_rotations[frame, joint]
                global_positions[frame, joint] = joints[joint]
            else:
                global_rotations[frame, joint] = (
                    global_rotations[frame, parent] @ local_rotations[frame, joint]
                )
                global_positions[frame, joint] = (
                    global_positions[frame, parent]
                    + global_rotations[frame, parent] @ (joints[joint] - joints[parent])
                )
        global_positions[frame] += translation[frame]

    # The fitted MAMMA joints belong to that subject's betas. This rig is the
    # mean shape, so the difference measures body proportion, not solve error.
    shape_deviation = np.linalg.norm(global_positions - fitted_joints, axis=-1)

    root_origin = SOURCE_TO_Y_UP @ global_positions[0, 0]
    canonical_positions = (
        np.einsum("ab,fjb->fja", SOURCE_TO_Y_UP, global_positions) - root_origin
    )
    canonical_rotations = np.einsum(
        "ab,fjbc->fjac", SOURCE_TO_Y_UP, global_rotations
    )
    return {
        "frames": frames,
        "canonical_positions": canonical_positions,
        "canonical_rotations": canonical_rotations,
        "root_travel_m": float(
            np.linalg.norm(canonical_positions[-1, 0] - canonical_positions[0, 0])
        ),
        "mean_shape_vs_fitted_joint_deviation_m": float(shape_deviation.max()),
        "mean_shape_vs_fitted_joint_deviation_mean_m": float(shape_deviation.mean()),
        "discarded_face_pose_magnitude_rad": face_pose_magnitude,
        "spin_repairs": spin_repairs,
        "worst_frame_delta_after_repair_deg": float(
            _relative_angles_deg(local_rotations).max()
        ),
    }


def _animate(
    armature: bpy.types.Object, data: dict[str, object], motion: dict[str, object]
) -> None:
    """Key every joint from the precomputed world matrices, verifying each frame."""

    names = data["names"]
    parents = np.asarray(data["parents"], dtype=np.int64)
    positions = motion["canonical_positions"]
    rotations = motion["canonical_rotations"]
    frames = int(motion["frames"])

    scene = bpy.context.scene
    scene.render.fps = SAMPLE_RATE_HZ
    scene.render.fps_base = 1.0
    # Blender's FBX round trip maps source frame 0 to imported frame 1. Author
    # at 0..N-1 so the delivered FBX opens on the conventional 1..N range.
    scene.frame_start = 0
    scene.frame_end = frames - 1

    rest_matrices = {bone.name: bone.matrix_local.copy() for bone in armature.data.bones}
    armature.data.pose_position = "POSE"
    previous_quaternions: dict[str, object] = {}
    for frame in range(frames):
        scene.frame_set(frame)
        desired: dict[str, Matrix] = {}
        for joint, name in enumerate(names):
            rest_orientation = np.asarray(rest_matrices[name].to_3x3(), dtype=np.float64)
            oriented = rotations[frame, joint] @ rest_orientation
            desired[name] = Matrix.Translation(
                Vector(positions[frame, joint].tolist())
            ) @ Matrix(oriented.tolist()).to_4x4()
        for joint, name in enumerate(names):
            pose_bone = armature.pose.bones[name]
            pose_bone.rotation_mode = "QUATERNION"
            parent_index = int(parents[joint])
            if parent_index < 0:
                basis = rest_matrices[name].inverted() @ desired[name]
            else:
                parent_name = names[parent_index]
                rest_relative = rest_matrices[parent_name].inverted() @ rest_matrices[name]
                desired_relative = desired[parent_name].inverted() @ desired[name]
                basis = rest_relative.inverted() @ desired_relative
            pose_bone.matrix_basis = basis
            quaternion = pose_bone.rotation_quaternion.copy()
            previous = previous_quaternions.get(name)
            if previous is not None and quaternion.dot(previous) < 0.0:
                pose_bone.rotation_quaternion = -quaternion
                quaternion = pose_bone.rotation_quaternion.copy()
            previous_quaternions[name] = quaternion
            pose_bone.keyframe_insert(data_path="location", frame=frame, group=name)
            pose_bone.keyframe_insert(
                data_path="rotation_quaternion", frame=frame, group=name
            )
            pose_bone.keyframe_insert(data_path="scale", frame=frame, group=name)
        bpy.context.view_layer.update()
        actual = np.asarray(
            [list(armature.pose.bones[name].matrix.translation) for name in names],
            dtype=np.float64,
        )
        error = np.linalg.norm(actual - positions[frame], axis=1)
        if float(error.max()) > 1e-5:
            worst = int(np.argmax(error))
            raise RuntimeError(
                f"Blender pose assignment differs at frame {frame} on "
                f"{names[worst]} by {float(error[worst]):.6f} m"
            )
    _linearize_action(armature.animation_data.action)
    armature.animation_data.action.name = f"{CHARACTER_NAME}-Take"
    scene.frame_set(0)


def _forbidden_labels(path: Path) -> list[str]:
    payload = path.read_bytes()
    return [label.decode("ascii") for label in FORBIDDEN_BINARY_LABELS if label in payload]


def main() -> None:
    model_path, skeleton_path, params_path, vertices_path, output, report_path = _arguments()
    _clear_scene()
    data = _load_sources(model_path, skeleton_path, "symmetric")
    _build_character(data, mirror_rolls=True)
    armature = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"][0]
    motion = _motion(model_path, params_path, vertices_path, data)
    _animate(armature, data, motion)
    source_contract = _scene_contract()

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=str(output),
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        use_mesh_modifiers=True,
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        axis_forward="-Z",
        axis_up="Y",
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode="AUTO",
        embed_textures=False,
    )
    if not output.is_file() or output.stat().st_size < 1_000_000:
        raise RuntimeError("macap animated FBX is missing or unexpectedly small")
    leaked = _forbidden_labels(output)

    _clear_scene()
    bpy.context.scene.render.fps = SAMPLE_RATE_HZ
    bpy.context.scene.render.fps_base = 1.0
    bpy.ops.import_scene.fbx(filepath=str(output), automatic_bone_orientation=False)
    roundtrip = _scene_contract()
    imported = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"][0]
    action = imported.animation_data.action if imported.animation_data else None
    frame_range = [float(value) for value in action.frame_range] if action else []
    # Blender's FBX import also groups the armature object's own transform
    # curves, so intersect with the bone names rather than counting groups.
    keyed_groups = {curve.group.name for curve in action.fcurves if curve.group} if action else set()
    keyed_bones = len(keyed_groups & set(data["names"]))
    mesh = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    worst_weight_total = max(
        abs(sum(group.weight for group in vertex.groups) - 1.0)
        for vertex in mesh.data.vertices
    )
    worst_influences = max(len(vertex.groups) for vertex in mesh.data.vertices)

    failures = []
    for key in ("bone_count", "bone_names", "bone_parents", "root_bones"):
        if roundtrip[key] != source_contract[key]:
            failures.append(f"{key} changed during FBX round trip")
    if roundtrip["vertices"] != 10475 or roundtrip["triangles"] != 20908:
        failures.append("locked-head SMPL-X topology changed")
    if roundtrip["uv_layers"] < 1:
        failures.append("SMPL-X UV map is absent")
    if roundtrip["vertex_groups"] != 55 or roundtrip["armature_modifiers"] != 1:
        failures.append("SMPL-X skinning contract changed")
    if worst_weight_total > 1e-4:
        failures.append("skin weights do not sum to one after the FBX round trip")
    if worst_influences > MAX_INFLUENCES:
        failures.append(f"a vertex exceeds {MAX_INFLUENCES} influences after the round trip")
    if action is None:
        failures.append("delivered FBX carries no animation")
    elif frame_range != [1.0, float(motion["frames"])]:
        failures.append(f"imported frame range is {frame_range}, expected 1..{motion['frames']}")
    if motion["worst_frame_delta_after_repair_deg"] > SPIN_TRIGGER_DEG_PER_FRAME:
        failures.append("a joint still exceeds the spin trigger after repair")
    if keyed_bones != 55:
        failures.append(f"{keyed_bones} bones are keyed, expected 55")
    if leaked:
        failures.append(f"forbidden provider labels present: {', '.join(leaked)}")

    report = {
        "schema_version": "autoanim.macap-animated-fbx/1.0",
        "status": "passed" if not failures else "failed",
        "character_name": CHARACTER_NAME,
        "output_fbx": str(output),
        "rig": "symmetrized mean-shape locked-head SMPL-X (betas = 0)",
        "motion_source": str(params_path),
        "frames": motion["frames"],
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "duration_s": motion["frames"] / SAMPLE_RATE_HZ,
        "root_travel_m": motion["root_travel_m"],
        "root_rebased_to_origin": True,
        "face_pose_zeroed": True,
        "discarded_face_pose_magnitude_rad": motion["discarded_face_pose_magnitude_rad"],
        "spin_repairs": motion["spin_repairs"],
        "worst_frame_delta_after_repair_deg": motion["worst_frame_delta_after_repair_deg"],
        "mean_shape_vs_fitted_joint_deviation_m": motion[
            "mean_shape_vs_fitted_joint_deviation_m"
        ],
        "mean_shape_vs_fitted_joint_deviation_mean_m": motion[
            "mean_shape_vs_fitted_joint_deviation_mean_m"
        ],
        "symmetry_after": data["symmetry_after"],
        "pose_correctives_baked": False,
        "texture_included": False,
        "gnm_included": False,
        "foot_contacts_solved": False,
        "keyed_bones": keyed_bones,
        "imported_frame_range": frame_range,
        "max_influences_after_roundtrip": worst_influences,
        "worst_weight_sum_error_roundtrip": worst_weight_total,
        "model_sha256": _sha256(model_path),
        "params_sha256": _sha256(params_path),
        "skeleton_contract_sha256": _sha256(skeleton_path),
        "output_sha256": _sha256(output),
        "source": source_contract,
        "roundtrip": roundtrip,
        "failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("macap animated FBX verification failed: " + "; ".join(failures))
    print(json.dumps({key: report[key] for key in ("status", "frames", "root_travel_m")}))


if __name__ == "__main__":
    main()
