"""Export quarantined GEM-X results as a safe SOMA-77 JSON/NPZ response.

This is the only provider-side step allowed to translate NVIDIA's internal
``torch.save`` artifact into the primitive-array format consumed by AutoAnim.
It also converts GEM-X's T-pose-relative rotations to AutoAnim's reviewed
post-rest-local quaternion convention.  Copying axis-angle values directly
does not preserve SOMA forward kinematics.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
import torch

from gem.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
)
from gem.utils.soma_utils.soma_layer import SomaLayer
try:
    from .video_timing import load_timing
except ImportError:  # Direct provider-worker execution.
    from video_timing import load_timing


GEM_X_COMMIT = "32992550dba114c62243fb55e361311972dce8f9"
CONTACT_INDICES = (69, 70, 74, 75, 14, 42)
CONTACT_NAMES = (
    "LeftFoot",
    "LeftToeBase",
    "RightFoot",
    "RightToeBase",
    "LeftHand",
    "RightHand",
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _xyzw(matrix: torch.Tensor) -> torch.Tensor:
    return matrix_to_quaternion(matrix)[..., [1, 2, 3, 0]]


def _continuous_quaternions(value: torch.Tensor) -> torch.Tensor:
    output = value.clone()
    for frame in range(1, len(output)):
        flip = torch.sum(output[frame - 1] * output[frame], dim=-1) < 0
        output[frame, flip] *= -1
    return output


@torch.inference_mode()
def export_soma_response(
    *,
    video_path: Path,
    timing_path: Path,
    prediction_path: Path,
    output_npz: Path,
    output_manifest: Path,
) -> dict[str, object]:
    for label, path in (
        ("Input video", video_path),
        ("GEM-X prediction", prediction_path),
        ("Frame timing", timing_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    started = time.perf_counter()
    timing = load_timing(timing_path, video_path)
    source_pts = np.asarray(timing["source_pts"], dtype=np.int64)
    timebase_numerator = int(timing["source_time_base"]["numerator"])
    timebase_denominator = int(timing["source_time_base"]["denominator"])
    prediction_bytes = prediction_path.read_bytes()
    prediction_sha256 = sha256(prediction_bytes).hexdigest()
    prediction = torch.load(
        BytesIO(prediction_bytes),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(prediction, dict) or "body_params_global" not in prediction:
        raise ValueError("GEM-X result is missing body_params_global")
    body = prediction["body_params_global"]
    required = {
        "body_pose": (len(source_pts), 228),
        "global_orient": (len(source_pts), 3),
        "transl": (len(source_pts), 3),
        "identity_coeffs": (len(source_pts), 45),
        "scale_params": (len(source_pts), 69),
    }
    if not isinstance(body, dict):
        raise ValueError("body_params_global must be a mapping")
    for name, shape in required.items():
        value = body.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"GEM-X {name} must be a finite tensor with shape {shape}")
        body[name] = value.detach().cpu().float()

    soma = SomaLayer(
        data_root="inputs/soma_assets",
        low_lod=True,
        device="cpu",
        identity_model_type="mhr",
        mode="warp",
    )
    rest_positions = soma.get_skeleton(
        body["identity_coeffs"][:1],
        body["scale_params"][:1],
    )[0]
    soma_output = soma(**body)
    joint_positions = soma_output["joints"]

    axis_angles = torch.cat(
        (
            body["global_orient"][:, None],
            body["body_pose"].reshape(len(source_pts), 76, 3),
        ),
        dim=1,
    )
    provider_deltas = axis_angle_to_matrix(axis_angles)
    # BatchedSkinning applies inv(R_parent_rest) @ delta @ R_joint_rest.
    # AutoAnim FK applies inv(R_parent_rest) @ R_joint_rest @ delta_local.
    # Therefore delta_local = inv(R_joint_rest) @ delta @ R_joint_rest.
    rest_world_matrices = soma.soma.t_pose_world[1:, :3, :3].float()
    local_matrices = (
        rest_world_matrices.transpose(-1, -2)[None]
        @ provider_deltas
        @ rest_world_matrices[None]
    )
    local_rotations = _continuous_quaternions(_xyzw(local_matrices))
    rest_world_rotations = _xyzw(rest_world_matrices)
    frame_seconds = (
        torch.from_numpy(np.diff(source_pts).astype(np.float32))
        * (timebase_numerator / timebase_denominator)
    )
    if not torch.isfinite(frame_seconds).all() or torch.any(frame_seconds <= 0):
        raise RuntimeError("Source PTS does not define positive frame intervals")
    contact_speed = torch.linalg.vector_norm(
        joint_positions[1:] - joint_positions[:-1],
        dim=-1,
    ) / frame_seconds[:, None]
    static = contact_speed < 0.15
    contacts = torch.cat((static, static[[-1]]), dim=0)[:, list(CONTACT_INDICES)]

    arrays = {
        "source_pts": source_pts.astype("<i8", copy=False),
        "root_translation_m": body["transl"].numpy().astype("<f4", copy=False),
        "local_rotations_xyzw": local_rotations.numpy().astype("<f4", copy=False),
        "rest_joint_positions_m": rest_positions.numpy().astype("<f4", copy=False),
        "rest_world_rotations_xyzw": rest_world_rotations.numpy().astype("<f4", copy=False),
        "joint_positions_m": joint_positions.numpy().astype("<f4", copy=False),
        "contacts": contacts.numpy().astype(np.bool_, copy=False),
    }
    if not all(np.isfinite(value).all() for name, value in arrays.items() if name != "contacts"):
        raise RuntimeError("Provider export produced a non-finite array")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    npz_temporary = output_npz.with_suffix(f"{output_npz.suffix}.partial")
    with npz_temporary.open("wb") as destination:
        np.savez(destination, **arrays)
    npz_temporary.replace(output_npz)
    npz_sha256 = _file_sha256(output_npz)
    manifest = {
        "schema_version": "autoanim.gem-x-provider-response/1.0",
        "provider_id": "nvidia_gem_x",
        "provider_git_commit_oid": GEM_X_COMMIT,
        "runtime_class": "apple_silicon_preview",
        "execution_provider": "CPUExecutionProvider",
        "camera_model": "static_camera_assumed",
        "operation": "video_capture",
        "motion_kind": "observed",
        "production_validated": False,
        "frame_count": len(source_pts),
        "source_time_base": {
            "numerator": timebase_numerator,
            "denominator": timebase_denominator,
        },
        "source_coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "+Z",
            "linear_unit_in_meters": 1.0,
            "source_to_canonical_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "contact_schema": {
            "id": "gem_x_somaskel77_contacts/1.0",
            "contact_names": list(CONTACT_NAMES),
            "provider_joint_indices": list(CONTACT_INDICES),
            "velocity_threshold_m_per_s": 0.15,
            "velocity_timing": "exact_source_pts",
        },
        "input_sha256": timing["input_sha256"],
        "provider_raw_motion_sha256": prediction_sha256,
        "motion_npz": {
            "file_name": output_npz.name,
            "sha256": npz_sha256,
            "arrays": {
                name: {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                }
                for name, value in arrays.items()
            },
        },
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_temporary = output_manifest.with_suffix(
        f"{output_manifest.suffix}.partial"
    )
    manifest_temporary.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    manifest_temporary.replace(output_manifest)
    return {
        "frame_count": len(source_pts),
        "fk_convention": "post_rest_local_xyzw",
        "manifest_path": str(output_manifest.resolve()),
        "motion_path": str(output_npz.resolve()),
        "motion_sha256": npz_sha256,
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    arguments = parser.parse_args()
    result = export_soma_response(
        video_path=arguments.video.resolve(),
        timing_path=arguments.timing.resolve(),
        prediction_path=arguments.prediction.resolve(),
        output_npz=arguments.output_npz.resolve(),
        output_manifest=arguments.output_manifest.resolve(),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
