from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from autoanim_gnm.nvidia_body_provider import (
    NvidiaBodyProviderError,
    load_gem_x_preview_response,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _response(tmp_path: Path, *, compressed: bool = False) -> tuple[Path, Path, dict]:
    frames = 2
    rotations = np.zeros((frames, 77, 4), dtype="<f4")
    rotations[..., 3] = 1
    rest_rotations = np.zeros((77, 4), dtype="<f4")
    rest_rotations[..., 3] = 1
    arrays = {
        "source_pts": np.array([0, 1], dtype="<i8"),
        "root_translation_m": np.zeros((frames, 3), dtype="<f4"),
        "local_rotations_xyzw": rotations,
        "rest_joint_positions_m": np.zeros((77, 3), dtype="<f4"),
        "rest_world_rotations_xyzw": rest_rotations,
        "joint_positions_m": np.zeros((frames, 77, 3), dtype="<f4"),
        "contacts": np.zeros((frames, 6), dtype=np.bool_),
    }
    artifact = tmp_path / "motion.npz"
    writer = np.savez_compressed if compressed else np.savez
    writer(artifact, **arrays)
    manifest = {
        "schema_version": "autoanim.gem-x-provider-response/1.0",
        "provider_id": "nvidia_gem_x",
        "provider_git_commit_oid": "32992550dba114c62243fb55e361311972dce8f9",
        "runtime_class": "apple_silicon_preview",
        "execution_provider": "CPUExecutionProvider",
        "camera_model": "static_camera_assumed",
        "operation": "video_capture",
        "motion_kind": "observed",
        "production_validated": False,
        "frame_count": frames,
        "source_time_base": {"numerator": 1, "denominator": 30},
        "source_coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "+Z",
            "linear_unit_in_meters": 1.0,
            "source_to_canonical_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "contact_schema": {
            "id": "gem_x_somaskel77_contacts/1.0",
            "contact_names": [
                "LeftFoot",
                "LeftToeBase",
                "RightFoot",
                "RightToeBase",
                "LeftHand",
                "RightHand",
            ],
            "provider_joint_indices": [69, 70, 74, 75, 14, 42],
            "velocity_threshold_m_per_s": 0.15,
            "velocity_timing": "exact_source_pts",
        },
        "input_sha256": "a" * 64,
        "provider_raw_motion_sha256": "b" * 64,
        "motion_npz": {
            "file_name": artifact.name,
            "sha256": _sha(artifact),
            "arrays": {
                name: {"dtype": value.dtype.str, "shape": list(value.shape)}
                for name, value in arrays.items()
            },
        },
    }
    manifest_path = tmp_path / "motion.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, artifact, manifest


def test_loads_cpu_preview_and_derives_exact_ticks(tmp_path: Path) -> None:
    manifest, _, _ = _response(tmp_path)

    result = load_gem_x_preview_response(manifest)

    assert result.ticks.tolist() == [0, 1600]
    assert result.duration_ticks == 1600
    assert result.manifest_dict()["production_validated"] is False
    assert result.root_translation_m.flags.writeable is False


def test_rejects_tampered_npz_before_array_use(tmp_path: Path) -> None:
    manifest, artifact, _ = _response(tmp_path)
    with artifact.open("ab") as destination:
        destination.write(b"tamper")

    with pytest.raises(NvidiaBodyProviderError, match="SHA-256"):
        load_gem_x_preview_response(manifest)


def test_rejects_compressed_npz_even_when_hash_matches(tmp_path: Path) -> None:
    manifest, _, _ = _response(tmp_path, compressed=True)

    with pytest.raises(NvidiaBodyProviderError, match="uncompressed"):
        load_gem_x_preview_response(manifest)


def test_rejects_provider_declared_shape_before_numpy_load(tmp_path: Path) -> None:
    manifest_path, _, manifest = _response(tmp_path)
    manifest["motion_npz"]["arrays"]["joint_positions_m"]["shape"] = [
        2_000_000_000,
        77,
        3,
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NvidiaBodyProviderError, match="Unsafe array declaration"):
        load_gem_x_preview_response(manifest_path)


def test_rejects_truncated_npy_payload_before_numpy_load(tmp_path: Path) -> None:
    manifest_path, artifact, manifest = _response(tmp_path)
    rebuilt = BytesIO()
    with zipfile.ZipFile(artifact) as source, zipfile.ZipFile(
        rebuilt, "w", compression=zipfile.ZIP_STORED
    ) as destination:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "joint_positions_m.npy":
                payload = payload[:-4]
            destination.writestr(info.filename, payload)
    artifact.write_bytes(rebuilt.getvalue())
    manifest["motion_npz"]["sha256"] = _sha(artifact)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NvidiaBodyProviderError, match="NPY header"):
        load_gem_x_preview_response(manifest_path)


def test_rejects_traversal_and_duplicate_json_members(tmp_path: Path) -> None:
    manifest_path, _, manifest = _response(tmp_path)
    manifest["motion_npz"]["file_name"] = "../motion.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(NvidiaBodyProviderError, match="Unsafe"):
        load_gem_x_preview_response(manifest_path)

    manifest_path, _, _ = _response(tmp_path)
    original = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        '{"schema_version":"duplicate",' + original[1:],
        encoding="utf-8",
    )
    with pytest.raises(NvidiaBodyProviderError, match="Duplicate"):
        load_gem_x_preview_response(manifest_path)


def test_rejects_manifest_and_artifact_symlinks(tmp_path: Path) -> None:
    manifest, artifact, _ = _response(tmp_path)
    manifest_link = tmp_path / "manifest-link.json"
    manifest_link.symlink_to(manifest)
    with pytest.raises(NvidiaBodyProviderError, match="symlinks"):
        load_gem_x_preview_response(manifest_link)

    artifact_link = tmp_path / "artifact-link.npz"
    artifact_link.symlink_to(artifact)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["motion_npz"]["file_name"] = artifact_link.name
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(NvidiaBodyProviderError, match="regular file"):
        load_gem_x_preview_response(manifest)
