from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from autoanim_gnm.body import BodyTrack, CANONICAL_HUMANOID
from autoanim_gnm import body_export


def _track() -> BodyTrack:
    frames = 3
    rotations = np.zeros((frames, 25, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    half = np.deg2rad(20.0) / 2.0
    rotations[1:, CANONICAL_HUMANOID.index("LeftUpperArm"), 2] = np.sin(half)
    rotations[1:, CANONICAL_HUMANOID.index("LeftUpperArm"), 3] = np.cos(half)
    eyes = np.zeros((frames, 2, 4), dtype=np.float32)
    eyes[..., 3] = 1.0
    gaze = np.zeros((frames, 3), dtype=np.float32)
    gaze[:, 2] = 1.0
    return BodyTrack(
        duration_ticks=3200,
        ticks_per_second=48_000,
        sample_rate_hz=30,
        joint_names=CANONICAL_HUMANOID.names,
        ticks=np.asarray([0, 1600, 3200], dtype=np.int64),
        root_translation_m=np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=np.float32,
        ),
        local_rotations_xyzw=rotations,
        foot_contacts=np.zeros((frames, 2), dtype=np.bool_),
        gaze_direction_body=gaze,
        gaze_strength=np.zeros(frames, dtype=np.float32),
        gnm_eye_rotations_xyzw=eyes,
        source_plan_sha256="1" * 64,
    )


def _body_asset(path: Path, *, names: tuple[str, ...] = CANONICAL_HUMANOID.names) -> None:
    rest = np.repeat(np.eye(4, dtype=np.float32)[None], 25, axis=0)
    rest[0, 1, 3] = 0.8
    inverse = np.repeat(np.eye(4, dtype=np.float32)[None], 25, axis=0)
    indices = np.zeros((3, 8), dtype=np.int16)
    weights = np.zeros((3, 8), dtype=np.float32)
    weights[:, 0] = 1.0
    np.savez_compressed(
        path,
        vertices_m=np.asarray(
            [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
            dtype=np.float32,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        joint_names=np.asarray(names),
        parents=np.asarray([joint.parent for joint in CANONICAL_HUMANOID.joints], dtype=np.int16),
        local_rest_matrices=rest,
        inverse_bind_matrices=inverse,
        joint_indices=indices,
        joint_weights=weights,
    )


def _read_glb(path: Path) -> tuple[dict, bytes]:
    payload = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", payload, 0)
    assert magic == b"glTF"
    assert version == 2
    assert total == len(payload)
    json_length, json_kind = struct.unpack_from("<I4s", payload, 12)
    assert json_kind == b"JSON"
    document = json.loads(payload[20 : 20 + json_length])
    binary_offset = 20 + json_length
    binary_length, binary_kind = struct.unpack_from("<I4s", payload, binary_offset)
    assert binary_kind == b"BIN\0"
    binary = payload[binary_offset + 8 : binary_offset + 8 + binary_length]
    return document, binary


def _accessor(document: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = {
        5123: np.dtype("<u2"),
        5125: np.dtype("<u4"),
        5126: np.dtype("<f4"),
    }[accessor["componentType"]]
    components = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }[accessor["type"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    return np.frombuffer(
        binary,
        dtype=dtype,
        count=int(accessor["count"]) * components,
        offset=offset,
    ).reshape(int(accessor["count"]), components)


def test_export_animated_body_glb_is_one_skin_one_timeline_and_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "body.json"
    manifest.write_text("{}", encoding="utf-8")
    asset = tmp_path / "body.npz"
    _body_asset(asset)
    monkeypatch.setattr(
        body_export,
        "load_and_validate_body_asset",
        lambda *args, **kwargs: {
            "gnm_head_socket": {"attachment_calibrated": False}
        },
    )
    output = tmp_path / "body.glb"
    result = body_export.export_animated_body_glb(
        output,
        body_manifest_path=manifest,
        body_asset_path=asset,
        track=_track(),
    )

    document, binary = _read_glb(output)
    assert len(document["skins"]) == 1
    assert len(document["animations"]) == 1
    animation = document["animations"][0]
    assert len(animation["channels"]) == 26
    assert {sampler["input"] for sampler in animation["samplers"]} == {
        animation["samplers"][0]["input"]
    }
    times = _accessor(
        document, binary, animation["samplers"][0]["input"]
    ).reshape(-1)
    np.testing.assert_allclose(times, [0.0, 1.0 / 30.0, 2.0 / 30.0])
    root_translation = _accessor(
        document, binary, animation["samplers"][0]["output"]
    )
    np.testing.assert_allclose(
        root_translation,
        [[0.0, 0.8, 0.0], [0.1, 0.8, 0.0], [0.2, 0.8, 0.0]],
    )
    extras = document["asset"]["extras"]
    assert extras["gnm_head_included"] is False
    assert extras["attachment_calibrated"] is False
    assert extras["production_validated"] is False
    assert extras["body_asset_sha256"] == result.body_asset_sha256
    assert extras["body_track_sha256"] == result.body_track_sha256
    assert result.frame_count == 3
    assert result.joint_count == 25
    with np.load(result.mapping_path, allow_pickle=False) as mapping:
        np.testing.assert_array_equal(mapping["ticks"], [0, 1600, 3200])
        np.testing.assert_array_equal(
            mapping["local_rotations_xyzw"],
            _track().local_rotations_xyzw,
        )


def test_export_rejects_body_asset_joint_order_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "body.json"
    manifest.write_text("{}", encoding="utf-8")
    asset = tmp_path / "body.npz"
    names = list(CANONICAL_HUMANOID.names)
    names[1], names[2] = names[2], names[1]
    _body_asset(asset, names=tuple(names))
    monkeypatch.setattr(
        body_export,
        "load_and_validate_body_asset",
        lambda *args, **kwargs: {
            "gnm_head_socket": {"attachment_calibrated": False}
        },
    )
    with pytest.raises(ValueError, match="joint orders"):
        body_export.export_animated_body_glb(
            tmp_path / "body.glb",
            body_manifest_path=manifest,
            body_asset_path=asset,
            track=_track(),
        )


def test_rest_basis_retarget_recovers_canonical_world_deformation() -> None:
    rest = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    rest[0, :3, :3] = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    rest[1, :3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    delta = np.zeros((2, 2, 4), dtype=np.float32)
    delta[..., 3] = 1.0
    angle = np.deg2rad(30.0) / 2.0
    delta[1, 0] = [0.0, np.sin(angle), 0.0, np.cos(angle)]
    delta[1, 1] = [0.0, 0.0, np.sin(angle), np.cos(angle)]
    _, animated_local = body_export._compose_rest_and_delta(
        rest,
        delta,
        np.asarray([-1, 0], dtype=np.int64),
    )

    rest_local = body_export.Rotation.from_matrix(rest[:, :3, :3]).as_quat()
    rest_world = np.stack(
        (
            rest_local[0],
            body_export._quaternion_multiply(rest_local[0], rest_local[1]),
        )
    )
    for frame in range(2):
        target_root = delta[frame, 0]
        target_child = body_export._quaternion_multiply(
            target_root, delta[frame, 1]
        )
        animated_root = animated_local[frame, 0]
        animated_child = body_export._quaternion_multiply(
            animated_root, animated_local[frame, 1]
        )
        root_deformation = body_export._quaternion_multiply(
            animated_root, body_export._quaternion_inverse(rest_world[0])
        )
        child_deformation = body_export._quaternion_multiply(
            animated_child, body_export._quaternion_inverse(rest_world[1])
        )
        np.testing.assert_allclose(
            np.abs(np.dot(root_deformation, target_root)), 1.0, atol=1.0e-6
        )
        np.testing.assert_allclose(
            np.abs(np.dot(child_deformation, target_child)), 1.0, atol=1.0e-6
        )
