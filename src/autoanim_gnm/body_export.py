"""Validated glTF 2.0 skin export for AutoAnim's canonical body track.

The exporter consumes only a body asset which has already crossed the
``body_provider`` fail-closed boundary.  It preserves the provider mesh and
bind matrices, animates the exact canonical joint order, and records the
source/body hashes in glTF metadata.  It intentionally does not attach a GNM
head: that requires a separately calibrated, character-bound head socket.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .body import BodyTrack, CANONICAL_HUMANOID, validate_body_track
from .body_provider import load_and_validate_body_asset, sha256_file
from .serialization import write_npz


BODY_GLTF_EXPORT_SCHEMA_VERSION = "autoanim.body-gltf-export/1.0"


@dataclass(frozen=True, slots=True)
class AnimatedBodyGLBExport:
    path: Path
    mapping_path: Path
    frame_count: int
    vertex_count: int
    triangle_count: int
    joint_count: int
    duration_seconds: float
    body_asset_sha256: str
    body_track_sha256: str
    attachment_calibrated: bool


class _Builder:
    def __init__(self) -> None:
        self.data = bytearray()
        self.buffer_views: list[dict[str, int]] = []
        self.accessors: list[dict[str, Any]] = []

    def accessor(
        self,
        array: np.ndarray,
        *,
        component_type: int,
        accessor_type: str,
        target: int | None = None,
        normalized: bool = False,
        bounds: bool = False,
    ) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        contiguous = np.ascontiguousarray(array)
        offset = len(self.data)
        self.data.extend(contiguous.tobytes(order="C"))
        view: dict[str, int] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": int(contiguous.nbytes),
        }
        if target is not None:
            view["target"] = target
        view_index = len(self.buffer_views)
        self.buffer_views.append(view)
        components = {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT4": 16,
        }[accessor_type]
        if contiguous.size % components:
            raise ValueError("Accessor data does not match its declared glTF type")
        accessor: dict[str, Any] = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": int(contiguous.size // components),
            "type": accessor_type,
        }
        if normalized:
            accessor["normalized"] = True
        if bounds:
            rows = contiguous.reshape(-1, components)
            accessor["min"] = rows.min(axis=0).astype(float).tolist()
            accessor["max"] = rows.max(axis=0).astype(float).tolist()
        index = len(self.accessors)
        self.accessors.append(accessor)
        return index


def _glb_bytes(document: dict[str, Any], binary: bytes) -> bytes:
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    padded_binary = binary + b"\0" * ((-len(binary)) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(padded_binary)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<I4s", len(encoded), b"JSON"),
            encoded,
            struct.pack("<I4s", len(padded_binary), b"BIN\0"),
            padded_binary,
        )
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    values = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    normals = np.zeros_like(values)
    p0, p1, p2 = (values[faces[:, corner]] for corner in range(3))
    face_normals = np.cross(p1 - p0, p2 - p0)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = lengths[:, 0] <= 1.0e-12
    normals[~fallback] /= lengths[~fallback]
    normals[fallback] = (0.0, 1.0, 0.0)
    return normals.astype(np.float32)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(np.asarray(left, dtype=np.float64), -1, 0)
    rx, ry, rz, rw = np.moveaxis(np.asarray(right, dtype=np.float64), -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _quaternion_inverse(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    inverse = quaternion.copy()
    inverse[..., :3] *= -1.0
    return inverse / np.sum(quaternion * quaternion, axis=-1, keepdims=True)


def _compose_rest_and_delta(
    local_rest_matrices: np.ndarray,
    local_delta_xyzw: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rest = np.asarray(local_rest_matrices, dtype=np.float64)
    delta = np.asarray(local_delta_xyzw, dtype=np.float64)
    hierarchy = np.asarray(parents, dtype=np.int64)
    translations = rest[:, :3, 3].astype(np.float32)
    rest_local = Rotation.from_matrix(rest[:, :3, :3]).as_quat()
    joint_count = len(rest_local)
    frame_count = len(delta)
    rest_world = np.zeros((joint_count, 4), dtype=np.float64)
    target_world = np.zeros((frame_count, joint_count, 4), dtype=np.float64)
    animated_world = np.zeros_like(target_world)
    animated_local = np.zeros_like(target_world)
    for joint_index, parent in enumerate(hierarchy.tolist()):
        if parent == -1:
            rest_world[joint_index] = rest_local[joint_index]
            target_world[:, joint_index] = delta[:, joint_index]
        else:
            rest_world[joint_index] = _quaternion_multiply(
                rest_world[parent], rest_local[joint_index]
            )
            target_world[:, joint_index] = _quaternion_multiply(
                target_world[:, parent], delta[:, joint_index]
            )
        animated_world[:, joint_index] = _quaternion_multiply(
            target_world[:, joint_index], rest_world[joint_index]
        )
        if parent == -1:
            animated_local[:, joint_index] = animated_world[:, joint_index]
        else:
            animated_local[:, joint_index] = _quaternion_multiply(
                _quaternion_inverse(animated_world[:, parent]),
                animated_world[:, joint_index],
            )
    animated_local /= np.linalg.norm(animated_local, axis=2, keepdims=True)
    for frame in range(1, len(animated_local)):
        flip = (
            np.sum(animated_local[frame - 1] * animated_local[frame], axis=1)
            < 0.0
        )
        animated_local[frame, flip] *= -1.0
    return translations, animated_local.astype(np.float32)


def _body_track_sha256(track: BodyTrack) -> str:
    return sha256(track.canonical_json_bytes()).hexdigest()


def export_animated_body_glb(
    path: str | Path,
    *,
    body_manifest_path: str | Path,
    body_asset_path: str | Path,
    track: BodyTrack,
    mapping_path: str | Path | None = None,
    expected_request_sha256: str | None = None,
) -> AnimatedBodyGLBExport:
    """Export an MPFB body animated by one validated canonical body track."""

    validate_body_track(track)
    manifest = load_and_validate_body_asset(
        body_manifest_path,
        body_asset_path,
        expected_request_sha256=expected_request_sha256,
    )
    with np.load(body_asset_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    joint_names = tuple(str(value) for value in arrays["joint_names"].tolist())
    if joint_names != CANONICAL_HUMANOID.names or joint_names != track.joint_names:
        raise ValueError("Body asset, track, and canonical joint orders do not match")
    parents = np.asarray(arrays["parents"], dtype=np.int64)
    vertices = np.asarray(arrays["vertices_m"], dtype=np.float32)
    triangles = np.asarray(arrays["triangles"], dtype=np.int32)
    joint_indices = np.asarray(arrays["joint_indices"], dtype=np.uint16)
    joint_weights = np.asarray(arrays["joint_weights"], dtype=np.float32)
    local_rest = np.asarray(arrays["local_rest_matrices"], dtype=np.float32)
    inverse_bind = np.asarray(arrays["inverse_bind_matrices"], dtype=np.float32)
    rest_translations, animated_rotations = _compose_rest_and_delta(
        local_rest, track.local_rotations_xyzw, parents
    )
    root_translation = (
        rest_translations[0][None, :] + track.root_translation_m
    ).astype(np.float32)
    timestamps = (track.ticks.astype(np.float64) / track.ticks_per_second).astype(
        np.float32
    )

    builder = _Builder()
    position_accessor = builder.accessor(
        vertices,
        component_type=5126,
        accessor_type="VEC3",
        target=34962,
        bounds=True,
    )
    normal_accessor = builder.accessor(
        _vertex_normals(vertices, triangles),
        component_type=5126,
        accessor_type="VEC3",
        target=34962,
    )
    joints0_accessor = builder.accessor(
        joint_indices[:, :4],
        component_type=5123,
        accessor_type="VEC4",
        target=34962,
    )
    weights0_accessor = builder.accessor(
        joint_weights[:, :4],
        component_type=5126,
        accessor_type="VEC4",
        target=34962,
    )
    attributes: dict[str, int] = {
        "POSITION": position_accessor,
        "NORMAL": normal_accessor,
        "JOINTS_0": joints0_accessor,
        "WEIGHTS_0": weights0_accessor,
    }
    if np.any(joint_weights[:, 4:] > 0.0):
        attributes["JOINTS_1"] = builder.accessor(
            joint_indices[:, 4:],
            component_type=5123,
            accessor_type="VEC4",
            target=34962,
        )
        attributes["WEIGHTS_1"] = builder.accessor(
            joint_weights[:, 4:],
            component_type=5126,
            accessor_type="VEC4",
            target=34962,
        )
    index_type = np.uint16 if len(vertices) <= 65_535 else np.uint32
    index_accessor = builder.accessor(
        triangles.astype(index_type),
        component_type=5123 if index_type is np.uint16 else 5125,
        accessor_type="SCALAR",
        target=34963,
        bounds=True,
    )
    # glTF matrices are serialized column-major.
    inverse_bind_accessor = builder.accessor(
        inverse_bind.transpose(0, 2, 1),
        component_type=5126,
        accessor_type="MAT4",
    )
    time_accessor = builder.accessor(
        timestamps,
        component_type=5126,
        accessor_type="SCALAR",
        bounds=True,
    )

    nodes: list[dict[str, Any]] = []
    for index, name in enumerate(joint_names):
        node: dict[str, Any] = {
            "name": name,
            "translation": rest_translations[index].astype(float).tolist(),
            "rotation": animated_rotations[0, index].astype(float).tolist(),
        }
        children = np.flatnonzero(parents == index).astype(int).tolist()
        if children:
            node["children"] = children
        nodes.append(node)
    mesh_node = len(nodes)
    nodes.append({"name": "AutoAnim_MPFB_Body", "mesh": 0, "skin": 0})

    samplers: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    root_translation_accessor = builder.accessor(
        root_translation,
        component_type=5126,
        accessor_type="VEC3",
    )
    samplers.append(
        {"input": time_accessor, "output": root_translation_accessor, "interpolation": "LINEAR"}
    )
    channels.append({"sampler": 0, "target": {"node": 0, "path": "translation"}})
    for joint_index in range(len(joint_names)):
        rotation_accessor = builder.accessor(
            animated_rotations[:, joint_index],
            component_type=5126,
            accessor_type="VEC4",
        )
        sampler_index = len(samplers)
        samplers.append(
            {"input": time_accessor, "output": rotation_accessor, "interpolation": "LINEAR"}
        )
        channels.append(
            {
                "sampler": sampler_index,
                "target": {"node": joint_index, "path": "rotation"},
            }
        )

    body_asset_sha256 = sha256_file(body_asset_path)
    body_track_sha256 = _body_track_sha256(track)
    calibrated = bool(manifest["gnm_head_socket"]["attachment_calibrated"])
    document: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "AutoAnim body exporter 1.0",
            "extras": {
                "schema_version": BODY_GLTF_EXPORT_SCHEMA_VERSION,
                "coordinate_system": "+Y_up_+Z_forward_meters",
                "body_asset_sha256": body_asset_sha256,
                "body_manifest_sha256": sha256_file(body_manifest_path),
                "body_track_sha256": body_track_sha256,
                "attachment_calibrated": calibrated,
                "gnm_head_included": False,
                "production_validated": False,
            },
        },
        "scene": 0,
        "scenes": [{"nodes": [0, mesh_node]}],
        "nodes": nodes,
        "meshes": [
            {
                "name": "MPFB_HM08_Body",
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "MPFB anatomical preview",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.58, 0.31, 0.22, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.82,
                },
                "doubleSided": False,
            }
        ],
        "skins": [
            {
                "name": "AutoAnim canonical humanoid",
                "inverseBindMatrices": inverse_bind_accessor,
                "skeleton": 0,
                "joints": list(range(len(joint_names))),
            }
        ],
        "animations": [{"name": "autoanim_body", "samplers": samplers, "channels": channels}],
        "buffers": [{"byteLength": len(builder.data)}],
        "bufferViews": builder.buffer_views,
        "accessors": builder.accessors,
    }
    output = Path(path)
    _atomic_bytes(output, _glb_bytes(document, bytes(builder.data)))
    mapping = (
        Path(mapping_path)
        if mapping_path is not None
        else output.with_name(f"{output.stem}-mapping.npz")
    )
    write_npz(
        mapping,
        ticks=track.ticks,
        timestamps_seconds=timestamps,
        root_translation_m=track.root_translation_m,
        local_rotations_xyzw=track.local_rotations_xyzw,
        body_asset_sha256=np.asarray(body_asset_sha256),
        body_track_sha256=np.asarray(body_track_sha256),
    )
    return AnimatedBodyGLBExport(
        path=output,
        mapping_path=mapping,
        frame_count=len(track.ticks),
        vertex_count=len(vertices),
        triangle_count=len(triangles),
        joint_count=len(joint_names),
        duration_seconds=float(timestamps[-1]),
        body_asset_sha256=body_asset_sha256,
        body_track_sha256=body_track_sha256,
        attachment_calibrated=calibrated,
    )


__all__ = [
    "AnimatedBodyGLBExport",
    "BODY_GLTF_EXPORT_SCHEMA_VERSION",
    "export_animated_body_glb",
]
