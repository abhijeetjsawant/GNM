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

from .body import (
    BodyTrack,
    HumanoidSkeleton,
    _rotate_vector,
    skeleton_for_joint_names,
    skeleton_for_track,
    validate_body_track,
)
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


def _shortest_arc_quaternion(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_direction = np.asarray(source, dtype=np.float64)
    target_direction = np.asarray(target, dtype=np.float64)
    source_direction /= np.linalg.norm(source_direction)
    target_direction /= np.linalg.norm(target_direction)
    dot = float(np.clip(np.dot(source_direction, target_direction), -1.0, 1.0))
    if dot >= 1.0 - 1.0e-12:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    if dot <= -1.0 + 1.0e-12:
        axis = np.cross(source_direction, (1.0, 0.0, 0.0))
        if np.linalg.norm(axis) <= 1.0e-8:
            axis = np.cross(source_direction, (0.0, 1.0, 0.0))
        axis /= np.linalg.norm(axis)
        return np.asarray((*axis, 0.0), dtype=np.float64)
    cross = np.cross(source_direction, target_direction)
    quaternion = np.asarray((*cross, 1.0 + dot), dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _orthonormal_frame(
    primary_direction: np.ndarray,
    secondary_direction: np.ndarray,
) -> np.ndarray:
    """Build a right-handed frame while preserving primary-axis direction."""

    primary = np.array(primary_direction, dtype=np.float64, copy=True)
    secondary = np.array(secondary_direction, dtype=np.float64, copy=True)
    primary_norm = float(np.linalg.norm(primary))
    if primary_norm <= 1.0e-8:
        raise ValueError("Bind-frame primary direction is degenerate")
    primary /= primary_norm
    secondary -= primary * float(np.dot(primary, secondary))
    secondary_norm = float(np.linalg.norm(secondary))
    if secondary_norm <= 1.0e-8:
        raise ValueError("Bind-frame secondary direction is degenerate")
    secondary /= secondary_norm
    normal = np.cross(primary, secondary)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-8:
        raise ValueError("Bind-frame normal is degenerate")
    normal /= normal_norm
    secondary = np.cross(normal, primary)
    secondary /= np.linalg.norm(secondary)
    return np.column_stack((primary, secondary, normal))


def _frame_alignment_quaternion(
    provider_primary: np.ndarray,
    provider_secondary: np.ndarray,
    canonical_primary: np.ndarray,
    canonical_secondary: np.ndarray,
) -> np.ndarray:
    """Return the world rotation mapping one provider bind frame to canonical."""

    provider_frame = _orthonormal_frame(provider_primary, provider_secondary)
    canonical_frame = _orthonormal_frame(canonical_primary, canonical_secondary)
    return Rotation.from_matrix(canonical_frame @ provider_frame.T).as_quat()


def _canonical_arm_bind_alignment(
    local_rest_matrices: np.ndarray,
    parents: np.ndarray,
    *,
    skeleton: HumanoidSkeleton,
) -> np.ndarray:
    """Align provider arm and detailed-hand bind frames to canonical axes."""

    rest = np.asarray(local_rest_matrices, dtype=np.float64)
    hierarchy = np.asarray(parents, dtype=np.int64)
    joint_count = len(skeleton.joints)
    if rest.shape != (joint_count, 4, 4) or hierarchy.shape != (joint_count,):
        raise ValueError("Body rest skeleton does not match the canonical joint count")
    rest_world = np.empty_like(rest)
    for joint_index, parent in enumerate(hierarchy.tolist()):
        rest_world[joint_index] = (
            rest[joint_index]
            if parent == -1
            else rest_world[parent] @ rest[joint_index]
        )
    positions = rest_world[:, :3, 3]
    canonical_positions = np.zeros((joint_count, 3), dtype=np.float64)
    for joint_index, joint in enumerate(skeleton.joints):
        canonical_positions[joint_index] = np.asarray(
            joint.rest_translation_m, dtype=np.float64
        )
        if joint.parent >= 0:
            canonical_positions[joint_index] += canonical_positions[joint.parent]
    alignment = np.zeros((joint_count, 4), dtype=np.float64)
    alignment[:, 3] = 1.0
    for side in ("Left", "Right"):
        chain = tuple(
            side + suffix
            for suffix in ("Shoulder", "UpperArm", "LowerArm", "Hand")
        )
        for parent_name, child_name in zip(chain[:-1], chain[1:], strict=True):
            parent_index = skeleton.index(parent_name)
            child_index = skeleton.index(child_name)
            provider_direction = positions[child_index] - positions[parent_index]
            canonical_direction = np.asarray(
                skeleton.joints[child_index].rest_translation_m,
                dtype=np.float64,
            )
            if (
                np.linalg.norm(provider_direction) <= 1.0e-8
                or np.linalg.norm(canonical_direction) <= 1.0e-8
            ):
                continue
            alignment[parent_index] = _shortest_arc_quaternion(
                provider_direction, canonical_direction
            )
        alignment[skeleton.index(chain[-1])] = alignment[
            skeleton.index(chain[-2])
        ]
    finger_chains = (
        ("ThumbMetacarpal", "ThumbProximal", "ThumbDistal"),
        ("IndexProximal", "IndexIntermediate", "IndexDistal"),
        ("MiddleProximal", "MiddleIntermediate", "MiddleDistal"),
        ("RingProximal", "RingIntermediate", "RingDistal"),
        ("LittleProximal", "LittleIntermediate", "LittleDistal"),
    )
    if all(
        side + joint_name in skeleton.names
        for side in ("Left", "Right")
        for chain in finger_chains
        for joint_name in chain
    ):
        for side in ("Left", "Right"):
            hand_index = skeleton.index(side + "Hand")
            index_root = skeleton.index(side + "IndexProximal")
            middle_root = skeleton.index(side + "MiddleProximal")
            little_root = skeleton.index(side + "LittleProximal")
            provider_palm_primary = (
                positions[middle_root] - positions[hand_index]
            )
            provider_palm_secondary = (
                positions[index_root] - positions[little_root]
            )
            canonical_palm_primary = (
                canonical_positions[middle_root] - canonical_positions[hand_index]
            )
            canonical_palm_secondary = (
                canonical_positions[index_root] - canonical_positions[little_root]
            )
            provider_palm_frame = _orthonormal_frame(
                provider_palm_primary, provider_palm_secondary
            )
            canonical_palm_frame = _orthonormal_frame(
                canonical_palm_primary, canonical_palm_secondary
            )
            alignment[hand_index] = Rotation.from_matrix(
                canonical_palm_frame @ provider_palm_frame.T
            ).as_quat()
            provider_palm_normal = provider_palm_frame[:, 2]
            canonical_palm_normal = canonical_palm_frame[:, 2]
            for finger_chain in finger_chains:
                for parent_name, child_name in zip(
                    finger_chain[:-1], finger_chain[1:], strict=True
                ):
                    parent_index = skeleton.index(side + parent_name)
                    child_index = skeleton.index(side + child_name)
                    alignment[parent_index] = _frame_alignment_quaternion(
                        positions[child_index] - positions[parent_index],
                        provider_palm_normal,
                        canonical_positions[child_index]
                        - canonical_positions[parent_index],
                        canonical_palm_normal,
                    )
                distal_index = skeleton.index(side + finger_chain[-1])
                intermediate_index = skeleton.index(side + finger_chain[-2])
                # The asset has no fingertip/end joint to define a distal axis.
                alignment[distal_index] = alignment[intermediate_index]
    return alignment


def _compose_rest_and_delta(
    local_rest_matrices: np.ndarray,
    local_delta_xyzw: np.ndarray,
    parents: np.ndarray,
    *,
    world_bind_alignment_xyzw: np.ndarray | None = None,
    rest_translations_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a track's world rotations onto the asset's rest frames.

    ``rest_translations_m`` is the TRACK's rest skeleton, ``[joint, 3]`` in the rig's own
    frame.  When it is given, the node translations returned are that skeleton's bone
    offsets **expressed in each parent's aligned asset rest frame**, so that the exported
    glTF's forward kinematics reproduces
    :func:`autoanim_gnm.body.forward_kinematics_positions` on the same track exactly.

    D3, and this is the whole of the two-skeleton defect it closes.  The animated world
    rotation this function builds is
    ``animated_world[j] = track_world[j] * alignment[j] * rest_world[j]``, and a glTF
    reader places a child at ``pos[p] + animated_world[p] * t_local[j]``.  Forward
    kinematics wants ``pos[p] + track_world[p] * rest_track[j]``.  Setting

        t_local[j] = inv(alignment[p] * rest_world[p]) * rest_track[j]

    makes the two identical for every joint and every frame, by construction, with no
    tolerance and no fit.  Until today ``t_local`` was the ASSET's own rest offset, so
    the exported GLB carried the provider's A-posed, 170 mm-half-span body while every
    instrument scored the code skeleton's T-posed, 270 mm one: measured at 81-195 mm per
    joint on the delivered build.

    ``None`` keeps the previous behaviour exactly -- the asset's own rest offsets, in
    float32 -- for callers that have no track skeleton in hand.
    """

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
    if world_bind_alignment_xyzw is None:
        alignment = np.zeros((joint_count, 4), dtype=np.float64)
        alignment[:, 3] = 1.0
    else:
        alignment = np.array(
            world_bind_alignment_xyzw, dtype=np.float64, copy=True
        )
        if alignment.shape != (joint_count, 4):
            raise ValueError("World bind alignment must have shape [joints, 4]")
        alignment_norms = np.linalg.norm(alignment, axis=1, keepdims=True)
        if not np.isfinite(alignment_norms).all() or np.any(
            alignment_norms <= 1.0e-12
        ):
            raise ValueError("World bind alignment must contain finite quaternions")
        alignment /= alignment_norms
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
            _quaternion_multiply(
                target_world[:, joint_index], alignment[joint_index]
            ),
            rest_world[joint_index],
        )
        if parent == -1:
            animated_local[:, joint_index] = animated_world[:, joint_index]
        else:
            animated_local[:, joint_index] = _quaternion_multiply(
                _quaternion_inverse(animated_world[:, parent]),
                animated_world[:, joint_index],
            )
    if rest_translations_m is not None:
        track_rest = np.asarray(rest_translations_m, dtype=np.float64)
        if track_rest.shape != (joint_count, 3) or not np.isfinite(track_rest).all():
            raise ValueError("Track rest translations must be a finite [joints, 3] array")
        placed = np.zeros((joint_count, 3), dtype=np.float64)
        for joint_index, parent in enumerate(hierarchy.tolist()):
            if parent == -1:
                # The root's rest offset is zero by contract; its world placement is the
                # animated root translation the caller adds, never the asset's own root
                # rest, which is a different body's.
                placed[joint_index] = track_rest[joint_index]
            else:
                parent_frame = _quaternion_multiply(
                    alignment[parent], rest_world[parent]
                )
                placed[joint_index] = _rotate_vector(
                    _quaternion_inverse(parent_frame), track_rest[joint_index]
                )
        translations = placed
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

    # D3: the TRACK's skeleton, not merely the skeleton its joint NAMES resolve to.
    target_skeleton = skeleton_for_track(track)
    validate_body_track(track, skeleton=target_skeleton)
    manifest = load_and_validate_body_asset(
        body_manifest_path,
        body_asset_path,
        expected_request_sha256=expected_request_sha256,
    )
    with np.load(body_asset_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    joint_names = tuple(str(value) for value in arrays["joint_names"].tolist())
    if joint_names != target_skeleton.names or joint_names != track.joint_names:
        raise ValueError("Body asset, track, and selected joint orders do not match")
    parents = np.asarray(arrays["parents"], dtype=np.int64)
    vertices = np.asarray(arrays["vertices_m"], dtype=np.float32)
    triangles = np.asarray(arrays["triangles"], dtype=np.int32)
    joint_indices = np.asarray(arrays["joint_indices"], dtype=np.uint16)
    joint_weights = np.asarray(arrays["joint_weights"], dtype=np.float32)
    local_rest = np.asarray(arrays["local_rest_matrices"], dtype=np.float32)
    inverse_bind = np.asarray(arrays["inverse_bind_matrices"], dtype=np.float32)
    rest_translations, animated_rotations = _compose_rest_and_delta(
        local_rest,
        track.local_rotations_xyzw,
        parents,
        world_bind_alignment_xyzw=_canonical_arm_bind_alignment(
            local_rest, parents, skeleton=target_skeleton
        ),
        rest_translations_m=track.rest_translations_m,
    )
    # `rest_translations[0]` is the TRACK root's rest offset, which is zero by contract,
    # so the animated root node carries the track's root translation and nothing else.
    # It used to be the ASSET's Root rest, a different body's offset added to our root.
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
        # D3: the rest the track was solved on, so an npz-only consumer cannot silently
        # run forward kinematics on the canonical body.
        rest_translations_m=track.rest_translations_m,
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
