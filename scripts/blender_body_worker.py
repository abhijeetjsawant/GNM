#!/usr/bin/env python3
"""Blender-side MPFB body worker.

Usage:
  blender --background --python-exit-code 31 --python scripts/blender_body_worker.py \
    -- request.json response.json

The worker performs no network or package installation.  MPFB 2.0.16 and its
CC0 system assets must already be installed in the isolated Blender profile.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import tomllib
import traceback

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from autoanim_gnm.body import ATTACHMENT_SCHEMA_VERSION, CANONICAL_HUMANOID, attachment_contract
from autoanim_gnm.body_provider import (
    BODY_ASSET_SCHEMA,
    MAKEHUMAN_LICENSE_URL,
    MAKEHUMAN_SYSTEM_ASSETS_URL,
    MPFB_RELEASE_URL,
    PINNED_BLENDER_SHA256,
    PINNED_BLENDER_URL,
    PINNED_BLENDER_VERSION,
    PINNED_MPFB_EXTENSION_SHA256,
    PINNED_MPFB_EXTENSION_URL,
    PINNED_MPFB_GIT_COMMIT,
    PINNED_MPFB_VERSION,
    PROVIDER_ID,
    BodyProviderError,
    DependencyIssue,
    audit_makehuman_system_assets_archive,
    audit_mpfb_extension_archive,
    blocked_body_provider_response,
    load_body_provider_request,
    sha256_file,
    succeeded_body_provider_response,
    validate_body_asset,
    validate_body_profile_attestation,
    write_body_provider_json,
)


def _dynamic_import(package_suffix: str, key: str):
    for module_name in tuple(sys.modules):
        if module_name.endswith(package_suffix):
            module = importlib.import_module(module_name)
            if hasattr(module, key):
                return getattr(module, key)
    raise ImportError(f"No loaded MPFB module ends with {package_suffix}")


def _mpfb_installed_version(human_service) -> str | None:
    module = importlib.import_module(human_service.__module__)
    module_file = Path(module.__file__).resolve()
    for directory in (module_file.parent, *module_file.parents):
        manifest = directory / "blender_manifest.toml"
        if manifest.is_file():
            with manifest.open("rb") as handle:
                value = tomllib.load(handle).get("version")
            return str(value) if value is not None else None
        if directory == directory.parent:
            break
    return None


def _mpfb_extension_root(human_service) -> Path:
    module = importlib.import_module(human_service.__module__)
    module_file = Path(module.__file__).resolve()
    for directory in (module_file.parent, *module_file.parents):
        if (directory / "blender_manifest.toml").is_file():
            return directory
        if directory == directory.parent:
            break
    raise BodyProviderError("Loaded MPFB extension root could not be located")


def _write_blocked(response_path: Path, request_id: str, issue: DependencyIssue) -> int:
    write_body_provider_json(response_path, blocked_body_provider_response(request_id, [issue]))
    return 2


def _archive_issue(
    environment_key: str,
    *,
    dependency: str,
    expected_sha256: str,
    missing_code: str,
    mismatch_code: str,
    validator=None,
) -> DependencyIssue | None:
    archive_value = os.environ.get(environment_key)
    if not archive_value or not Path(archive_value).is_file():
        return DependencyIssue(
            missing_code,
            dependency,
            expected_sha256,
            None,
            f"Set {environment_key} to the audited source archive",
        )
    observed = sha256_file(archive_value)
    if observed != expected_sha256:
        return DependencyIssue(
            mismatch_code,
            dependency,
            expected_sha256,
            observed,
            "Source archive digest does not match the request pin",
        )
    if validator is not None:
        try:
            validator(Path(archive_value))
        except BodyProviderError as exc:
            return DependencyIssue(
                mismatch_code,
                dependency,
                expected_sha256,
                observed,
                str(exc),
            )
    return None


def _parse_paths() -> tuple[Path, Path]:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    if len(args) != 2:
        raise BodyProviderError("Worker requires request and response paths after --")
    request_path = Path(args[0]).resolve()
    response_path = Path(args[1]).resolve()
    if request_path.parent != response_path.parent or request_path == response_path:
        raise BodyProviderError("Request and response must be distinct sibling files")
    return request_path, response_path


def _blender_to_autoanim_matrix() -> np.ndarray:
    # Blender/MPFB (+Z up, -Y forward) -> AutoAnim (+Y up, +Z forward).
    return np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _collapse_vertex_influences(
    vertex_groups,
    group_to_joint: dict[int, int],
    *,
    maximum_influences: int,
) -> list[tuple[int, float]]:
    """Aggregate MPFB deformation groups after canonical-joint collapse.

    Several source bones (twists, fingers and secondary spine segments) can
    resolve to the same selected canonical ancestor.  Summing those weights
    before truncation is essential: emitting each source group separately
    creates duplicate active joint indices and changes the effective weight
    when only the largest eight slots survive.
    """

    accumulated: dict[int, float] = {}
    for group in vertex_groups:
        joint = group_to_joint.get(group.group)
        weight = float(group.weight)
        if joint is None or not np.isfinite(weight) or weight <= 0.0:
            continue
        accumulated[joint] = accumulated.get(joint, 0.0) + weight
    weighted = sorted(accumulated.items(), key=lambda item: item[1], reverse=True)
    weighted = weighted[:maximum_influences]
    total = sum(weight for _, weight in weighted)
    if not np.isfinite(total) or total <= 0.0:
        return []
    return [(joint, weight / total) for joint, weight in weighted]


def _cleanup_failed_export(*paths: Path | None) -> None:
    """Remove only current-request artifacts that cannot have passed validation."""

    for path in paths:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                print(f"Could not remove failed body artifact {path}: {exc}", file=sys.stderr)


def _extract_asset(
    bpy, basemesh, rig, request: dict, request_sha256: str
) -> tuple[dict, dict[str, np.ndarray]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = basemesh.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        convert = _blender_to_autoanim_matrix()
        convert3 = convert[:3, :3]
        world = np.asarray(basemesh.matrix_world, dtype=np.float64)
        vertices = np.asarray(
            [convert3 @ (world @ np.asarray([*vertex.co, 1.0]))[:3] for vertex in mesh.vertices],
            dtype=np.float32,
        )
        triangles_list: list[tuple[int, int, int]] = []
        for polygon in mesh.polygons:
            indices = list(polygon.vertices)
            if len(indices) < 3:
                continue
            for offset in range(1, len(indices) - 1):
                triangles_list.append((indices[0], indices[offset], indices[offset + 1]))
        triangles = np.asarray(triangles_list, dtype=np.int32)

        mapping = request["skeleton"]["joint_map"]
        source_bones = rig.data.bones
        missing = [source for source in mapping.values() if source not in source_bones]
        if missing:
            raise BodyProviderError(f"MPFB default rig is missing mapped bones: {missing}")

        source_global: dict[str, np.ndarray] = {}
        for canonical_name, source_name in mapping.items():
            source = np.asarray(rig.matrix_world @ source_bones[source_name].matrix_local, dtype=np.float64)
            source_global[canonical_name] = convert @ source @ np.linalg.inv(convert)
        local = np.empty((25, 4, 4), dtype=np.float64)
        global_rest = np.empty_like(local)
        for index, joint in enumerate(CANONICAL_HUMANOID.joints):
            global_rest[index] = source_global[joint.name]
            local[index] = (
                global_rest[index]
                if joint.parent == -1
                else np.linalg.inv(global_rest[joint.parent]) @ global_rest[index]
            )
        inverse = np.linalg.inv(global_rest)

        group_to_joint: dict[int, int] = {}
        selected_source_to_canonical = {
            source: CANONICAL_HUMANOID.index(canonical)
            for canonical, source in mapping.items()
        }
        # Collapse every MPFB deformation bone to its nearest selected
        # canonical ancestor.  Without this, weights on the second spine,
        # limb-twist, finger, toe and facial bones would silently disappear.
        source_to_canonical: dict[str, int] = {}
        for source_bone in rig.data.bones:
            cursor = source_bone
            while cursor is not None:
                if cursor.name in selected_source_to_canonical:
                    source_to_canonical[source_bone.name] = selected_source_to_canonical[cursor.name]
                    break
                cursor = cursor.parent
        for group in basemesh.vertex_groups:
            if group.name in source_to_canonical:
                group_to_joint[group.index] = source_to_canonical[group.name]
        influences = 8
        joint_indices = np.zeros((len(mesh.vertices), influences), dtype=np.int16)
        joint_weights = np.zeros((len(mesh.vertices), influences), dtype=np.float32)
        for vertex in mesh.vertices:
            weighted = _collapse_vertex_influences(
                vertex.groups,
                group_to_joint,
                maximum_influences=influences,
            )
            if not weighted:
                raise BodyProviderError(f"Vertex {vertex.index} has no canonical skin influence")
            for slot, (joint_index, weight) in enumerate(weighted):
                joint_indices[vertex.index, slot] = joint_index
                joint_weights[vertex.index, slot] = weight

        head_index = CANONICAL_HUMANOID.index("Head")
        # Calibration is intentionally deferred.  Identity means the GNM
        # model is parented in Head-local space, never that the seam is fitted.
        head_socket = np.eye(4, dtype=np.float32)
        neck_index = CANONICAL_HUMANOID.index("Neck")
        head_origin = global_rest[head_index][:3, 3]
        neck_origin = global_rest[neck_index][:3, 3]
        axis = head_origin - neck_origin
        axis /= np.linalg.norm(axis)
        projection = (vertices - neck_origin) @ axis
        radius = max(float(np.ptp(vertices[:, 1])) * 0.0125, 0.006)
        head_or_neck_weighted = np.any(
            ((joint_indices == head_index) | (joint_indices == neck_index))
            & (joint_weights > 1e-8),
            axis=1,
        )
        seam = np.flatnonzero(
            (np.abs(projection) <= radius) & head_or_neck_weighted
        ).astype(np.int32)
        if seam.size < 3:
            raise BodyProviderError("Could not derive a non-empty neck seam from the hm08 mesh")

        npz_placeholder = "0" * 64
        contract = attachment_contract()
        manifest = {
            "schema_version": BODY_ASSET_SCHEMA,
            "request_id": request["request_id"],
            "provider": {
                "id": PROVIDER_ID,
                "basemesh": "hm08",
                "rig": "default",
                "blender_version": PINNED_BLENDER_VERSION,
                "mpfb_version": PINNED_MPFB_VERSION,
            },
            "coordinate_system": {"handedness": "right", "up_axis": "+Y", "forward_axis": "+Z", "linear_unit": "meter"},
            "skeleton": {
                "schema_version": CANONICAL_HUMANOID.schema_version,
                "joint_names": list(CANONICAL_HUMANOID.names),
                "parents": [joint.parent for joint in CANONICAL_HUMANOID.joints],
            },
            "mesh": {"vertex_count": int(vertices.shape[0]), "triangle_count": int(triangles.shape[0]), "neutral_pose": True},
            "skin": {"max_influences": influences, "weights_normalized": True, "inverse_bind_semantics": "global_bind_matrix @ inverse_bind_matrix = identity"},
            "license": {"asset_spdx": "CC0-1.0", "commercial_use": True, "code_boundary": "MPFB GPL code executed out-of-process; only CC0 asset output crosses boundary"},
            "provenance": {
                "mpfb_release_url": MPFB_RELEASE_URL,
                "blender_url": PINNED_BLENDER_URL,
                "blender_sha256": PINNED_BLENDER_SHA256,
                "mpfb_git_commit": PINNED_MPFB_GIT_COMMIT,
                "mpfb_extension_url": PINNED_MPFB_EXTENSION_URL,
                "mpfb_extension_sha256": PINNED_MPFB_EXTENSION_SHA256,
                "system_assets_url": MAKEHUMAN_SYSTEM_ASSETS_URL,
                "system_assets_sha256": request["provider"]["system_assets_sha256"],
                "makehuman_license_url": MAKEHUMAN_LICENSE_URL,
            },
            "gnm_head_socket": {
                "schema_version": ATTACHMENT_SCHEMA_VERSION,
                "parent_joint": "Head",
                "matrix_semantics": "GNM model to canonical Head-local transform, in meters; identity until calibrated",
                "geometry_policy": "provider head retained for registration; downstream attachment owns replacement",
                "composition_order": contract["composition_order"],
                "body_base_owner": contract["rules"]["body_base_owner"],
                "face_owner": contract["rules"]["face_expression_owner"],
                "attachment_calibrated": False,
            },
            "artifact": {
                "npz_sha256": npz_placeholder,
                "request_sha256": request_sha256,
                "real_provider_export": True,
            },
        }
        arrays = {
            "vertices_m": vertices,
            "triangles": triangles,
            "joint_names": np.asarray(CANONICAL_HUMANOID.names, dtype="U32"),
            "parents": np.asarray([joint.parent for joint in CANONICAL_HUMANOID.joints], dtype=np.int16),
            "local_rest_matrices": local.astype(np.float32),
            "inverse_bind_matrices": inverse.astype(np.float32),
            "joint_indices": joint_indices,
            "joint_weights": joint_weights,
            "gnm_head_socket_matrix": head_socket,
            "neck_seam_vertex_indices": seam,
        }
        return manifest, arrays
    finally:
        evaluated.to_mesh_clear()


def main() -> int:
    request_path, response_path = _parse_paths()
    npz_path: Path | None = None
    manifest_path: Path | None = None
    try:
        request = load_body_provider_request(request_path)
    except BodyProviderError as exc:
        # No trustworthy request_id is available, so do not emit a forged response.
        print(f"Invalid body-provider request: {exc}", file=sys.stderr)
        return 3

    try:
        import bpy
    except ImportError:
        return _write_blocked(
            response_path,
            request["request_id"],
            DependencyIssue("BLENDER_PYTHON_REQUIRED", "Blender Python", PINNED_BLENDER_VERSION, None, "bpy is unavailable; run this script through Blender"),
        )
    observed_blender = ".".join(str(value) for value in bpy.app.version)
    if observed_blender != PINNED_BLENDER_VERSION:
        return _write_blocked(
            response_path,
            request["request_id"],
            DependencyIssue("BLENDER_VERSION_MISMATCH", "Blender", PINNED_BLENDER_VERSION, observed_blender, "Exact Blender LTS patch is required"),
        )
    for issue in (
        _archive_issue(
            "AUTOANIM_MPFB_EXTENSION_ZIP",
            dependency="MPFB extension archive",
            expected_sha256=PINNED_MPFB_EXTENSION_SHA256,
            missing_code="MPFB_EXTENSION_ARCHIVE_MISSING",
            mismatch_code="MPFB_EXTENSION_ARCHIVE_HASH_MISMATCH",
            validator=audit_mpfb_extension_archive,
        ),
        _archive_issue(
            "AUTOANIM_MAKEHUMAN_SYSTEM_ASSETS_ZIP",
            dependency="MakeHuman system-assets archive",
            expected_sha256=request["provider"]["system_assets_sha256"],
            missing_code="MAKEHUMAN_SYSTEM_ASSETS_ARCHIVE_MISSING",
            mismatch_code="MAKEHUMAN_SYSTEM_ASSETS_ARCHIVE_HASH_MISMATCH",
            validator=lambda path: audit_makehuman_system_assets_archive(
                path,
                expected_sha256=request["provider"]["system_assets_sha256"],
            ),
        ),
    ):
        if issue is not None:
            return _write_blocked(response_path, request["request_id"], issue)
    try:
        HumanService = _dynamic_import("mpfb.services.humanservice", "HumanService")
        AssetService = _dynamic_import("mpfb.services.assetservice", "AssetService")
        LocationService = _dynamic_import("mpfb.services.locationservice", "LocationService")
    except ImportError:
        return _write_blocked(
            response_path,
            request["request_id"],
            DependencyIssue("MPFB_EXTENSION_NOT_LOADED", "MPFB extension", PINNED_MPFB_VERSION, None, "Install, enable, and pin the official MPFB extension in this Blender profile"),
        )
    observed_mpfb = _mpfb_installed_version(HumanService)
    if observed_mpfb != PINNED_MPFB_VERSION:
        return _write_blocked(
            response_path,
            request["request_id"],
            DependencyIssue("MPFB_VERSION_MISMATCH", "MPFB extension", PINNED_MPFB_VERSION, observed_mpfb, "Loaded extension manifest does not match the pinned release"),
        )
    system_assets_state = AssetService.check_if_modern_makehuman_system_assets_installed()
    if system_assets_state != (True, True) or not AssetService.system_assets_pack_is_installed():
        return _write_blocked(
            response_path,
            request["request_id"],
            DependencyIssue(
                "MAKEHUMAN_SYSTEM_ASSETS_NOT_INSTALLED",
                "MakeHuman system assets",
                request["provider"]["system_assets_sha256"],
                str(system_assets_state),
                "Install the audited archive in this isolated MPFB profile",
            ),
        )
    attestation_value = os.environ.get("AUTOANIM_BODY_PROFILE_ATTESTATION")
    if not attestation_value or not Path(attestation_value).is_file():
        return _write_blocked(
            response_path,
            request["request_id"],
            DependencyIssue(
                "BODY_PROFILE_ATTESTATION_MISSING",
                "Isolated body-provider profile",
                request["provider"]["system_assets_sha256"],
                None,
                "Install the audited archives with install_blender_body_profile.py",
            ),
        )
    try:
        validate_body_profile_attestation(
            attestation_value,
            extension_root=_mpfb_extension_root(HumanService),
            system_assets_root=Path(LocationService.get_user_home()).resolve() / "data",
            expected_mpfb_archive_sha256=PINNED_MPFB_EXTENSION_SHA256,
            expected_system_assets_archive_sha256=request["provider"][
                "system_assets_sha256"
            ],
        )
    except (BodyProviderError, OSError) as exc:
        return _write_blocked(
            response_path,
            request["request_id"],
            DependencyIssue(
                "BODY_PROFILE_ATTESTATION_MISMATCH",
                "Isolated body-provider profile",
                request["provider"]["system_assets_sha256"],
                None,
                str(exc),
            ),
        )

    try:
        basemesh = HumanService.create_human()
        rig = HumanService.add_builtin_rig(basemesh, "default")
        manifest, arrays = _extract_asset(
            bpy, basemesh, rig, request, sha256_file(request_path)
        )
        output_dir = response_path.parent
        npz_path = output_dir / request["output"]["asset_npz"]
        manifest_path = output_dir / request["output"]["manifest_json"]
        np.savez_compressed(npz_path, **arrays)
        npz_hash = sha256_file(npz_path)
        manifest["artifact"]["npz_sha256"] = npz_hash
        validate_body_asset(manifest, arrays, asset_sha256=npz_hash)
        write_body_provider_json(manifest_path, manifest)
        response = succeeded_body_provider_response(
            request["request_id"],
            manifest_json=manifest_path.name,
            manifest_sha256=sha256_file(manifest_path),
            asset_npz=npz_path.name,
            asset_sha256=npz_hash,
        )
        write_body_provider_json(response_path, response)
        return 0
    except Exception as exc:
        traceback.print_exc()
        _cleanup_failed_export(npz_path, manifest_path)
        return _write_blocked(
            response_path,
            request["request_id"],
            DependencyIssue("PROVIDER_EXPORT_FAILED", "MPFB hm08 export", "validated neutral skinned body", type(exc).__name__, str(exc) or "Export failed"),
        )


if __name__ == "__main__":
    raise SystemExit(main())
