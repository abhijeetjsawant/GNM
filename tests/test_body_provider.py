from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from autoanim_gnm.body import ATTACHMENT_SCHEMA_VERSION, CANONICAL_HUMANOID, attachment_contract
from autoanim_gnm.body_provider import (
    ARRAY_KEYS,
    BODY_ASSET_SCHEMA,
    BODY_PROVIDER_REQUEST_SCHEMA,
    BODY_PROVIDER_RESPONSE_SCHEMA,
    CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256,
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
    audit_body_provider_dependencies,
    blocked_body_provider_response,
    default_body_provider_request,
    load_and_validate_body_asset,
    load_and_validate_body_provider_result,
    load_body_provider_request,
    succeeded_body_provider_response,
    sha256_file,
    validate_body_asset,
    validate_body_provider_request,
    validate_body_provider_response,
    write_body_provider_json,
)


SYSTEM_ASSETS_SHA = CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256


def _structural_fixture() -> tuple[dict, dict[str, np.ndarray]]:
    # This small numeric fixture only exercises the validator. It is never
    # serialized as provider evidence or treated as a production body.
    vertices = np.asarray(
        [
            [-0.3, 0.0, -0.15],
            [0.3, 0.0, -0.15],
            [0.3, 0.0, 0.15],
            [-0.3, 0.0, 0.15],
            [-0.3, 1.7, -0.15],
            [0.3, 1.7, -0.15],
            [0.3, 1.7, 0.15],
            [-0.3, 1.7, 0.15],
        ],
        dtype=np.float32,
    )
    triangles = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )
    joint_count = len(CANONICAL_HUMANOID.joints)
    parents = np.asarray([joint.parent for joint in CANONICAL_HUMANOID.joints], dtype=np.int16)
    local = np.broadcast_to(np.eye(4), (joint_count, 4, 4)).copy()
    for index, joint in enumerate(CANONICAL_HUMANOID.joints):
        local[index, :3, 3] = joint.rest_translation_m
    global_rest = np.empty_like(local)
    for index, parent in enumerate(parents.tolist()):
        global_rest[index] = local[index] if parent == -1 else global_rest[parent] @ local[index]
    inverse = np.linalg.inv(global_rest)

    indices = np.zeros((vertices.shape[0], 4), dtype=np.int16)
    weights = np.zeros((vertices.shape[0], 4), dtype=np.float32)
    indices[:4, 0] = CANONICAL_HUMANOID.index("Hips")
    indices[4:, 0] = CANONICAL_HUMANOID.index("Head")
    weights[:, 0] = 1.0

    contract = attachment_contract()
    manifest = {
        "schema_version": BODY_ASSET_SCHEMA,
        "request_id": "fixture-001",
        "provider": {
            "id": PROVIDER_ID,
            "basemesh": "hm08",
            "rig": "default",
            "blender_version": PINNED_BLENDER_VERSION,
            "mpfb_version": PINNED_MPFB_VERSION,
        },
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "+Z",
            "linear_unit": "meter",
        },
        "skeleton": {
            "schema_version": CANONICAL_HUMANOID.schema_version,
            "joint_names": list(CANONICAL_HUMANOID.names),
            "parents": parents.tolist(),
        },
        "mesh": {
            "vertex_count": int(vertices.shape[0]),
            "triangle_count": int(triangles.shape[0]),
            "neutral_pose": True,
        },
        "skin": {
            "max_influences": 4,
            "weights_normalized": True,
            "inverse_bind_semantics": "global_bind_matrix @ inverse_bind_matrix = identity",
        },
        "license": {
            "asset_spdx": "CC0-1.0",
            "commercial_use": True,
            "code_boundary": "MPFB GPL code executed out-of-process; only CC0 asset output crosses boundary",
        },
        "provenance": {
            "mpfb_release_url": MPFB_RELEASE_URL,
            "blender_url": PINNED_BLENDER_URL,
            "blender_sha256": PINNED_BLENDER_SHA256,
            "mpfb_git_commit": PINNED_MPFB_GIT_COMMIT,
            "mpfb_extension_url": PINNED_MPFB_EXTENSION_URL,
            "mpfb_extension_sha256": PINNED_MPFB_EXTENSION_SHA256,
            "system_assets_url": MAKEHUMAN_SYSTEM_ASSETS_URL,
            "system_assets_sha256": SYSTEM_ASSETS_SHA,
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
            "npz_sha256": "a" * 64,
            "request_sha256": "b" * 64,
            "real_provider_export": True,
        },
    }
    arrays = {
        "vertices_m": vertices,
        "triangles": triangles,
        "joint_names": np.asarray(CANONICAL_HUMANOID.names, dtype="U32"),
        "parents": parents,
        "local_rest_matrices": local.astype(np.float32),
        "inverse_bind_matrices": inverse.astype(np.float32),
        "joint_indices": indices,
        "joint_weights": weights,
        "gnm_head_socket_matrix": np.eye(4, dtype=np.float32),
        "neck_seam_vertex_indices": np.asarray([4, 5, 6, 7], dtype=np.int32),
    }
    assert set(arrays) == ARRAY_KEYS
    return manifest, arrays


def test_default_request_pins_reviewed_official_provider_and_ordered_joint_map() -> None:
    request = default_body_provider_request("body-job_001", system_assets_sha256=SYSTEM_ASSETS_SHA)
    assert request["schema_version"] == BODY_PROVIDER_REQUEST_SCHEMA
    assert request["provider"] == {
        "id": PROVIDER_ID,
        "basemesh": "hm08",
        "rig": "default",
        "blender_version": "4.5.11",
        "blender_url": PINNED_BLENDER_URL,
        "blender_sha256": PINNED_BLENDER_SHA256,
        "mpfb_version": "2.0.16",
        "mpfb_git_commit": PINNED_MPFB_GIT_COMMIT,
        "mpfb_extension_url": PINNED_MPFB_EXTENSION_URL,
        "mpfb_extension_sha256": PINNED_MPFB_EXTENSION_SHA256,
        "system_assets_url": MAKEHUMAN_SYSTEM_ASSETS_URL,
        "system_assets_sha256": SYSTEM_ASSETS_SHA,
    }
    assert tuple(request["skeleton"]["joint_map"]) == CANONICAL_HUMANOID.names
    assert request["skeleton"]["joint_map"]["LeftEye"] == "eye.R"
    assert request["skeleton"]["joint_map"]["RightUpperArm"] == "upperarm01.L"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda request: request.update(extra=True), "fields mismatch"),
        (lambda request: request["provider"].update(blender_version="4.5.10"), "not the pinned"),
        (lambda request: request["provider"].update(system_assets_sha256="not-a-hash"), "SHA-256"),
        (lambda request: request["output"].update(asset_npz="../escape.npz"), "basename"),
        (lambda request: request["output"].update(asset_npz=".npz"), "must end"),
        (lambda request: request["output"].update(manifest_json="different.json"), "basenames must match"),
        (lambda request: request["skeleton"]["joint_map"].update(LeftEye="eye.L"), "reviewed MPFB"),
    ],
)
def test_request_validation_fails_closed(mutation, match: str) -> None:
    request = default_body_provider_request("body-job", system_assets_sha256=SYSTEM_ASSETS_SHA)
    mutation(request)
    with pytest.raises(BodyProviderError, match=match):
        validate_body_provider_request(request)


def test_request_loader_rejects_nonfinite_json_and_oversize(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(BodyProviderError, match="Non-finite"):
        load_body_provider_request(invalid)

    invalid.write_bytes(b" " * (256 * 1024 + 1))
    with pytest.raises(BodyProviderError, match="exceeds"):
        load_body_provider_request(invalid)


def test_request_loader_rejects_duplicate_json_members(tmp_path: Path) -> None:
    invalid = tmp_path / "duplicate.json"
    invalid.write_text(
        '{"schema_version":"autoanim.blender-body-request/1.0",'
        '"schema_version":"autoanim.blender-body-request/1.0"}',
        encoding="utf-8",
    )
    with pytest.raises(BodyProviderError, match="Duplicate JSON member"):
        load_body_provider_request(invalid)


def test_response_contract_never_allows_provider_to_claim_production_readiness() -> None:
    blocked = blocked_body_provider_response(
        "job-1",
        [DependencyIssue("MPFB_EXTENSION_MISSING", "MPFB", "2.0.16", None, "not installed")],
    )
    assert blocked["schema_version"] == BODY_PROVIDER_RESPONSE_SCHEMA
    assert blocked["status"] == "blocked"
    assert blocked["artifacts"] is None

    succeeded = succeeded_body_provider_response(
        "job-1",
        manifest_json="neutral-body.json",
        manifest_sha256="1" * 64,
        asset_npz="neutral-body.npz",
        asset_sha256="2" * 64,
    )
    assert succeeded["status"] == "succeeded"
    assert succeeded["production_validated"] is False
    succeeded["production_validated"] = True
    with pytest.raises(BodyProviderError, match="cannot claim production"):
        validate_body_provider_response(succeeded)


def test_structural_fixture_passes_all_body_asset_gates() -> None:
    manifest, arrays = _structural_fixture()
    validated = validate_body_asset(manifest, arrays)
    assert validated["skeleton"]["joint_names"] == list(CANONICAL_HUMANOID.names)
    assert validated["gnm_head_socket"]["attachment_calibrated"] is False


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda manifest: manifest["coordinate_system"].update(linear_unit="centimeter"), "coordinates"),
        (lambda manifest: manifest["skeleton"]["parents"].__setitem__(6, 4), "canonical"),
        (lambda manifest: manifest["license"].update(asset_spdx="GPL-3.0"), "license"),
        (lambda manifest: manifest["provenance"].update(mpfb_git_commit="0" * 40), "not pinned"),
        (lambda manifest: manifest["gnm_head_socket"].update(parent_joint="Neck"), "socket"),
        (lambda manifest: manifest["gnm_head_socket"].update(attachment_calibrated=True), "socket"),
        (lambda manifest: manifest["artifact"].update(real_provider_export=False), "real external export"),
    ],
)
def test_manifest_semantic_mutations_are_rejected(mutation, match: str) -> None:
    manifest, arrays = _structural_fixture()
    mutation(manifest)
    with pytest.raises(BodyProviderError, match=match):
        validate_body_asset(manifest, arrays)


@pytest.mark.parametrize(
    ("key", "mutator", "match"),
    [
        ("vertices_m", lambda value: value.__setitem__((0, 0), np.nan), "non-finite"),
        ("vertices_m", lambda value: value.__setitem__((slice(None), 1), 0.01), "plausible meters"),
        ("triangles", lambda value: value.__setitem__((0, 2), value[0, 1]), "repeated-index"),
        ("triangles", lambda value: value.__setitem__((0, 2), 999), "out of range"),
        ("parents", lambda value: value.__setitem__(6, 4), "canonical hierarchy"),
        ("inverse_bind_matrices", lambda value: value.__setitem__((6, 0, 3), value[6, 0, 3] + 0.2), "inconsistent"),
        ("joint_indices", lambda value: value.__setitem__((0, 0), 25), "out of range"),
        ("joint_weights", lambda value: value.__setitem__((0, 0), 0.5), "sum to one"),
        ("gnm_head_socket_matrix", lambda value: value.__setitem__((0, 0), 2.0), "scale or shear"),
        ("gnm_head_socket_matrix", lambda value: value.__setitem__((0, 3), 0.1), "must be identity"),
        ("neck_seam_vertex_indices", lambda value: value.__setitem__(0, 0), "Head or Neck"),
    ],
)
def test_numeric_artifact_mutations_are_rejected(key: str, mutator, match: str) -> None:
    manifest, arrays = _structural_fixture()
    arrays[key] = arrays[key].copy()
    mutator(arrays[key])
    with pytest.raises(BodyProviderError, match=match):
        validate_body_asset(manifest, arrays)


def test_duplicate_active_skin_influence_and_unknown_npz_array_are_rejected() -> None:
    manifest, arrays = _structural_fixture()
    arrays["joint_indices"] = arrays["joint_indices"].copy()
    arrays["joint_weights"] = arrays["joint_weights"].copy()
    arrays["joint_indices"][0, :2] = CANONICAL_HUMANOID.index("Hips")
    arrays["joint_weights"][0, :2] = 0.5
    with pytest.raises(BodyProviderError, match="duplicate active"):
        validate_body_asset(manifest, arrays)

    manifest, arrays = _structural_fixture()
    arrays["surprise"] = np.zeros(1)
    with pytest.raises(BodyProviderError, match="arrays mismatch"):
        validate_body_asset(manifest, arrays)


def test_npz_and_manifest_hash_are_verified_after_safe_decode(tmp_path: Path) -> None:
    manifest, arrays = _structural_fixture()
    asset_path = tmp_path / "neutral-body.npz"
    manifest_path = tmp_path / "neutral-body.json"
    np.savez_compressed(asset_path, **arrays)
    manifest["artifact"]["npz_sha256"] = "0" * 64
    write_body_provider_json(manifest_path, manifest)
    with pytest.raises(BodyProviderError, match="hash does not match"):
        load_and_validate_body_asset(manifest_path, asset_path)


def test_success_result_is_bound_to_request_names_ids_and_hashes(tmp_path: Path) -> None:
    request = default_body_provider_request(
        "bound-result", system_assets_sha256=SYSTEM_ASSETS_SHA
    )
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    asset_path = tmp_path / request["output"]["asset_npz"]
    manifest_path = tmp_path / request["output"]["manifest_json"]
    write_body_provider_json(request_path, request)

    manifest, arrays = _structural_fixture()
    manifest["request_id"] = request["request_id"]
    np.savez_compressed(asset_path, **arrays)
    manifest["artifact"]["npz_sha256"] = sha256_file(asset_path)
    manifest["artifact"]["request_sha256"] = sha256_file(request_path)
    write_body_provider_json(manifest_path, manifest)
    response = succeeded_body_provider_response(
        request["request_id"],
        manifest_json=manifest_path.name,
        manifest_sha256=sha256_file(manifest_path),
        asset_npz=asset_path.name,
        asset_sha256=sha256_file(asset_path),
    )
    write_body_provider_json(response_path, response)

    validated = load_and_validate_body_provider_result(request_path, response_path)
    assert validated["status"] == "succeeded"

    manifest["artifact"]["request_sha256"] = "0" * 64
    write_body_provider_json(manifest_path, manifest)
    response["artifacts"]["manifest_sha256"] = sha256_file(manifest_path)
    write_body_provider_json(response_path, response)
    with pytest.raises(BodyProviderError, match="not bound"):
        load_and_validate_body_provider_result(request_path, response_path)


def test_dependency_audit_reports_exact_version_and_archive_failures(tmp_path: Path) -> None:
    blender = tmp_path / "blender"
    blender.write_text("#!/bin/sh\necho 'Blender 4.2.0'\n", encoding="utf-8")
    blender.chmod(0o755)
    mpfb = tmp_path / "mpfb.zip"
    mpfb.write_bytes(b"wrong mpfb bytes")
    assets = tmp_path / "assets.zip"
    assets.write_bytes(b"wrong asset bytes")

    issues = audit_body_provider_dependencies(
        blender,
        mpfb_extension_zip=mpfb,
        system_assets_zip=assets,
        system_assets_sha256=SYSTEM_ASSETS_SHA,
    )
    assert [issue.code for issue in issues] == [
        "BLENDER_VERSION_MISMATCH",
        "MPFB_EXTENSION_HASH_MISMATCH",
        "MAKEHUMAN_SYSTEM_ASSETS_HASH_MISMATCH",
    ]
    assert issues[0].expected == PINNED_BLENDER_VERSION
    assert issues[0].observed == "4.2.0"


def test_worker_run_under_cpython_returns_typed_blender_dependency_block(tmp_path: Path) -> None:
    request = default_body_provider_request("worker-cpython", system_assets_sha256=SYSTEM_ASSETS_SHA)
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    write_body_provider_json(request_path, request)
    worker = Path(__file__).parents[1] / "scripts" / "blender_body_worker.py"

    completed = subprocess.run(
        [sys.executable, str(worker), "--", str(request_path), str(response_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    response = json.loads(response_path.read_text(encoding="utf-8"))
    validate_body_provider_response(response)
    assert response["status"] == "blocked"
    assert response["dependency_issues"][0]["code"] == "BLENDER_PYTHON_REQUIRED"


def test_worker_refuses_unattested_source_archives_before_loading_mpfb(
    tmp_path: Path,
) -> None:
    request = default_body_provider_request(
        "worker-archives", system_assets_sha256=SYSTEM_ASSETS_SHA
    )
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    write_body_provider_json(request_path, request)
    # Minimal bpy stand-in reaches only the dependency gate; MPFB is never
    # imported, generated, or represented by a fake body.
    (tmp_path / "bpy.py").write_text(
        "class _App:\n    version = (4, 5, 11)\napp = _App()\n",
        encoding="utf-8",
    )
    worker = Path(__file__).parents[1] / "scripts" / "blender_body_worker.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    environment.pop("AUTOANIM_MPFB_EXTENSION_ZIP", None)
    environment.pop("AUTOANIM_MAKEHUMAN_SYSTEM_ASSETS_ZIP", None)

    completed = subprocess.run(
        [sys.executable, str(worker), "--", str(request_path), str(response_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert completed.returncode == 2
    response = json.loads(response_path.read_text(encoding="utf-8"))
    validate_body_provider_response(response)
    assert response["dependency_issues"][0]["code"] == "MPFB_EXTENSION_ARCHIVE_MISSING"
