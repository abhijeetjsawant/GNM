"""Fail-closed boundary for externally generated, skinned humanoid bodies.

The provider worker is deliberately separated from the application process:
MPFB is GPL Blender code and its MakeHuman data is an external dependency.  A
worker result is accepted only after the JSON envelope, provenance and every
numeric array have been independently validated here. Content hashes bind the
request and output; they do not authenticate a remote worker or execution host.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping
from urllib.parse import urlparse
import zipfile

import numpy as np

from .body import (
    ATTACHMENT_SCHEMA_VERSION,
    CANONICAL_HUMANOID,
    SKELETON_SCHEMA_VERSION,
    attachment_contract,
)


BODY_PROVIDER_REQUEST_SCHEMA = "autoanim.blender-body-request/1.0"
BODY_PROVIDER_RESPONSE_SCHEMA = "autoanim.blender-body-response/1.0"
BODY_ASSET_SCHEMA = "autoanim.skinned-body-asset/1.0"
PROVIDER_ID = "makehuman_hm08_mpfb"

PINNED_BLENDER_VERSION = "4.5.11"
PINNED_BLENDER_URL = (
    "https://download.blender.org/release/Blender4.5/"
    "blender-4.5.11-macos-arm64.dmg"
)
PINNED_BLENDER_SHA256 = (
    "1fad76c7da9451c7d6db99f1a5ed3c0a1a461d0aa07bf2b639e2fb4804ca4f13"
)
PINNED_MPFB_VERSION = "2.0.16"
PINNED_MPFB_GIT_COMMIT = "f47e9a1bb57a02ec3a33089bfe9e19f85bbf70ec"
PINNED_MPFB_EXTENSION_URL = (
    "https://extensions.blender.org/download/"
    "sha256:b5cdc8b08147e0c6463e4faa01147491b13a0b062f73415363f029debd11c934/"
    "add-on-mpfb-v2.0.16.zip"
)
PINNED_MPFB_EXTENSION_SHA256 = (
    "b5cdc8b08147e0c6463e4faa01147491b13a0b062f73415363f029debd11c934"
)
MAKEHUMAN_SYSTEM_ASSETS_URL = (
    "https://files2.makehumancommunity.org/asset_packs/"
    "makehuman_system_assets/makehuman_system_assets_cc0.zip"
)
MPFB_RELEASE_URL = (
    "https://static.makehumancommunity.org/mpfb/releases/release_2016.html"
)
MAKEHUMAN_LICENSE_URL = "https://static.makehumancommunity.org/about/license.html"

MAX_JSON_BYTES = 256 * 1024
MAX_NPZ_BYTES = 256 * 1024 * 1024
MAX_VERTEX_COUNT = 500_000
MAX_TRIANGLE_COUNT = 1_000_000
MAX_INFLUENCES = 8
MIN_BODY_HEIGHT_M = 0.5
MAX_BODY_HEIGHT_M = 3.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

ARRAY_KEYS = frozenset(
    {
        "vertices_m",
        "triangles",
        "joint_names",
        "parents",
        "local_rest_matrices",
        "inverse_bind_matrices",
        "joint_indices",
        "joint_weights",
        "gnm_head_socket_matrix",
        "neck_seam_vertex_indices",
    }
)


class BodyProviderError(ValueError):
    """An external body request, response, or artifact was not trustworthy."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BodyProviderError(f"Duplicate JSON member: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class DependencyIssue:
    code: str
    dependency: str
    expected: str
    observed: str | None
    detail: str

    def as_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "dependency": self.dependency,
            "expected": self.expected,
            "observed": self.observed,
            "detail": self.detail,
        }


# MPFB's anatomical .L is positive Blender X, whereas AutoAnim's canonical
# Left joints are negative X.  The explicit mapping records that label swap;
# the worker never guesses it from names.
DEFAULT_MPFB_JOINT_MAP: dict[str, str] = {
    "Root": "root",
    "Hips": "spine05",
    "Spine": "spine04",
    "Chest": "spine03",
    "UpperChest": "spine01",
    "Neck": "neck01",
    "Head": "head",
    "LeftEye": "eye.R",
    "RightEye": "eye.L",
    "LeftShoulder": "clavicle.R",
    "LeftUpperArm": "upperarm01.R",
    "LeftLowerArm": "lowerarm01.R",
    "LeftHand": "wrist.R",
    "RightShoulder": "clavicle.L",
    "RightUpperArm": "upperarm01.L",
    "RightLowerArm": "lowerarm01.L",
    "RightHand": "wrist.L",
    "LeftUpperLeg": "upperleg01.R",
    "LeftLowerLeg": "lowerleg01.R",
    "LeftFoot": "foot.R",
    "LeftToes": "toe1-1.R",
    "RightUpperLeg": "upperleg01.L",
    "RightLowerLeg": "lowerleg01.L",
    "RightFoot": "foot.L",
    "RightToes": "toe1-1.L",
}


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BodyProviderError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise BodyProviderError(
            f"{label} fields mismatch; missing={missing}, unknown={unknown}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise BodyProviderError(f"{label} must be a lowercase SHA-256")
    return value


def _https(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise BodyProviderError(f"{label} must be an HTTPS URL")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise BodyProviderError(f"{label} must be an HTTPS URL without credentials")
    return value


def _safe_filename(value: Any, suffix: str, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise BodyProviderError(f"{label} must be a basename")
    if not value.endswith(suffix) or value in {suffix, ".."}:
        raise BodyProviderError(f"{label} must end in {suffix}")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_body_provider_request(
    request_id: str,
    *,
    system_assets_sha256: str,
    asset_npz: str = "neutral-body.npz",
    manifest_json: str = "neutral-body.json",
) -> dict[str, Any]:
    """Build the pinned hm08/MPFB request; caller must pin the asset-pack bytes."""

    request = {
        "schema_version": BODY_PROVIDER_REQUEST_SCHEMA,
        "request_id": request_id,
        "provider": {
            "id": PROVIDER_ID,
            "basemesh": "hm08",
            "rig": "default",
            "blender_version": PINNED_BLENDER_VERSION,
            "blender_url": PINNED_BLENDER_URL,
            "blender_sha256": PINNED_BLENDER_SHA256,
            "mpfb_version": PINNED_MPFB_VERSION,
            "mpfb_git_commit": PINNED_MPFB_GIT_COMMIT,
            "mpfb_extension_url": PINNED_MPFB_EXTENSION_URL,
            "mpfb_extension_sha256": PINNED_MPFB_EXTENSION_SHA256,
            "system_assets_url": MAKEHUMAN_SYSTEM_ASSETS_URL,
            "system_assets_sha256": system_assets_sha256,
        },
        "skeleton": {
            "schema_version": SKELETON_SCHEMA_VERSION,
            "joint_map": dict(DEFAULT_MPFB_JOINT_MAP),
        },
        "output": {"asset_npz": asset_npz, "manifest_json": manifest_json},
    }
    validate_body_provider_request(request)
    return request


def validate_body_provider_request(request: Any) -> dict[str, Any]:
    root = _exact_keys(
        request,
        {"schema_version", "request_id", "provider", "skeleton", "output"},
        "request",
    )
    if root["schema_version"] != BODY_PROVIDER_REQUEST_SCHEMA:
        raise BodyProviderError("Unsupported body-provider request schema")
    request_id = root["request_id"]
    if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
        raise BodyProviderError("request_id is not a safe identifier")

    provider = _exact_keys(
        root["provider"],
        {
            "id",
            "basemesh",
            "rig",
            "blender_version",
            "blender_url",
            "blender_sha256",
            "mpfb_version",
            "mpfb_git_commit",
            "mpfb_extension_url",
            "mpfb_extension_sha256",
            "system_assets_url",
            "system_assets_sha256",
        },
        "request.provider",
    )
    pinned = {
        "id": PROVIDER_ID,
        "basemesh": "hm08",
        "rig": "default",
        "blender_version": PINNED_BLENDER_VERSION,
        "blender_url": PINNED_BLENDER_URL,
        "blender_sha256": PINNED_BLENDER_SHA256,
        "mpfb_version": PINNED_MPFB_VERSION,
        "mpfb_git_commit": PINNED_MPFB_GIT_COMMIT,
        "mpfb_extension_url": PINNED_MPFB_EXTENSION_URL,
        "mpfb_extension_sha256": PINNED_MPFB_EXTENSION_SHA256,
        "system_assets_url": MAKEHUMAN_SYSTEM_ASSETS_URL,
    }
    for key, expected in pinned.items():
        if provider[key] != expected:
            raise BodyProviderError(f"request.provider.{key} is not the pinned value")
    _sha(provider["system_assets_sha256"], "request.provider.system_assets_sha256")

    skeleton = _exact_keys(
        root["skeleton"], {"schema_version", "joint_map"}, "request.skeleton"
    )
    if skeleton["schema_version"] != SKELETON_SCHEMA_VERSION:
        raise BodyProviderError("Request uses an unsupported skeleton schema")
    mapping = skeleton["joint_map"]
    if not isinstance(mapping, dict) or set(mapping) != set(CANONICAL_HUMANOID.names):
        raise BodyProviderError("joint_map keys must be the canonical 25 joints")
    if mapping != DEFAULT_MPFB_JOINT_MAP:
        raise BodyProviderError("joint_map does not match the reviewed MPFB default-rig map")
    if len(set(mapping.values())) != len(mapping) or not all(
        isinstance(value, str) and value for value in mapping.values()
    ):
        raise BodyProviderError("joint_map source bones must be unique non-empty strings")

    output = _exact_keys(
        root["output"], {"asset_npz", "manifest_json"}, "request.output"
    )
    _safe_filename(output["asset_npz"], ".npz", "request.output.asset_npz")
    _safe_filename(output["manifest_json"], ".json", "request.output.manifest_json")
    if output["asset_npz"].removesuffix(".npz") != output["manifest_json"].removesuffix(
        ".json"
    ):
        raise BodyProviderError("Output artifact basenames must match")
    return root


def load_body_provider_request(path: str | os.PathLike[str]) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file() or file_path.stat().st_size > MAX_JSON_BYTES:
        raise BodyProviderError("Request JSON is missing or exceeds 256 KiB")
    try:
        request = json.loads(
            file_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                BodyProviderError(f"Non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BodyProviderError("Request is not valid UTF-8 JSON") from exc
    return validate_body_provider_request(request)


def write_body_provider_json(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    data = _canonical_json_bytes(value)
    if len(data) > MAX_JSON_BYTES:
        raise BodyProviderError("Body-provider JSON exceeds 256 KiB")
    Path(path).write_bytes(data + b"\n")


def blocked_body_provider_response(
    request_id: str, issues: list[DependencyIssue]
) -> dict[str, Any]:
    if not issues:
        raise BodyProviderError("A blocked response requires at least one dependency issue")
    response = {
        "schema_version": BODY_PROVIDER_RESPONSE_SCHEMA,
        "request_id": request_id,
        "status": "blocked",
        "production_validated": False,
        "artifacts": None,
        "dependency_issues": [issue.as_dict() for issue in issues],
    }
    validate_body_provider_response(response)
    return response


def succeeded_body_provider_response(
    request_id: str,
    *,
    manifest_json: str,
    manifest_sha256: str,
    asset_npz: str,
    asset_sha256: str,
) -> dict[str, Any]:
    response = {
        "schema_version": BODY_PROVIDER_RESPONSE_SCHEMA,
        "request_id": request_id,
        "status": "succeeded",
        # This phase validates the provider export, not visual head/body fit.
        "production_validated": False,
        "artifacts": {
            "manifest_json": manifest_json,
            "manifest_sha256": manifest_sha256,
            "asset_npz": asset_npz,
            "asset_sha256": asset_sha256,
        },
        "dependency_issues": [],
    }
    validate_body_provider_response(response)
    return response


def validate_body_provider_response(response: Any) -> dict[str, Any]:
    root = _exact_keys(
        response,
        {
            "schema_version",
            "request_id",
            "status",
            "production_validated",
            "artifacts",
            "dependency_issues",
        },
        "response",
    )
    if root["schema_version"] != BODY_PROVIDER_RESPONSE_SCHEMA:
        raise BodyProviderError("Unsupported body-provider response schema")
    if not isinstance(root["request_id"], str) or not _REQUEST_ID_RE.fullmatch(
        root["request_id"]
    ):
        raise BodyProviderError("response.request_id is not safe")
    if root["production_validated"] is not False:
        raise BodyProviderError("Provider export cannot claim production validation")
    if root["status"] not in {"blocked", "succeeded"}:
        raise BodyProviderError("response.status must be blocked or succeeded")
    issues = root["dependency_issues"]
    if not isinstance(issues, list):
        raise BodyProviderError("response.dependency_issues must be an array")
    for index, issue_value in enumerate(issues):
        issue = _exact_keys(
            issue_value,
            {"code", "dependency", "expected", "observed", "detail"},
            f"response.dependency_issues[{index}]",
        )
        for key in ("code", "dependency", "expected", "detail"):
            if not isinstance(issue[key], str) or not issue[key]:
                raise BodyProviderError(f"Dependency issue {key} must be non-empty")
        if issue["observed"] is not None and not isinstance(issue["observed"], str):
            raise BodyProviderError("Dependency issue observed must be text or null")
    if root["status"] == "blocked":
        if root["artifacts"] is not None or not issues:
            raise BodyProviderError("Blocked response must have issues and no artifacts")
    else:
        if issues:
            raise BodyProviderError("Succeeded response cannot retain dependency issues")
        artifacts = _exact_keys(
            root["artifacts"],
            {"manifest_json", "manifest_sha256", "asset_npz", "asset_sha256"},
            "response.artifacts",
        )
        _safe_filename(artifacts["manifest_json"], ".json", "manifest_json")
        _safe_filename(artifacts["asset_npz"], ".npz", "asset_npz")
        _sha(artifacts["manifest_sha256"], "manifest_sha256")
        _sha(artifacts["asset_sha256"], "asset_sha256")
    return root


def _require_array(
    arrays: Mapping[str, np.ndarray], key: str, shape: tuple[int | None, ...], kinds: str
) -> np.ndarray:
    value = np.asarray(arrays[key])
    if value.dtype.kind not in kinds or value.ndim != len(shape):
        raise BodyProviderError(f"{key} has invalid dtype or rank")
    if any(expected is not None and value.shape[i] != expected for i, expected in enumerate(shape)):
        raise BodyProviderError(f"{key} has invalid shape {value.shape}")
    return value


def _validate_rigid_matrices(value: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(value)):
        raise BodyProviderError(f"{label} contains non-finite values")
    expected_row = np.broadcast_to([0.0, 0.0, 0.0, 1.0], value[..., 3, :].shape)
    if not np.allclose(value[..., 3, :], expected_row, atol=1e-6):
        raise BodyProviderError(f"{label} is not affine")
    rotation = value[..., :3, :3]
    identity = np.broadcast_to(np.eye(3), rotation.shape)
    if not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, identity, atol=2e-4):
        raise BodyProviderError(f"{label} contains scale or shear")
    if not np.allclose(np.linalg.det(rotation), 1.0, atol=2e-4):
        raise BodyProviderError(f"{label} changes handedness")


def _global_rest_matrices(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    output = np.empty_like(local, dtype=np.float64)
    for index, parent in enumerate(parents.tolist()):
        output[index] = local[index] if parent == -1 else output[parent] @ local[index]
    return output


def validate_body_asset(
    manifest: Any, arrays: Mapping[str, np.ndarray], *, asset_sha256: str | None = None
) -> dict[str, Any]:
    """Validate one decoded manifest/NPZ pair without trusting Blender or MPFB."""

    root = _exact_keys(
        manifest,
        {
            "schema_version",
            "request_id",
            "provider",
            "coordinate_system",
            "skeleton",
            "mesh",
            "skin",
            "license",
            "provenance",
            "gnm_head_socket",
            "artifact",
        },
        "body manifest",
    )
    if root["schema_version"] != BODY_ASSET_SCHEMA:
        raise BodyProviderError("Unsupported body asset schema")
    if not isinstance(root["request_id"], str) or not _REQUEST_ID_RE.fullmatch(
        root["request_id"]
    ):
        raise BodyProviderError("Manifest request_id is not safe")

    provider = _exact_keys(
        root["provider"],
        {"id", "basemesh", "rig", "blender_version", "mpfb_version"},
        "manifest.provider",
    )
    expected_provider = {
        "id": PROVIDER_ID,
        "basemesh": "hm08",
        "rig": "default",
        "blender_version": PINNED_BLENDER_VERSION,
        "mpfb_version": PINNED_MPFB_VERSION,
    }
    if provider != expected_provider:
        raise BodyProviderError("Manifest provider is not the pinned hm08/MPFB provider")

    coordinates = _exact_keys(
        root["coordinate_system"],
        {"handedness", "up_axis", "forward_axis", "linear_unit"},
        "manifest.coordinate_system",
    )
    if coordinates != {
        "handedness": "right",
        "up_axis": "+Y",
        "forward_axis": "+Z",
        "linear_unit": "meter",
    }:
        raise BodyProviderError("Body coordinates must be right-handed +Y-up +Z-forward meters")

    skeleton = _exact_keys(
        root["skeleton"], {"schema_version", "joint_names", "parents"}, "manifest.skeleton"
    )
    canonical_parents = [joint.parent for joint in CANONICAL_HUMANOID.joints]
    if (
        skeleton["schema_version"] != SKELETON_SCHEMA_VERSION
        or skeleton["joint_names"] != list(CANONICAL_HUMANOID.names)
        or skeleton["parents"] != canonical_parents
    ):
        raise BodyProviderError("Manifest skeleton is not the canonical 25-joint skeleton")

    mesh = _exact_keys(
        root["mesh"], {"vertex_count", "triangle_count", "neutral_pose"}, "manifest.mesh"
    )
    if mesh["neutral_pose"] is not True:
        raise BodyProviderError("Body mesh must be exported in neutral pose")
    if type(mesh["vertex_count"]) is not int or not 3 <= mesh["vertex_count"] <= MAX_VERTEX_COUNT:
        raise BodyProviderError("Body vertex_count is outside limits")
    if type(mesh["triangle_count"]) is not int or not 1 <= mesh["triangle_count"] <= MAX_TRIANGLE_COUNT:
        raise BodyProviderError("Body triangle_count is outside limits")

    skin = _exact_keys(
        root["skin"],
        {"max_influences", "weights_normalized", "inverse_bind_semantics"},
        "manifest.skin",
    )
    if (
        type(skin["max_influences"]) is not int
        or not 1 <= skin["max_influences"] <= MAX_INFLUENCES
        or skin["weights_normalized"] is not True
        or skin["inverse_bind_semantics"] != "global_bind_matrix @ inverse_bind_matrix = identity"
    ):
        raise BodyProviderError("Manifest skin contract is invalid")

    license_value = _exact_keys(
        root["license"],
        {"asset_spdx", "commercial_use", "code_boundary"},
        "manifest.license",
    )
    if license_value != {
        "asset_spdx": "CC0-1.0",
        "commercial_use": True,
        "code_boundary": "MPFB GPL code executed out-of-process; only CC0 asset output crosses boundary",
    }:
        raise BodyProviderError("Body asset license/provenance boundary is not explicit")

    provenance = _exact_keys(
        root["provenance"],
        {
            "mpfb_release_url",
            "blender_url",
            "blender_sha256",
            "mpfb_git_commit",
            "mpfb_extension_url",
            "mpfb_extension_sha256",
            "system_assets_url",
            "system_assets_sha256",
            "makehuman_license_url",
        },
        "manifest.provenance",
    )
    pinned_provenance = {
        "mpfb_release_url": MPFB_RELEASE_URL,
        "blender_url": PINNED_BLENDER_URL,
        "blender_sha256": PINNED_BLENDER_SHA256,
        "mpfb_git_commit": PINNED_MPFB_GIT_COMMIT,
        "mpfb_extension_url": PINNED_MPFB_EXTENSION_URL,
        "mpfb_extension_sha256": PINNED_MPFB_EXTENSION_SHA256,
        "system_assets_url": MAKEHUMAN_SYSTEM_ASSETS_URL,
        "makehuman_license_url": MAKEHUMAN_LICENSE_URL,
    }
    for key, expected in pinned_provenance.items():
        if provenance[key] != expected:
            raise BodyProviderError(f"Manifest provenance {key} is not pinned")
    for key in (
        "mpfb_release_url",
        "blender_url",
        "mpfb_extension_url",
        "system_assets_url",
        "makehuman_license_url",
    ):
        _https(provenance[key], f"manifest.provenance.{key}")
    _sha(provenance["system_assets_sha256"], "manifest.provenance.system_assets_sha256")

    socket = _exact_keys(
        root["gnm_head_socket"],
        {
            "schema_version",
            "parent_joint",
            "matrix_semantics",
            "geometry_policy",
            "composition_order",
            "body_base_owner",
            "face_owner",
            "attachment_calibrated",
        },
        "manifest.gnm_head_socket",
    )
    contract = attachment_contract()
    expected_socket = {
        "schema_version": ATTACHMENT_SCHEMA_VERSION,
        "parent_joint": "Head",
        "matrix_semantics": "GNM model to canonical Head-local transform, in meters; identity until calibrated",
        "geometry_policy": "provider head retained for registration; downstream attachment owns replacement",
        "composition_order": contract["composition_order"],
        "body_base_owner": contract["rules"]["body_base_owner"],
        "face_owner": contract["rules"]["face_expression_owner"],
        "attachment_calibrated": False,
    }
    if socket != expected_socket:
        raise BodyProviderError("GNM head socket ownership/calibration contract is invalid")

    artifact = _exact_keys(
        root["artifact"],
        {"npz_sha256", "request_sha256", "real_provider_export"},
        "manifest.artifact",
    )
    _sha(artifact["npz_sha256"], "manifest.artifact.npz_sha256")
    _sha(artifact["request_sha256"], "manifest.artifact.request_sha256")
    if artifact["real_provider_export"] is not True:
        raise BodyProviderError("Production provider artifact must identify a real external export")
    if asset_sha256 is not None and artifact["npz_sha256"] != _sha(asset_sha256, "asset_sha256"):
        raise BodyProviderError("NPZ hash does not match the manifest")

    if set(arrays) != ARRAY_KEYS:
        raise BodyProviderError(
            f"Body NPZ arrays mismatch; missing={sorted(ARRAY_KEYS-set(arrays))}, "
            f"unknown={sorted(set(arrays)-ARRAY_KEYS)}"
        )
    vertex_count = mesh["vertex_count"]
    triangle_count = mesh["triangle_count"]
    joint_count = len(CANONICAL_HUMANOID.joints)
    influences = skin["max_influences"]
    vertices = _require_array(arrays, "vertices_m", (vertex_count, 3), "f")
    triangles = _require_array(arrays, "triangles", (triangle_count, 3), "iu")
    names = _require_array(arrays, "joint_names", (joint_count,), "US")
    parents = _require_array(arrays, "parents", (joint_count,), "iu")
    local = _require_array(arrays, "local_rest_matrices", (joint_count, 4, 4), "f")
    inverse = _require_array(arrays, "inverse_bind_matrices", (joint_count, 4, 4), "f")
    indices = _require_array(arrays, "joint_indices", (vertex_count, influences), "iu")
    weights = _require_array(arrays, "joint_weights", (vertex_count, influences), "f")
    socket_matrix = _require_array(arrays, "gnm_head_socket_matrix", (4, 4), "f")
    seam = _require_array(arrays, "neck_seam_vertex_indices", (None,), "iu")

    if not np.all(np.isfinite(vertices)):
        raise BodyProviderError("vertices_m contains non-finite values")
    extents = np.ptp(vertices.astype(np.float64), axis=0)
    if not MIN_BODY_HEIGHT_M <= extents[1] <= MAX_BODY_HEIGHT_M or np.any(extents <= 1e-4):
        raise BodyProviderError("Body mesh bounds are not plausible meters")
    if np.any(triangles < 0) or np.any(triangles >= vertex_count):
        raise BodyProviderError("Triangle indices are out of range")
    if np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
        raise BodyProviderError("Body mesh contains repeated-index triangles")
    a, b, c = (vertices[triangles[:, index]].astype(np.float64) for index in range(3))
    if np.any(np.linalg.norm(np.cross(b - a, c - a), axis=1) <= 1e-12):
        raise BodyProviderError("Body mesh contains zero-area triangles")

    if tuple(str(value) for value in names.tolist()) != CANONICAL_HUMANOID.names:
        raise BodyProviderError("NPZ joint names do not match the canonical joint order")
    if parents.tolist() != canonical_parents:
        raise BodyProviderError("NPZ parents do not match the canonical hierarchy")
    _validate_rigid_matrices(local, "local_rest_matrices")
    _validate_rigid_matrices(inverse, "inverse_bind_matrices")
    global_rest = _global_rest_matrices(local.astype(np.float64), parents)
    expected_identity = np.broadcast_to(np.eye(4), global_rest.shape)
    if not np.allclose(global_rest @ inverse, expected_identity, atol=2e-4):
        raise BodyProviderError("Inverse bind matrices are inconsistent with rest transforms")

    if np.any(indices < 0) or np.any(indices >= joint_count):
        raise BodyProviderError("Skin joint index is out of range")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or np.any(weights > 1.0):
        raise BodyProviderError("Skin weights must be finite values in [0,1]")
    if not np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4):
        raise BodyProviderError("Skin weights must sum to one per vertex")
    for row_indices, row_weights in zip(indices.tolist(), weights.tolist(), strict=True):
        active = [index for index, weight in zip(row_indices, row_weights, strict=True) if weight > 1e-8]
        if len(active) != len(set(active)):
            raise BodyProviderError("A vertex has duplicate active joint influences")

    _validate_rigid_matrices(socket_matrix[None, ...], "gnm_head_socket_matrix")
    if not np.allclose(socket_matrix, np.eye(4), atol=1e-7):
        raise BodyProviderError("Uncalibrated GNM head socket matrix must be identity")
    if seam.size < 3 or seam.size > vertex_count or len(set(seam.tolist())) != seam.size:
        raise BodyProviderError("Neck seam must contain at least three unique vertices")
    if np.any(seam < 0) or np.any(seam >= vertex_count):
        raise BodyProviderError("Neck seam vertex index is out of range")
    head = CANONICAL_HUMANOID.index("Head")
    neck = CANONICAL_HUMANOID.index("Neck")
    for vertex_index in seam.tolist():
        active = indices[vertex_index][weights[vertex_index] > 1e-8]
        if head not in active and neck not in active:
            raise BodyProviderError("Neck seam vertices must be influenced by Head or Neck")
    return root


def load_and_validate_body_asset(
    manifest_path: str | os.PathLike[str],
    asset_path: str | os.PathLike[str],
    *,
    expected_request_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    asset_file = Path(asset_path)
    if not manifest_file.is_file() or manifest_file.stat().st_size > MAX_JSON_BYTES:
        raise BodyProviderError("Body manifest is missing or exceeds 256 KiB")
    if not asset_file.is_file() or asset_file.stat().st_size > MAX_NPZ_BYTES:
        raise BodyProviderError("Body NPZ is missing or exceeds the compressed size limit")
    try:
        manifest = json.loads(
            manifest_file.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                BodyProviderError(f"Non-finite JSON number: {value}")
            ),
        )
        with zipfile.ZipFile(asset_file) as zipped:
            expected_names = {f"{key}.npy" for key in ARRAY_KEYS}
            infos = zipped.infolist()
            if (
                {info.filename for info in infos} != expected_names
                or len(infos) != len(expected_names)
                or any(info.flag_bits & 0x1 for info in infos)
                or sum(info.file_size for info in infos) > MAX_NPZ_BYTES
            ):
                raise BodyProviderError(
                    "Body NPZ container layout or expanded size is unsafe"
                )
        with np.load(asset_file, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except BodyProviderError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        raise BodyProviderError("Body artifact cannot be decoded safely") from exc
    validated = validate_body_asset(
        manifest, arrays, asset_sha256=sha256_file(asset_file)
    )
    if (
        expected_request_sha256 is not None
        and validated["artifact"]["request_sha256"]
        != _sha(expected_request_sha256, "expected_request_sha256")
    ):
        raise BodyProviderError("Body manifest is not bound to the provider request")
    return validated


def load_and_validate_body_provider_result(
    request_path: str | os.PathLike[str], response_path: str | os.PathLike[str]
) -> dict[str, Any]:
    """Validate a worker envelope and bind successful artifacts to its request."""

    request_file = Path(request_path)
    response_file = Path(response_path)
    if request_file.parent.resolve() != response_file.parent.resolve():
        raise BodyProviderError("Request and response must be sibling files")
    request = load_body_provider_request(request_file)
    if not response_file.is_file() or response_file.stat().st_size > MAX_JSON_BYTES:
        raise BodyProviderError("Response JSON is missing or exceeds 256 KiB")
    try:
        response = json.loads(
            response_file.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                BodyProviderError(f"Non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BodyProviderError("Response is not valid UTF-8 JSON") from exc
    validate_body_provider_response(response)
    if response["request_id"] != request["request_id"]:
        raise BodyProviderError("Response request_id does not match the request")
    if response["status"] == "blocked":
        return response

    artifacts = response["artifacts"]
    if (
        artifacts["manifest_json"] != request["output"]["manifest_json"]
        or artifacts["asset_npz"] != request["output"]["asset_npz"]
    ):
        raise BodyProviderError("Response artifact names do not match the request")
    manifest_path = response_file.parent / artifacts["manifest_json"]
    asset_path = response_file.parent / artifacts["asset_npz"]
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != artifacts["manifest_sha256"]
    ):
        raise BodyProviderError("Response manifest hash does not match its artifact")
    if not asset_path.is_file() or sha256_file(asset_path) != artifacts["asset_sha256"]:
        raise BodyProviderError("Response NPZ hash does not match its artifact")
    manifest = load_and_validate_body_asset(
        manifest_path,
        asset_path,
        expected_request_sha256=sha256_file(request_file),
    )
    if manifest["request_id"] != request["request_id"]:
        raise BodyProviderError("Body manifest request_id does not match the request")
    if (
        manifest["provenance"]["system_assets_sha256"]
        != request["provider"]["system_assets_sha256"]
    ):
        raise BodyProviderError(
            "Body manifest system-assets digest does not match the request"
        )
    return response


def audit_body_provider_dependencies(
    blender_executable: str | os.PathLike[str],
    *,
    mpfb_extension_zip: str | os.PathLike[str] | None = None,
    system_assets_zip: str | os.PathLike[str] | None = None,
    system_assets_sha256: str | None = None,
    timeout_seconds: float = 15.0,
) -> list[DependencyIssue]:
    """Inspect local provider inputs without installing add-ons or changing Blender."""

    issues: list[DependencyIssue] = []
    blender = Path(blender_executable)
    if not blender.is_file():
        issues.append(
            DependencyIssue(
                "BLENDER_MISSING", "Blender", PINNED_BLENDER_VERSION, None, "Executable not found"
            )
        )
    else:
        try:
            completed = subprocess.run(
                [str(blender), "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            first_line = (completed.stdout or completed.stderr).splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError, IndexError) as exc:
            issues.append(
                DependencyIssue(
                    "BLENDER_UNUSABLE",
                    "Blender",
                    PINNED_BLENDER_VERSION,
                    None,
                    f"Could not query version: {type(exc).__name__}",
                )
            )
        else:
            observed = first_line.removeprefix("Blender ").split()[0]
            if completed.returncode != 0 or observed != PINNED_BLENDER_VERSION:
                issues.append(
                    DependencyIssue(
                        "BLENDER_VERSION_MISMATCH",
                        "Blender",
                        PINNED_BLENDER_VERSION,
                        observed,
                        "Exact Blender LTS patch is required for reproducible export",
                    )
                )

    def audit_archive(
        path_value: str | os.PathLike[str] | None,
        dependency: str,
        expected_hash: str | None,
        missing_code: str,
        hash_code: str,
    ) -> None:
        if path_value is None or not Path(path_value).is_file():
            issues.append(
                DependencyIssue(missing_code, dependency, expected_hash or "caller-pinned SHA-256", None, "Archive not found")
            )
            return
        observed_hash = sha256_file(path_value)
        if expected_hash is None or not _SHA256_RE.fullmatch(expected_hash):
            issues.append(
                DependencyIssue(
                    "DEPENDENCY_HASH_UNPINNED",
                    dependency,
                    "caller-pinned SHA-256",
                    observed_hash,
                    "Provider refuses an archive without an expected digest",
                )
            )
        elif observed_hash != expected_hash:
            issues.append(
                DependencyIssue(hash_code, dependency, expected_hash, observed_hash, "Archive digest mismatch")
            )

    audit_archive(
        mpfb_extension_zip,
        "MPFB extension",
        PINNED_MPFB_EXTENSION_SHA256,
        "MPFB_EXTENSION_MISSING",
        "MPFB_EXTENSION_HASH_MISMATCH",
    )
    audit_archive(
        system_assets_zip,
        "MakeHuman system assets",
        system_assets_sha256,
        "MAKEHUMAN_SYSTEM_ASSETS_MISSING",
        "MAKEHUMAN_SYSTEM_ASSETS_HASH_MISMATCH",
    )
    return issues
