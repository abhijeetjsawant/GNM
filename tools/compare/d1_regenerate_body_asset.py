#!/usr/bin/env python3
"""D1 (fix): regenerate the body asset through the PINNED provider, and diff it.

WHAT THIS CLOSES. `tools/compare/d1_asset_relabel.py` DERIVED the repaired asset by
permuting the delivered one. That proves the relabel is self-consistent and proves nothing
about the provider: the derived file still carries `artifact.request_sha256` of the
PRE-repair request, whose joint map is the one with the sides swapped. The only thing that
closes it is a real run of the pinned Blender/MPFB worker under the corrected map, and a
diff showing the two agree.

WHAT IS COMPARED. Every array in the npz, by exact equality where exact equality is owed:
`vertices_m`, `triangles`, `joint_weights` and `joint_indices` are the basemesh and its
skin and must be identical; `joint_names` and `parents` are the contract; the rest and
inverse-bind matrices are float32 from the same rig and are compared to a stated tolerance
rather than bit-for-bit, because MPFB rebuilds them and float accumulation order is not
owed. Any difference beyond that is reported rather than absorbed.

NOTHING IS WRITTEN OVER THE EXISTING RUN. `.cache/autoanim_gnm/body-provider/run/
detailed-hands` is the asset every other lane and the BEFORE arm still use.

    .venv/bin/python tools/compare/d1_regenerate_body_asset.py
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from autoanim_gnm.body_provider import (  # noqa: E402
    CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256,
    default_body_provider_request,
    load_and_validate_body_provider_result,
    write_body_provider_json,
)

CACHE = ROOT / ".cache/autoanim_gnm/body-provider"
DERIVED = ROOT / "artifacts/compare/d1-fix/body-run"
OUT = ROOT / "artifacts/compare/d1-fix/body-run-regenerated"
REPORT = ROOT / "artifacts/compare/d1-fix/asset-regeneration.json"

EXACT = ("vertices_m", "triangles", "joint_indices", "joint_weights",
         "joint_names", "parents", "neck_seam_vertex_indices")
TOLERANT = ("local_rest_matrices", "inverse_bind_matrices", "gnm_head_socket_matrix")


def bootstrap_module():
    """Import the bootstrap script for its mount/profile helpers rather than repeating
    them. Re-implementing the environment or the DMG mount would be a second definition of
    the pinned provider, which is the thing this whole step is about."""
    path = ROOT / "scripts" / "bootstrap_body_provider.py"
    spec = importlib.util.spec_from_file_location("bootstrap_body_provider", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    arguments = parser.parse_args()

    boot = bootstrap_module()
    dmg = CACHE / "downloads" / "blender-4.5.11-macos-arm64.dmg"
    extension = CACHE / "downloads" / "add-on-mpfb-v2.0.16.zip"
    assets = CACHE / "downloads" / "makehuman_system_assets_cc0-mirror1.zip"
    attestation = CACHE / "profile-attestation.json"
    for path in (dmg, extension, assets, attestation):
        if not path.exists():
            raise SystemExit(f"missing pinned input: {path}")

    environment = boot._profile_environment(CACHE)
    executable, mounted = boot._mounted_blender(CACHE, dmg)
    try:
        runtime = boot._verify_blender_runtime(executable, environment)
        arguments.out.mkdir(parents=True, exist_ok=True)
        request_path = arguments.out / "request.json"
        response_path = arguments.out / "response.json"
        # The SAME request_id the delivered run used, so the request digest differs by the
        # joint map and by nothing else.
        request = default_body_provider_request(
            "production-detailed-hands",
            system_assets_sha256=CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256,
            detailed_hands=True,
        )
        write_body_provider_json(request_path, request)
        worker_environment = dict(environment)
        worker_environment.update({
            "AUTOANIM_MPFB_EXTENSION_ZIP": str(extension),
            "AUTOANIM_MAKEHUMAN_SYSTEM_ASSETS_ZIP": str(assets),
            "AUTOANIM_BODY_PROFILE_ATTESTATION": str(attestation),
        })
        subprocess.run(
            [str(executable), "--background", "--python-exit-code", "31", "--python",
             str(ROOT / "scripts" / "blender_body_worker.py"), "--",
             str(request_path), str(response_path)],
            check=True, env=worker_environment)
        response = load_and_validate_body_provider_result(request_path, response_path)
    finally:
        if mounted:
            subprocess.run(["hdiutil", "detach", str(CACHE / "mount")], check=False)

    if response["status"] != "succeeded":
        raise SystemExit(f"provider run did not succeed: {response}")

    fresh_npz = arguments.out / response["artifacts"]["asset_npz"]
    fresh_manifest = json.loads((arguments.out / response["artifacts"]["manifest_json"]).read_text())
    derived_npz = DERIVED / "neutral-body.npz"
    delivered_npz = CACHE / "run" / "detailed-hands" / "neutral-body.npz"
    delivered_manifest = json.loads((CACHE / "run" / "detailed-hands" / "neutral-body.json").read_text())

    fresh = dict(np.load(fresh_npz, allow_pickle=True))
    derived = dict(np.load(derived_npz, allow_pickle=True))

    arrays: dict = {}
    identical = True
    for key in sorted(set(fresh) | set(derived)):
        if key not in fresh or key not in derived:
            arrays[key] = {"present_in_regenerated": key in fresh,
                           "present_in_derived": key in derived, "verdict": "MISSING"}
            identical = False
            continue
        a, b = fresh[key], derived[key]
        if a.shape != b.shape:
            arrays[key] = {"shape_regenerated": list(a.shape), "shape_derived": list(b.shape),
                           "verdict": "SHAPE MISMATCH"}
            identical = False
            continue
        if key in EXACT:
            same = bool(np.array_equal(a, b))
            arrays[key] = {"comparison": "exact", "identical": same,
                           "verdict": "PASS" if same else "FAIL"}
            identical = identical and same
        else:
            delta = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
            same = delta <= 1e-6
            arrays[key] = {"comparison": "max absolute difference",
                           "max_abs_difference": delta,
                           "tolerance": 1e-6,
                           "verdict": "PASS" if same else "FAIL"}
            identical = identical and same

    report = {
        "step": "D1 (fix)",
        "instrument": "tools/compare/d1_regenerate_body_asset.py",
        "question": ("does a real run of the pinned Blender/MPFB worker under the REPAIRED "
                     "joint map produce the asset that was DERIVED by permuting the "
                     "delivered one?"),
        "blender_runtime": runtime,
        "request_id": request["request_id"],
        "joint_map_sides": {
            "regenerated_LeftUpperArm": request["skeleton"]["joint_map"]["LeftUpperArm"],
            "delivered_LeftUpperArm": "upperarm01.R",
        },
        "request_sha256": {
            "regenerated": response["artifacts"]["request_sha256"]
            if "request_sha256" in response.get("artifacts", {})
            else fresh_manifest["artifact"]["request_sha256"],
            "delivered": delivered_manifest["artifact"]["request_sha256"],
            "they_differ_because": "the joint map is part of the request",
        },
        "asset_sha256": {
            "regenerated": sha256(fresh_npz.read_bytes()).hexdigest(),
            "derived_by_permutation": sha256(derived_npz.read_bytes()).hexdigest(),
            "delivered_pre_repair": sha256(delivered_npz.read_bytes()).hexdigest(),
            "note": ("the npz digests are NOT expected to match: np.savez writes its own "
                     "container. The arrays are what must agree, and they are compared "
                     "one by one below."),
        },
        "manifest_binds_to_its_own_request": (
            fresh_manifest["artifact"]["request_sha256"]
            != delivered_manifest["artifact"]["request_sha256"]),
        "arrays": arrays,
        "verdict": "PASS" if identical else "FAIL",
        "reading": (
            "If every array agrees, then the derivation and a real provider run are the "
            "same thing and the derived asset's only defect was its provenance -- which "
            "this run supplies. If any array differs, the derivation was an assumption "
            "about MPFB's symmetry that MPFB does not honour, and the regenerated asset is "
            "the one to ship."),
        "blind_to": (
            "This compares two ASSETS. It says nothing about whether the delivered "
            "character is right -- that is the facing gate's job -- and nothing about the "
            "MakeHuman basemesh itself. It also cannot see a difference the two runs share "
            "because they use the same pinned inputs, which is the point of pinning them."),
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2))
    print(f"wrote {arguments.report}")
    for key, entry in arrays.items():
        print(f"  {entry['verdict']:16s} {key}")
    print(f"  {report['verdict']}  overall")


if __name__ == "__main__":
    main()
