"""The build's default body run must be the asset the corrected joint map produces.

The facing mirror (D1, 2026-09-02) was born in `DEFAULT_MPFB_JOINT_MAP`: `Left*` joints mapped to
MPFB's `.R` bones. The asset under the old default run was that map's output, so after the map
was corrected the run's manifest still bound to the pre-repair request. CLAUDE.md's rule is to
check the SHA chain, never assume it, so this pins three links: the default run's manifest binds
to the request file beside it; that request carries the corrected mapping; and the corrected
mapping is what the code produces today. If any of them breaks, a build with default flags would
put the corrected skeleton on the old asset -- and the only thing that would notice is
`facing_location.py`'s guard, silently, after the fact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _default_body_run() -> Path:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_cm", ROOT / "scripts" / "build_commercial_multiview_comparison.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return Path(module.DEFAULT_BODY_RUN)


@pytest.fixture(scope="module")
def run_dir() -> Path:
    run = _default_body_run()
    if not (run / "neutral-body.json").exists():
        pytest.skip(f"default body run not installed at {run}")
    return run


def test_the_manifest_binds_to_the_request_beside_it(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "neutral-body.json").read_text())
    request_sha = hashlib.sha256((run_dir / "request.json").read_bytes()).hexdigest()
    assert manifest["artifact"]["request_sha256"] == request_sha


def test_the_request_carries_the_corrected_left_right_mapping(run_dir: Path) -> None:
    request = json.loads((run_dir / "request.json").read_text())
    joint_map = request["skeleton"]["joint_map"]
    assert joint_map["LeftUpperArm"].endswith(".L"), joint_map["LeftUpperArm"]
    assert joint_map["RightUpperArm"].endswith(".R"), joint_map["RightUpperArm"]
    assert joint_map["LeftUpperLeg"].endswith(".L") and joint_map["RightUpperLeg"].endswith(".R")


def test_the_request_mapping_is_what_the_code_produces_today(run_dir: Path) -> None:
    from autoanim_gnm.body_provider import DETAILED_MPFB_JOINT_MAP

    request = json.loads((run_dir / "request.json").read_text())
    joint_map = request["skeleton"]["joint_map"]
    for name, bone in DETAILED_MPFB_JOINT_MAP.items():
        assert joint_map.get(name) == bone, (name, joint_map.get(name), bone)


def test_the_run_directory_name_carries_its_request_hash(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "neutral-body.json").read_text())
    assert run_dir.name.endswith(manifest["artifact"]["request_sha256"][:8]), run_dir.name
