#!/usr/bin/env python3
"""Build one sealed GNM-to-body attachment binding reproducibly.

The resulting files are still preview artifacts until the character's identity,
look-dev and measured collar gates are approved.  This command exists so a
reviewer never has to rely on an opaque, hand-created ``binding.npz``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from autoanim_gnm.body_binding import calibrate_gnm_body_binding
from autoanim_gnm.gnm_adapter import GNMAdapter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BODY_RUN = (
    ROOT / ".cache" / "autoanim_gnm" / "body-provider" / "run" / "detailed-hands-fbd9784b"  # regenerated under the corrected joint map, 2026-09-02
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "n5-1-production-assembly"


def _identity_and_joint_positions(
    adapter: GNMAdapter, identity_path: Path | None
) -> tuple[np.ndarray, np.ndarray]:
    if identity_path is None:
        identity = np.zeros(adapter.identity_dim, dtype=np.float32)
    else:
        identity = np.asarray(
            np.load(identity_path, allow_pickle=False), dtype=np.float32
        )
    if identity.shape != (adapter.identity_dim,) or not np.isfinite(identity).all():
        raise ValueError("GNM identity must be one finite 253-value .npy array")
    model = adapter.model
    positions = np.asarray(model.template_joint_positions, dtype=np.float32)
    positions = positions + np.einsum(
        "i,ijk->jk",
        identity,
        np.asarray(model.joint_identity_basis, dtype=np.float32),
    )
    if positions.shape != (4, 3) or not np.isfinite(positions).all():
        raise ValueError("GNM joint positions are invalid")
    return identity, positions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-run", type=Path, default=DEFAULT_BODY_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--identity", type=Path, help="Optional finite GNM identity .npy"
    )
    parser.add_argument("--gnm-cut-fraction", type=float, default=0.35)
    parser.add_argument("--collar-height-mm", type=float, default=15.0)
    parser.add_argument("--minimum-cut-loop-edge-mm", type=float, default=0.75)
    arguments = parser.parse_args()

    body_run = arguments.body_run.resolve(strict=True)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    adapter = GNMAdapter()
    identity, joints = _identity_and_joint_positions(adapter, arguments.identity)
    manifest = calibrate_gnm_body_binding(
        output / "binding.json",
        output / "binding.npz",
        body_manifest_path=body_run / "neutral-body.json",
        body_asset_path=body_run / "neutral-body.npz",
        gnm_neutral_vertices=adapter.mesh(identity=identity),
        gnm_triangles=adapter.triangles,
        gnm_identity=identity,
        gnm_neck_position=joints[0],
        gnm_head_position=joints[1],
        gnm_cut_fraction=arguments.gnm_cut_fraction,
        collar_height_m=arguments.collar_height_mm / 1000.0,
        minimum_cut_loop_edge_m=arguments.minimum_cut_loop_edge_mm / 1000.0,
    )
    print(
        "binding built:",
        output / "binding.json",
        "loops=",
        manifest["metrics"]["body_neck_loop_vertices"],
        manifest["metrics"]["gnm_neck_loop_vertices"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
