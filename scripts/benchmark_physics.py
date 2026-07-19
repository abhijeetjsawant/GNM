#!/usr/bin/env python3
"""Benchmark the opt-in Rust P0 core with real GNM topology and targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.physics import PhysicsSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, help="Exact release dylib path")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--kernel", choices=("auto", "scalar", "stable", "neon"), default="auto"
    )
    return parser


def _actual_gnm_targets(adapter: GNMAdapter, frames: int) -> np.ndarray:
    phase = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False, dtype=np.float32)
    expressions = np.zeros((frames, adapter.expression_dim), dtype=np.float32)
    expressions[:, 200] = 0.35 * (0.5 + 0.5 * np.sin(phase))
    expressions[:, 250] = 0.18 * np.maximum(np.sin(phase * 0.5), 0.0)
    expressions[:, 350] = 0.12 * np.maximum(np.sin(phase * 1.5), 0.0)
    return np.ascontiguousarray(adapter.mesh(expression=expressions), dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames < 2 or args.repeats < 1:
        raise SystemExit("--frames must be at least 2 and --repeats must be positive")

    adapter = GNMAdapter()
    triangles = np.ascontiguousarray(adapter.triangles, dtype=np.uint32)
    weights = np.ones(adapter.model.num_vertices, dtype=np.float32)
    targets = _actual_gnm_targets(adapter, args.frames)
    phase = np.linspace(0.0, 2.0 * np.pi, args.frames, endpoint=False, dtype=np.float32)
    accelerations = np.zeros((args.frames, 3), dtype=np.float32)
    accelerations[:, 0] = 9.0 * np.sin(phase)
    accelerations[:, 1] = 4.0 * np.sin(phase * 2.0)

    durations: list[float] = []
    final_report: dict[str, object] | None = None
    for _ in range(args.repeats):
        with PhysicsSimulator(
            triangles,
            weights,
            library_path=args.library,
            threads=args.threads,
            kernel=args.kernel,
        ) as simulator:
            started = time.perf_counter()
            output = simulator.simulate(targets, accelerations)
            durations.append(time.perf_counter() - started)
            final_report = simulator.report()
        if not np.isfinite(output).all() or not final_report["finite"]:
            raise RuntimeError("Native physics produced a nonfinite benchmark result")

    samples_ms = np.asarray(durations, dtype=np.float64) * 1_000.0
    assert final_report is not None
    result = {
        "benchmark": "real-gnm-p0-ctypes",
        "vertices": int(targets.shape[1]),
        "triangles": int(triangles.shape[0]),
        "edges": int(final_report["edge_count"]),
        "frames_per_repeat": args.frames,
        "repeats": args.repeats,
        "threads": args.threads,
        "kernel": final_report["kernel"],
        "library": str(simulator.library_path),
        "total_ms": samples_ms.tolist(),
        "p50_ms": float(np.percentile(samples_ms, 50)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "p50_ms_per_frame": float(np.percentile(samples_ms, 50) / args.frames),
        "p95_ms_per_frame": float(np.percentile(samples_ms, 95) / args.frames),
        "report": final_report,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
