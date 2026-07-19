from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _worker_module():
    path = Path(__file__).parents[1] / "scripts" / "blender_body_worker.py"
    spec = importlib.util.spec_from_file_location("autoanim_test_blender_body_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collapsed_source_bones_are_aggregated_before_truncation() -> None:
    worker = _worker_module()
    source_groups = [
        SimpleNamespace(group=0, weight=0.25),
        SimpleNamespace(group=1, weight=0.35),
        SimpleNamespace(group=2, weight=0.30),
        SimpleNamespace(group=3, weight=0.10),
    ]

    influences = worker._collapse_vertex_influences(
        source_groups,
        {0: 4, 1: 4, 2: 9, 3: 12},
        maximum_influences=2,
    )

    assert [joint for joint, _ in influences] == [4, 9]
    assert influences[0][1] == pytest.approx(0.6 / 0.9)
    assert influences[1][1] == pytest.approx(0.3 / 0.9)
    assert len({joint for joint, weight in influences if weight > 0.0}) == len(influences)
    assert sum(weight for _, weight in influences) == pytest.approx(1.0)


def test_invalid_and_unmapped_source_weights_cannot_create_an_influence() -> None:
    worker = _worker_module()
    source_groups = [
        SimpleNamespace(group=0, weight=0.0),
        SimpleNamespace(group=1, weight=-0.2),
        SimpleNamespace(group=2, weight=float("nan")),
        SimpleNamespace(group=99, weight=1.0),
    ]

    assert worker._collapse_vertex_influences(
        source_groups,
        {0: 1, 1: 2, 2: 3},
        maximum_influences=8,
    ) == []


def test_collapsed_weights_are_finite_unique_and_normalized() -> None:
    worker = _worker_module()
    source_groups = [
        SimpleNamespace(group=index, weight=1.0 / (index + 1))
        for index in range(20)
    ]
    mapping = {index: index // 2 for index in range(20)}

    influences = worker._collapse_vertex_influences(
        source_groups,
        mapping,
        maximum_influences=8,
    )
    joints = [joint for joint, _ in influences]
    weights = np.asarray([weight for _, weight in influences])

    assert len(influences) == 8
    assert len(joints) == len(set(joints))
    assert np.isfinite(weights).all()
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1.0e-12)


def test_failed_export_cleanup_is_narrow_and_idempotent(tmp_path: Path) -> None:
    worker = _worker_module()
    asset = tmp_path / "neutral-body.npz"
    manifest = tmp_path / "neutral-body.json"
    unrelated = tmp_path / "artist-notes.txt"
    asset.write_bytes(b"unvalidated")
    manifest.write_bytes(b"unvalidated")
    unrelated.write_text("keep", encoding="utf-8")

    worker._cleanup_failed_export(asset, manifest, None)
    worker._cleanup_failed_export(asset, manifest)

    assert not asset.exists()
    assert not manifest.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
