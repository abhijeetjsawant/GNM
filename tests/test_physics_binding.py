from pathlib import Path

import numpy as np
import pytest

from autoanim_gnm.gnm_adapter import EXPECTED, GNMAdapter
from autoanim_gnm.physics import (
    PhysicsError,
    PhysicsInputError,
    PhysicsSimulator,
    find_local_release_library,
)


RELEASE_LIBRARY = find_local_release_library()
pytestmark = pytest.mark.skipif(
    RELEASE_LIBRARY is None,
    reason="AutoAnim physics release dylib is truly absent",
)


def _triangles(adapter: GNMAdapter) -> np.ndarray:
    return np.ascontiguousarray(adapter.triangles, dtype=np.uint32)


def _weights(adapter: GNMAdapter, value: float = 1.0) -> np.ndarray:
    return np.full(adapter.model.num_vertices, value, dtype=np.float32)


def _simulator(
    adapter: GNMAdapter, weights: np.ndarray | None = None
) -> PhysicsSimulator:
    assert RELEASE_LIBRARY is not None
    return PhysicsSimulator(
        _triangles(adapter),
        _weights(adapter) if weights is None else weights,
        library_path=RELEASE_LIBRARY,
        threads=4,
        kernel="auto",
    )


def test_real_gnm_topology_and_report_contract(adapter: GNMAdapter) -> None:
    with _simulator(adapter) as simulator:
        report = simulator.report()

    assert report["vertex_count"] == EXPECTED["vertices"] == 17_821
    assert report["triangle_count"] == EXPECTED["triangles"] == 35_324
    assert report["edge_count"] == 53_135
    assert report["schema_version"] == 1
    assert report["backend"] == "cpu-rayon-target-relative-jacobi-xpbd"
    assert report["kernel"] in {
        "scalar_reference",
        "stable_auto_vectorized",
        "neon_intrinsics",
    }
    assert report["threads"] == 4
    assert report["frame_count"] == 0
    assert report["substeps"] == simulator.config.substeps
    assert report["iterations"] == simulator.config.iterations
    for field in ("topology_sha256", "config_sha256", "input_sha256", "output_sha256"):
        assert len(report[field]) == 64
        int(report[field], 16)
    assert isinstance(report["simd_claim"], bool)
    assert report["fallback_reason"] is None or isinstance(report["fallback_reason"], str)
    assert report["finite"] is True
    assert report["target_relative"] is True
    assert report["externally_accelerated_frames"] == 0


def test_real_gnm_moving_target_is_exact_no_op(adapter: GNMAdapter) -> None:
    expression = np.zeros(adapter.expression_dim, dtype=np.float32)
    expression[200] = np.float32(0.45)
    targets = np.ascontiguousarray(
        np.stack((adapter.mesh(), adapter.mesh(expression=expression))),
        dtype=np.float32,
    )

    with _simulator(adapter) as simulator:
        output = simulator.simulate(targets)
        report = simulator.report()

    np.testing.assert_array_equal(output, targets)
    assert report["frame_count"] == 2
    assert report["max_displacement_m"] == 0.0
    assert report["rms_displacement_m"] == 0.0
    assert report["max_edge_strain"] == 0.0


def test_real_gnm_protected_vertex_is_exact_under_force(adapter: GNMAdapter) -> None:
    protected = np.array([0, 100, 10_000, 17_820], dtype=np.intp)
    weights = _weights(adapter)
    weights[protected] = 0.0
    neutral = adapter.mesh()
    targets = np.ascontiguousarray(np.repeat(neutral[None, :, :], 4, axis=0))
    accelerations = np.ascontiguousarray(
        np.tile(np.array([250.0, -125.0, 80.0], dtype=np.float32), (4, 1))
    )

    with _simulator(adapter, weights) as simulator:
        output = simulator.simulate(targets, accelerations)
        report = simulator.report()

    np.testing.assert_array_equal(output[:, protected], targets[:, protected])
    assert np.any(output[:, 1] != targets[:, 1])
    assert report["pinned_drift_m"] == 0.0
    assert report["externally_accelerated_frames"] == 4


def test_real_gnm_force_stays_finite_and_within_hard_cap(adapter: GNMAdapter) -> None:
    neutral = adapter.mesh()
    targets = np.ascontiguousarray(np.repeat(neutral[None, :, :], 3, axis=0))
    accelerations = np.full((3, 3), 1.0e12, dtype=np.float32)

    with _simulator(adapter) as simulator:
        output = simulator.simulate(targets, accelerations)
        report = simulator.report()
        cap = simulator.config.max_displacement_m

    displacement = np.linalg.norm(output - targets, axis=-1)
    assert np.isfinite(output).all()
    assert float(np.max(displacement)) <= cap + 1.0e-7
    assert report["finite"] is True
    assert report["max_displacement_m"] <= cap + 1.0e-7
    assert report["max_displacement_m"] > 0.0


def test_binding_rejects_implicit_array_conversion_and_use_after_close(
    adapter: GNMAdapter,
) -> None:
    assert RELEASE_LIBRARY is not None
    weights = _weights(adapter)
    with pytest.raises(PhysicsInputError, match="uint32"):
        PhysicsSimulator(
            adapter.triangles,
            weights,
            library_path=RELEASE_LIBRARY,
        )

    simulator = _simulator(adapter)
    neutral = adapter.mesh()
    with pytest.raises(PhysicsInputError, match="float32"):
        simulator.simulate(neutral.astype(np.float64))
    with pytest.raises(PhysicsInputError, match="C-contiguous"):
        simulator.simulate(neutral[:, ::-1])
    simulator.close()
    simulator.close()
    with pytest.raises(PhysicsError, match="closed"):
        simulator.report()


def test_supplied_missing_library_never_falls_back(adapter: GNMAdapter, tmp_path: Path) -> None:
    with pytest.raises(PhysicsError, match="absent"):
        PhysicsSimulator(
            _triangles(adapter),
            _weights(adapter),
            library_path=tmp_path / "missing.dylib",
        )
