from __future__ import annotations

import numpy as np
import pytest

from autoanim_gnm.animated_gltf import _rendered_lip_contact_frames
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.mouth_geometry import (
    MOUTH_BOUNDARY_CYCLE_SHA256,
    MOUTH_BOUNDARY_SET_SHA256,
    boundary_opening_displacement,
    discover_mouth_boundary,
    measure_mouth_boundary,
)


def test_gnm_v3_mouth_boundary_is_topology_bound(adapter: GNMAdapter) -> None:
    topology = discover_mouth_boundary(adapter)
    upper_path = topology.cycle[topology.upper_path_positions]
    lower_path = topology.cycle[topology.lower_path_positions]
    upper_lip = adapter.vertex_group("upper_lip") > 0.5
    lower_lip = adapter.vertex_group("lower_lip") > 0.5

    assert topology.set_sha256 == MOUTH_BOUNDARY_SET_SHA256
    assert topology.cycle_sha256 == MOUTH_BOUNDARY_CYCLE_SHA256
    assert topology.cycle.shape == (58,)
    assert len(np.unique(topology.cycle)) == 58
    assert topology.right_corner_position == 2
    assert topology.left_corner_position == 30
    assert not topology.cycle.flags.writeable
    assert np.count_nonzero(upper_lip[upper_path]) >= len(upper_path) - 2
    assert np.count_nonzero(lower_lip[lower_path]) >= len(lower_path) - 2
    assert not np.any(lower_lip[upper_path[1:-1]])
    assert not np.any(upper_lip[lower_path[1:-1]])


def test_neutral_true_mouth_opening_matches_rendered_boundary(
    adapter: GNMAdapter,
) -> None:
    topology = discover_mouth_boundary(adapter)
    measurement = measure_mouth_boundary(
        adapter.mesh(),
        adapter.compact_template,
        topology,
    )

    assert measurement.signed_central_gap_m * 1000.0 == pytest.approx(
        -0.809968, abs=1.0e-5
    )
    assert measurement.opening_area_m2 * 1.0e6 == pytest.approx(
        20.098234, abs=1.0e-4
    )
    assert measurement.signed_central_gap_interocular == pytest.approx(
        -0.00923793, abs=1.0e-7
    )


def test_boundary_measurement_is_rigid_transform_invariant(
    adapter: GNMAdapter,
) -> None:
    topology = discover_mouth_boundary(adapter)
    mesh = adapter.mesh().astype(np.float64)
    landmarks = np.asarray(adapter.compact_template, dtype=np.float64)
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    translation = np.asarray((0.4, -0.2, 0.7), dtype=np.float64)
    baseline = measure_mouth_boundary(mesh, landmarks, topology)
    transformed = measure_mouth_boundary(
        mesh @ rotation.T + translation,
        landmarks @ rotation.T + translation,
        topology,
    )

    assert transformed.signed_central_gap_interocular == pytest.approx(
        baseline.signed_central_gap_interocular, abs=1.0e-10
    )
    assert transformed.opening_area_interocular_squared == pytest.approx(
        baseline.opening_area_interocular_squared, abs=1.0e-10
    )


def test_boundary_displacement_opens_the_rendered_hole(
    adapter: GNMAdapter,
) -> None:
    topology = discover_mouth_boundary(adapter)
    mesh = adapter.mesh()
    landmarks = np.asarray(adapter.compact_template, dtype=np.float32)
    baseline = measure_mouth_boundary(mesh, landmarks, topology)
    desired = boundary_opening_displacement(
        mesh,
        landmarks,
        topology,
        lift_m=0.004,
    )
    revised = mesh.copy()
    revised[topology.cycle] += desired
    opened = measure_mouth_boundary(revised, landmarks, topology)

    assert opened.signed_central_gap_m >= baseline.signed_central_gap_m + 0.0039
    assert opened.opening_area_m2 > baseline.opening_area_m2


def test_export_contact_gate_uses_rendered_boundary_area(
    adapter: GNMAdapter,
) -> None:
    topology = discover_mouth_boundary(adapter)
    neutral = adapter.mesh()
    contracted = neutral.copy()
    cycle = topology.cycle
    center = np.mean(contracted[cycle], axis=0)
    contracted[cycle] = center + np.float32(0.5) * (
        contracted[cycle] - center
    )
    frames = np.stack((neutral, contracted))
    landmarks = np.repeat(adapter.compact_template[None], len(frames), axis=0)

    contact = _rendered_lip_contact_frames(
        frames,
        landmarks,
        topology,
        gap_threshold_interocular=0.006,
        area_threshold_interocular_squared=0.0015,
    )

    np.testing.assert_array_equal(contact, (False, True))
