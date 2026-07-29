from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.spatial import cKDTree
import trimesh

from autoanim_gnm.expression_showcase import (
    _bake_soft_contact_corrective,
    create_showcase_track,
    evaluate_showcase_frames,
)
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.mesh_contact import count_triangle_intersection_pairs
from autoanim_gnm.physics import (
    PhysicsError,
    PhysicsInputError,
    find_local_release_library,
)
from autoanim_gnm.semantic_decoder import ExpressionDecoder
from autoanim_gnm.soft_contact import (
    GNMSoftContactTopology,
    SoftContactSimulator,
    build_gnm_soft_contact_topology,
)


RELEASE_LIBRARY = find_local_release_library()
pytestmark = pytest.mark.skipif(
    RELEASE_LIBRARY is None,
    reason="AutoAnim physics release dylib is truly absent",
)


@dataclass(slots=True)
class SolvedContact:
    adapter: GNMAdapter
    full_targets: np.ndarray
    full_output: np.ndarray
    baked_output: np.ndarray
    local_targets: np.ndarray
    local_output: np.ndarray
    topology: GNMSoftContactTopology
    report: dict[str, object]


@pytest.fixture(scope="module")
def solved_contact() -> SolvedContact:
    assert RELEASE_LIBRARY is not None
    adapter = GNMAdapter()
    decoder = ExpressionDecoder(
        "gnm/shape/data/semantic_sampler/expression_decoder_model.h5"
    )
    track = create_showcase_track(adapter, decoder, fps=30)
    all_targets = evaluate_showcase_frames(adapter, track)
    start = round(11.5 * track.fps)
    stop = round(14.7 * track.fps) + 1
    targets = np.ascontiguousarray(all_targets[start:stop], dtype=np.float32)
    topology = build_gnm_soft_contact_topology(adapter, targets)
    local_targets = np.ascontiguousarray(
        targets[:, topology.global_vertex_indices],
        dtype=np.float32,
    )
    with SoftContactSimulator(
        topology,
        library_path=RELEASE_LIBRARY,
    ) as simulator:
        local_output = simulator.simulate(local_targets)
        report = simulator.report()
    output = targets.copy()
    output[:, topology.global_vertex_indices] = local_output
    baked_output, _ = _bake_soft_contact_corrective(
        targets,
        output,
        track.timestamps[start:stop],
    )
    return SolvedContact(
        adapter=adapter,
        full_targets=targets,
        full_output=output,
        baked_output=baked_output,
        local_targets=local_targets,
        local_output=local_output,
        topology=topology,
        report=report,
    )


def _closest_tongue_lip_distance(
    mesh: np.ndarray,
    adapter: GNMAdapter,
) -> float:
    tongue = np.flatnonzero(adapter.vertex_group("tongue") > 0.5)
    lip_triangles = np.concatenate(
        (
            np.asarray(adapter.model.triangles_group("lower_lip"), dtype=np.int32),
            np.asarray(adapter.model.triangles_group("upper_lip"), dtype=np.int32),
        ),
        axis=0,
    )
    triangles = mesh[lip_triangles]
    nearest_count = 12
    _, nearest = cKDTree(np.mean(triangles, axis=1)).query(
        mesh[tongue],
        k=nearest_count,
    )
    points = np.repeat(
        mesh[tongue, None, :],
        nearest_count,
        axis=1,
    ).reshape(-1, 3)
    closest = trimesh.triangles.closest_point(
        triangles[nearest.reshape(-1)],
        points,
    )
    return float(np.min(np.linalg.norm(points - closest, axis=1)))


def test_real_gnm_topology_is_volumetric_and_target_enters_contact_shell(
    solved_contact: SolvedContact,
) -> None:
    topology = solved_contact.topology
    assert topology.tongue_vertex_count == 933
    assert topology.lower_lip_vertex_count == 145
    assert topology.upper_lip_vertex_count == 145
    assert topology.rest_positions.shape == (1_223, 3)
    assert topology.tetrahedra.shape[0] >= 2_500
    assert topology.contact_pairs.shape[0] >= 4_000
    assert topology.boundary_vertex_count == 40
    assert topology.volume_coverage >= 0.94
    assert not topology.rest_positions.flags.writeable
    assert not topology.tetrahedra.flags.writeable

    # 12.4 s in the original timeline is 0.9 s into this sliced fixture.
    settling_frame = round((12.4 - 11.5) * 30)
    target_distance = _closest_tongue_lip_distance(
        solved_contact.full_targets[settling_frame],
        solved_contact.adapter,
    )
    assert 0.0002 <= target_distance < 0.001


def test_real_gnm_soft_contact_is_two_way_volume_safe_and_intersection_free(
    solved_contact: SolvedContact,
) -> None:
    adapter = solved_contact.adapter
    tongue_triangles = np.asarray(
        adapter.model.triangles_group("tongue"), dtype=np.int32
    )
    lip_triangles = (
        np.asarray(adapter.model.triangles_group("lower_lip"), dtype=np.int32),
        np.asarray(adapter.model.triangles_group("upper_lip"), dtype=np.int32),
    )
    for target, output, baked in zip(
        solved_contact.full_targets,
        solved_contact.full_output,
        solved_contact.baked_output,
        strict=True,
    ):
        for lips in lip_triangles:
            assert (
                count_triangle_intersection_pairs(
                    target,
                    tongue_triangles,
                    lips,
                )
                == 0
            )
            assert (
                count_triangle_intersection_pairs(
                    output,
                    tongue_triangles,
                    lips,
                )
                == 0
            )
            assert (
                count_triangle_intersection_pairs(
                    baked,
                    tongue_triangles,
                    lips,
                )
                == 0
            )

    report = solved_contact.report
    assert report["backend"] == "rust-xpbd-volumetric-soft-contact"
    assert report["continuous_collision_detection"] is False
    assert report["finite"] is True
    assert report["inverted_tetrahedron_samples"] == 0
    assert report["minimum_tetrahedron_volume_ratio"] >= 0.25
    assert report["maximum_tetrahedron_volume_ratio"] <= 2.5
    assert report["contact_projection_count"] > 0
    assert 0.0005 <= report["minimum_contact_separation_m"] <= 0.00125

    displacement = np.linalg.norm(
        solved_contact.local_output - solved_contact.local_targets,
        axis=2,
    )
    tongue_end = solved_contact.topology.tongue_vertex_count
    lower_lip_end = tongue_end + solved_contact.topology.lower_lip_vertex_count
    assert float(np.max(displacement[:, :tongue_end])) >= 0.0001
    assert float(np.max(displacement[:, tongue_end:lower_lip_end])) >= 0.0001
    assert float(np.max(displacement[:, lower_lip_end:])) >= 0.0001

    baked_local = solved_contact.baked_output[
        :, solved_contact.topology.global_vertex_indices
    ]
    tetrahedra = solved_contact.topology.tetrahedra
    target_tetrahedra = solved_contact.local_targets[:, tetrahedra]
    baked_tetrahedra = baked_local[:, tetrahedra]
    target_six_volume = np.einsum(
        "ftj,ftj->ft",
        target_tetrahedra[:, :, 1] - target_tetrahedra[:, :, 0],
        np.cross(
            target_tetrahedra[:, :, 2] - target_tetrahedra[:, :, 0],
            target_tetrahedra[:, :, 3] - target_tetrahedra[:, :, 0],
        ),
    )
    baked_six_volume = np.einsum(
        "ftj,ftj->ft",
        baked_tetrahedra[:, :, 1] - baked_tetrahedra[:, :, 0],
        np.cross(
            baked_tetrahedra[:, :, 2] - baked_tetrahedra[:, :, 0],
            baked_tetrahedra[:, :, 3] - baked_tetrahedra[:, :, 0],
        ),
    )
    assert float(np.min(baked_six_volume / target_six_volume)) > 0.0


def test_soft_contact_chunks_are_bit_identical_and_boundary_is_strict(
    solved_contact: SolvedContact,
) -> None:
    assert RELEASE_LIBRARY is not None
    midpoint = solved_contact.local_targets.shape[0] // 2
    with SoftContactSimulator(
        solved_contact.topology,
        library_path=RELEASE_LIBRARY,
    ) as simulator:
        chunked = np.concatenate(
            (
                simulator.simulate(solved_contact.local_targets[:midpoint]),
                simulator.simulate(solved_contact.local_targets[midpoint:]),
            ),
            axis=0,
        )
        report = simulator.report()
    np.testing.assert_array_equal(chunked, solved_contact.local_output)
    assert report["output_sha256"] == solved_contact.report["output_sha256"]

    simulator = SoftContactSimulator(
        solved_contact.topology,
        library_path=RELEASE_LIBRARY,
    )
    with pytest.raises(PhysicsInputError, match="float32"):
        simulator.simulate(solved_contact.local_targets.astype(np.float64))
    with pytest.raises(PhysicsInputError, match="C-contiguous"):
        simulator.simulate(solved_contact.local_targets[:, :, ::-1])
    simulator.close()
    simulator.close()
    with pytest.raises(PhysicsError, match="closed"):
        simulator.report()
