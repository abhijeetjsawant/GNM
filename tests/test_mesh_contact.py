from __future__ import annotations

import numpy as np
import pytest

from autoanim_gnm.mesh_contact import count_triangle_intersection_pairs


def test_triangle_contact_counts_crossings_but_not_separated_surfaces() -> None:
    vertices = np.asarray(
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -0.5, -1.0),
            (0.0, -0.5, 1.0),
            (0.0, 0.5, 0.0),
            (-1.0, -1.0, 2.0),
            (1.0, -1.0, 2.0),
            (0.0, 1.0, 2.0),
        ),
        dtype=np.float32,
    )
    first = np.asarray(((0, 1, 2),), dtype=np.int32)

    assert (
        count_triangle_intersection_pairs(
            vertices,
            first,
            np.asarray(((3, 4, 5),), dtype=np.int32),
        )
        == 1
    )
    assert (
        count_triangle_intersection_pairs(
            vertices,
            first,
            np.asarray(((6, 7, 8),), dtype=np.int32),
        )
        == 0
    )


def test_triangle_contact_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="vertices"):
        count_triangle_intersection_pairs(
            np.asarray(((np.nan, 0.0, 0.0),), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
        )
