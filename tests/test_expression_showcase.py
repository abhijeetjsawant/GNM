from __future__ import annotations

import numpy as np

from autoanim_gnm.expression_showcase import (
    create_showcase_track,
    evaluate_showcase_frames,
)
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.mesh_contact import count_triangle_intersection_pairs
from autoanim_gnm.mouth_geometry import discover_mouth_boundary, measure_mouth_boundary
from autoanim_gnm.oral_validation import validate_oral_frames
from autoanim_gnm.semantic_decoder import ExpressionDecoder


def test_semantic_decoder_samples_the_checked_in_cvae_deterministically() -> None:
    decoder = ExpressionDecoder(
        "gnm/shape/data/semantic_sampler/expression_decoder_model.h5"
    )

    first = decoder.sample("surprise", rng=np.random.default_rng(7))
    second = decoder.sample("surprise", rng=np.random.default_rng(7))

    np.testing.assert_array_equal(first, second)
    assert first.shape == (383,)
    assert np.isfinite(first).all()
    assert not np.array_equal(first, decoder.prototype("surprise"))


def test_showcase_uses_google_native_mouth_and_tongue_controls() -> None:
    adapter = GNMAdapter()
    decoder = ExpressionDecoder(
        "gnm/shape/data/semantic_sampler/expression_decoder_model.h5"
    )
    track = create_showcase_track(adapter, decoder, fps=30)

    assert track.expression.shape == (451, 383)
    assert track.rotations.shape == (451, 4, 3)
    assert np.array_equal(track.expression[0], np.zeros(383, dtype=np.float32))
    assert np.array_equal(track.expression[-1], np.zeros(383, dtype=np.float32))
    assert track.mouth_opening[0] == 0.0
    assert track.mouth_opening[-1] == 0.0
    assert np.max(track.mouth_opening) == 1.0
    assert np.isfinite(track.expression).all()
    assert np.max(np.abs(track.expression)) <= 3.0

    keyed = {keyframe.label: keyframe for keyframe in track.keyframes}
    wide = keyed["Google native wide mouth"]
    tongue = keyed["Google native tongue forward"]
    extended_tongue = keyed["Tongue settles on lower lip"]
    np.testing.assert_allclose(
        wide.expression[200:203],
        np.asarray((-2.6, 1.0, -2.1), dtype=np.float32),
        atol=0.0,
    )
    np.testing.assert_allclose(
        tongue.expression[200:203],
        wide.expression[200:203],
        atol=0.0,
    )
    np.testing.assert_allclose(
        tongue.expression[350:354],
        np.asarray((0.7, -1.7, 0.0, 0.0), dtype=np.float32),
        atol=0.0,
    )
    np.testing.assert_allclose(
        extended_tongue.expression[350:354],
        np.asarray((0.739, 2.18, 0.0, 0.0), dtype=np.float32),
        atol=0.0,
    )
    assert extended_tongue.expression[203] == np.float32(-0.82)
    np.testing.assert_array_equal(
        keyed["Tongue supported by soft tissue"].expression,
        extended_tongue.expression,
    )

    full_frames = evaluate_showcase_frames(adapter, track)
    native_frames = adapter.mesh(
        expression=track.expression,
        rotations=track.rotations,
        translation=track.translation,
    )
    np.testing.assert_allclose(full_frames, native_frames, atol=1.0e-7)

    key_indices = np.asarray(
        [round(keyframe.time_seconds * track.fps) for keyframe in track.keyframes],
        dtype=np.int64,
    )
    key_frames = full_frames[key_indices]
    topology = discover_mouth_boundary(adapter)
    wide_index = track.keyframes.index(wide)
    wide_frame = key_frames[wide_index]
    wide_landmarks = np.sum(
        wide_frame[adapter.landmark_indices]
        * adapter.landmark_weights[..., None],
        axis=-2,
    )
    wide_measurement = measure_mouth_boundary(
        wide_frame,
        wide_landmarks,
        topology,
    )
    assert wide_measurement.opening_area_m2 >= 0.00045

    tongue = np.flatnonzero(adapter.vertex_group("tongue") > 0.5)
    lower_lip = np.flatnonzero(adapter.vertex_group("lower_lip") > 0.5)
    chin = np.flatnonzero(adapter.vertex_group("chin_region") > 0.5)
    neutral = adapter.mesh()
    assert np.max(np.linalg.norm(key_frames[:, tongue] - neutral[tongue], axis=2)) >= 0.02
    extended_index = track.keyframes.index(extended_tongue)
    extended_frame = key_frames[extended_index]
    assert (
        np.min(extended_frame[tongue, 1])
        - np.min(extended_frame[lower_lip, 1])
        >= 0.0035
    )
    assert (
        np.min(extended_frame[tongue, 1])
        - np.max(extended_frame[chin, 1])
        >= 0.00325
    )
    assert (
        np.max(extended_frame[tongue, 2])
        - np.max(extended_frame[lower_lip, 2])
        >= 0.008
    )
    tongue_triangles = np.asarray(
        adapter.model.triangles_group("tongue"),
        dtype=np.int32,
    )
    upper_lip_triangles = np.asarray(
        adapter.model.triangles_group("upper_lip"),
        dtype=np.int32,
    )
    lower_lip_triangles = np.asarray(
        adapter.model.triangles_group("lower_lip"),
        dtype=np.int32,
    )
    for label in (
        "Tongue settles on lower lip",
        "Tongue supported by soft tissue",
        "Tongue pressure releases",
    ):
        frame = key_frames[track.keyframes.index(keyed[label])]
        assert (
            count_triangle_intersection_pairs(
                frame,
                tongue_triangles,
                upper_lip_triangles,
            )
            == 0
        )
        assert (
            count_triangle_intersection_pairs(
                frame,
                tongue_triangles,
                lower_lip_triangles,
            )
            == 0
        )
    first_tongue_frame = round(
        keyed["Google native tongue forward"].time_seconds * track.fps
    )
    recovery_frame = round(
        keyed["Open-mouth recovery"].time_seconds * track.fps
    )
    for frame in full_frames[first_tongue_frame : recovery_frame + 1]:
        assert (
            count_triangle_intersection_pairs(
                frame,
                tongue_triangles,
                upper_lip_triangles,
            )
            == 0
        )
        assert (
            count_triangle_intersection_pairs(
                frame,
                tongue_triangles,
                lower_lip_triangles,
            )
            == 0
        )

    oral = validate_oral_frames(
        full_frames,
        adapter=adapter,
        timestamps=track.timestamps,
        source_kind="test_expression_showcase",
    )
    assert not np.any(oral.tongue_teeth_collision_risk_frames)
