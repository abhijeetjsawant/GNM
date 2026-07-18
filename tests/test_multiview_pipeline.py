from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoanim_gnm.artifacts import JobStore
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.multiview import (
    CameraIntrinsics,
    PerspectiveCamera,
    project_points,
)
from autoanim_gnm.multiview_pipeline import (
    REAR_VIEW_CAVEAT,
    _component_texture_metrics,
    _normalise_roles,
    _validate_neutral_texture_captures,
    run_multiview_pipeline,
    texture_camera_from_fit,
)


CACHE = Path(os.environ.get("AUTOANIM_CACHE_DIR", ".cache/autoanim_gnm"))
FIXTURES = Path(os.environ.get("AUTOANIM_TEST_FIXTURES", CACHE / "fixtures"))
MODEL = CACHE / "face_landmarker.task"
ASTRONAUT = FIXTURES / "astronaut.png"


def test_fitted_camera_conversion_preserves_pixel_projection() -> None:
    intrinsics = CameraIntrinsics(950.0, 930.0, 321.0, 299.0)
    camera = PerspectiveCamera(0.53, -0.12, 0.07, 0.015, -0.02, 0.82, intrinsics)
    points = np.asarray(
        ((-0.04, 0.03, 0.01), (0.06, -0.02, 0.03), (0.0, 0.08, -0.01)),
        dtype=np.float64,
    )
    expected = project_points(points, camera)
    converted = texture_camera_from_fit(camera)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera_points = (converted.world_to_camera @ homogeneous.T).T[:, :3]
    normalized = camera_points[:, :2] / camera_points[:, 2:3]
    pixels = (
        converted.intrinsics
        @ np.column_stack((normalized, np.ones(len(points)))).T
    ).T
    actual = pixels[:, :2] / pixels[:, 2:3]
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert np.linalg.det(converted.world_to_camera[:3, :3]) == pytest.approx(1.0)
    camera_center = np.linalg.inv(converted.world_to_camera)[:3, 3]
    assert camera_center[2] > float(np.max(points[:, 2]))
    camera_forward_world = converted.world_to_camera[:3, :3].T @ np.asarray(
        (0.0, 0.0, 1.0)
    )
    assert camera_forward_world[2] < -0.80


def test_capture_roles_are_ordered_and_rear_is_rejected() -> None:
    assert _normalise_roles(None, 3) == ("front", "left_3q", "right_3q")
    assert _normalise_roles(("Front", "left 3q"), 2) == ("front", "left_3q")
    with pytest.raises(AutoAnimError, match="exactly one"):
        _normalise_roles(("front",), 2)
    with pytest.raises(AutoAnimError, match="Rear-head") as caught:
        _normalise_roles(("front", "back"), 2)
    assert caught.value.message == REAR_VIEW_CAVEAT


def test_gnm_atlas_mirror_fill_is_explicitly_disabled(tmp_path: Path) -> None:
    with pytest.raises(AutoAnimError, match="Atlas mirror fill is disabled") as caught:
        run_multiview_pipeline(
            (tmp_path / "front.jpg", tmp_path / "profile.jpg"),
            tmp_path / "out",
            model_path=tmp_path / "model.task",
            mirror_fill=True,
        )
    assert caught.value.code == "INPUT_INVALID"


def test_component_texture_metrics_expose_hidden_anatomy() -> None:
    baked = SimpleNamespace(
        triangle_index=np.asarray(((0, 1, 2), (-1, 3, 4)), dtype=np.int32),
        atlas_mask=np.asarray(((True, True, True), (False, True, True))),
        observed=np.asarray(((True, False, True), (False, False, False))),
        mirrored=np.zeros((2, 3), dtype=bool),
        inpainted=np.asarray(((False, False, False), (False, True, False))),
        generic=np.asarray(((False, True, False), (True, False, True))),
    )
    atlas = SimpleNamespace(
        triangle_components=np.asarray((0, 0, 1, 1, 1), dtype=np.int16),
        component_names=("skin", "eye"),
    )

    component_map, metrics = _component_texture_metrics(baked, atlas)

    np.testing.assert_array_equal(
        component_map, np.asarray(((0, 0, 1), (-1, 1, 1)), dtype=np.int16)
    )
    assert metrics["skin"] == {
        "atlas_texels": 2,
        "observed_texels": 1,
        "observed_fraction": 0.5,
        "mirrored_texels": 0,
        "mirrored_fraction": 0.0,
        "inpainted_texels": 0,
        "inpainted_fraction": 0.0,
        "generic_texels": 1,
        "generic_fraction": 0.5,
    }
    assert metrics["eye"]["observed_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["eye"]["inpainted_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["eye"]["generic_fraction"] == pytest.approx(1.0 / 3.0)


def test_expression_capture_is_rejected_before_neutral_texture_bake() -> None:
    detections = (
        SimpleNamespace(strong_expression_score=0.10),
        SimpleNamespace(strong_expression_score=0.80),
    )
    fitted = SimpleNamespace(
        nuisance=(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32))
    )
    with pytest.raises(AutoAnimError, match="Texture captures must be neutral") as caught:
        _validate_neutral_texture_captures(
            detections, fitted, (0, 1), ("front", "left_3q")
        )
    assert caught.value.code == "FIT_REJECTED"
    assert caught.value.details["views"][0]["view_index"] == 1

    detections = tuple(
        SimpleNamespace(strong_expression_score=0.10) for _ in range(2)
    )
    fitted = SimpleNamespace(
        nuisance=(
            np.zeros(4, dtype=np.float32),
            np.asarray((0.0, 0.0, 0.30, 0.0), dtype=np.float32),
        )
    )
    with pytest.raises(AutoAnimError) as nuisance_caught:
        _validate_neutral_texture_captures(
            detections, fitted, (0, 1), ("front", "left_3q")
        )
    assert nuisance_caught.value.details["views"][0]["fitted_nuisance_peak"] == pytest.approx(
        0.30
    )


def test_multi_input_job_manifest_retains_order_hashes_and_names(tmp_path: Path) -> None:
    first = tmp_path / "one.jpg"
    second = tmp_path / "two.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    store = JobStore(tmp_path / "jobs")
    _, job_dir, retained, manifest = store.start_many(
        "multiview_reconstruction",
        (first, second),
        {"roles": ["front", "left_3q"]},
        original_names=("front portrait.jpg", "left profile.png"),
    )
    assert [path.name for path in retained] == ["input-01.jpg", "input-02.png"]
    assert [value["name"] for value in manifest["inputs"]] == [
        "front_portrait.jpg",
        "left_profile.png",
    ]
    assert manifest["input"]["media_type"] == "multipart/mixed"
    final = store.finish(manifest, job_dir, {"status": "succeeded", "artifacts": {}}, {})
    assert final["inputs"] == manifest["inputs"]
    persisted = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    assert persisted["inputs"] == manifest["inputs"]


def test_camera_sidecar_is_hash_audited_and_preserved_through_finish(
    tmp_path: Path,
) -> None:
    images = []
    for index in range(3):
        image = tmp_path / f"view-{index}.png"
        image.write_bytes(f"image-{index}".encode())
        images.append(image)
    sidecar = tmp_path / "rig.json"
    sidecar.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    store = JobStore(tmp_path / "jobs")

    _, job_dir, _, manifest = store.start_many(
        "multiview_reconstruction",
        tuple(images),
        {},
        attachments={"camera_calibration": sidecar},
    )

    assert len(manifest["attachments"]) == 1
    retained = job_dir / manifest["attachments"][0]["retained_name"]
    assert retained.read_bytes() == sidecar.read_bytes()
    assert manifest["attachments"][0]["sha256"] == hashlib.sha256(
        sidecar.read_bytes()
    ).hexdigest()
    final = store.finish(manifest, job_dir, {"status": "succeeded", "artifacts": {}}, {})
    assert final["attachments"] == manifest["attachments"]


@pytest.mark.skipif(not MODEL.exists() or not ASTRONAUT.exists(), reason="real photo unavailable")
def test_duplicate_real_views_fail_diversity_instead_of_faking_multiview(
    tmp_path: Path,
) -> None:
    with pytest.raises(AutoAnimError, match="viewpoint diversity") as caught:
        run_multiview_pipeline(
            (ASTRONAUT, ASTRONAUT),
            tmp_path,
            model_path=MODEL,
            roles=("front", "front"),
            texture_size=128,
        )
    assert caught.value.code == "FIT_REJECTED"
    assert caught.value.details["yaw_span_degrees"] < 20.0
