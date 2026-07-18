from pathlib import Path
import os

import cv2
import numpy as np
import pytest

from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.fitting import IdentityFitter, project_landmarks
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.image import FaceExtractor, MAPPING_NAME
from autoanim_gnm.image_pipeline import IMAGE_CAVEAT, run_image_pipeline
from autoanim_gnm.rig import ControlRig


CACHE = Path(os.environ.get("AUTOANIM_CACHE_DIR", ".cache/autoanim_gnm"))
FIXTURES = Path(os.environ.get("AUTOANIM_TEST_FIXTURES", CACHE / "fixtures"))
MODEL = CACHE / "face_landmarker.task"
ASTRONAUT = FIXTURES / "astronaut.png"
PORTRAIT = FIXTURES / "official-portrait.jpg"


@pytest.fixture(scope="session")
def fitter(adapter: GNMAdapter, rig: ControlRig) -> IdentityFitter:
    return IdentityFitter(adapter, rig)


def test_unobservable_blocks_are_exactly_absent(adapter: GNMAdapter) -> None:
    assert not np.any(adapter.compact_identity_basis[170:])
    assert not np.any(adapter.compact_expression_basis[350:])


def test_synthetic_recovery_gate(adapter: GNMAdapter, fitter: IdentityFitter) -> None:
    skin = adapter.vertex_group("skin_exterior") > 0.5
    nmes: list[float] = []
    cosines: list[float] = []
    means: list[float] = []
    p95s: list[float] = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        truth = rng.normal(0, 0.5, 20)
        camera = np.asarray(
            (
                np.deg2rad((-18, 0, 18)[seed % 3]),
                -0.04,
                0.02,
                np.log(1800),
                320,
                320,
            )
        )
        target = fitter._landmarks(truth, np.zeros(4), 20)
        observed = project_landmarks(target, camera) + rng.normal(0, 0.5, (68, 2))
        fit = fitter.fit(observed, (640, 640), modes=20, compute_stability=False)
        recovered = fit.identity[:20]
        nmes.append(fit.nme)
        cosines.append(float(np.dot(truth, recovered) / (np.linalg.norm(truth) * np.linalg.norm(recovered))))
        truth_full = np.zeros(253, dtype=np.float32)
        truth_full[:20] = truth
        error_mm = np.linalg.norm(
            adapter.mesh(identity=truth_full)[skin] - adapter.mesh(identity=fit.identity)[skin], axis=1
        ) * 1000
        means.append(float(np.mean(error_mm)))
        p95s.append(float(np.percentile(error_mm, 95)))
    assert np.median(nmes) <= 0.015
    assert np.median(cosines) >= 0.75
    assert np.median(means) <= 1.5
    assert np.median(p95s) <= 3.0


@pytest.mark.skipif(not MODEL.exists() or not ASTRONAUT.exists(), reason="real image fixture unavailable")
def test_real_astronaut_correspondence_orientation() -> None:
    detected = FaceExtractor(MODEL).detect(ASTRONAUT)
    points = detected.landmarks
    assert points.shape == (68, 2)
    assert points[0, 0] < points[16, 0]
    assert points[36, 0] < points[45, 0]
    assert points[8, 1] > points[0, 1]
    assert detected.face_width > 80


@pytest.mark.parametrize("image_path", [ASTRONAUT, PORTRAIT])
def test_real_photo_end_to_end(image_path: Path, tmp_path: Path) -> None:
    if not MODEL.exists() or not image_path.exists():
        pytest.skip("real image fixture unavailable")
    result = run_image_pipeline(image_path, tmp_path, model_path=MODEL)
    assert result["status"] == "succeeded"
    assert result["detection"]["mapping"] == MAPPING_NAME
    assert result["fit"]["confidence"] in {"high", "medium"}
    assert result["fit"]["nme"] <= 0.060
    assert result["fit"]["stability_rms"] <= 0.75
    assert IMAGE_CAVEAT in result["warnings"]
    assert (tmp_path / "fitted.obj").stat().st_size > 1_000_000
    assert cv2.imread(str(tmp_path / "overlay.png")) is not None
    assert cv2.imread(str(tmp_path / "mesh-preview.png")).shape == (640, 640, 3)
    with np.load(tmp_path / "fit.npz", allow_pickle=False) as values:
        identity = values["identity"]
        assert identity.shape == (253,)
        assert np.linalg.norm(identity[:20]) > 0.1
        assert not np.any(identity[20:])
        assert not np.any(values["expression"])


def test_small_synthetic_fit_is_low_or_rejected(fitter: IdentityFitter) -> None:
    camera = np.asarray((0, 0, 0, np.log(350), 128, 128))
    observed = project_landmarks(fitter.template, camera)
    fit = fitter.fit(
        observed,
        (256, 256),
        face_width=70,
        modes=20,
        compute_stability=False,
    )
    assert fit.confidence == "low"
    assert "SMALL_FACE" in fit.confidence_reasons


@pytest.mark.skipif(not MODEL.exists() or not ASTRONAUT.exists() or not PORTRAIT.exists(), reason="negative fixtures unavailable")
def test_actual_zero_multiple_and_extreme_inputs_are_typed(tmp_path: Path) -> None:
    extractor = FaceExtractor(MODEL)
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.zeros((512, 512, 3), dtype=np.uint8))
    with pytest.raises(AutoAnimError) as caught:
        extractor.detect(blank)
    assert caught.value.code == "FACE_NOT_FOUND"

    portrait = cv2.resize(cv2.imread(str(PORTRAIT)), None, fx=0.4, fy=0.4)
    two_faces = tmp_path / "two-faces.jpg"
    cv2.imwrite(str(two_faces), np.hstack((portrait, portrait)))
    with pytest.raises(AutoAnimError) as caught:
        extractor.detect(two_faces)
    assert caught.value.code == "MULTIPLE_FACES"

    astronaut = cv2.imread(str(ASTRONAUT))
    matrix = cv2.getRotationMatrix2D((256, 256), 60, 1)
    rotated = tmp_path / "rotated.png"
    cv2.imwrite(str(rotated), cv2.warpAffine(astronaut, matrix, (512, 512)))
    with pytest.raises(AutoAnimError) as caught:
        run_image_pipeline(rotated, tmp_path / "rotated-result", model_path=MODEL, allow_low_confidence=True)
    assert caught.value.code == "FIT_REJECTED"

    tiny = tmp_path / "tiny.png"
    cv2.imwrite(str(tiny), cv2.resize(astronaut, (256, 256)))
    with pytest.raises(AutoAnimError) as caught:
        run_image_pipeline(tiny, tmp_path / "tiny-result", model_path=MODEL, allow_low_confidence=True)
    assert caught.value.code in {"FACE_NOT_FOUND", "FIT_REJECTED"}

    cropped = tmp_path / "cropped.png"
    cv2.imwrite(str(cropped), astronaut[:, 210:])
    with pytest.raises(AutoAnimError) as caught:
        run_image_pipeline(cropped, tmp_path / "cropped-result", model_path=MODEL, allow_low_confidence=True)
    assert caught.value.code in {"FACE_NOT_FOUND", "FIT_REJECTED"}
