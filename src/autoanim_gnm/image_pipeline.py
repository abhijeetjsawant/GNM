"""End-to-end single-photo visible-geometry fitting service."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .errors import AutoAnimError
from .fitting import IdentityFitter
from .gnm_adapter import GNMAdapter
from .gltf_export import export_gnm_glb
from .image import FaceExtractor, MAPPING_NAME
from .rig import ControlRig
from .render import MeshRenderer
from .semantic_decoder import ExpressionDecoder
from .serialization import write_json, write_npz


IMAGE_CAVEAT = "Single-view visible-geometry estimate; not a metric 3D clone."
OCCLUSION_CAVEAT = "Occlusion is inferred only from image bounds and fit residuals."


def draw_overlay(image: np.ndarray, observed: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    output = image.copy()
    for actual, predicted in zip(observed, fitted, strict=True):
        a = tuple(np.rint(actual).astype(int))
        p = tuple(np.rint(predicted).astype(int))
        cv2.line(output, a, p, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.circle(output, a, 3, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.circle(output, p, 3, (220, 0, 220), -1, cv2.LINE_AA)
    return output


def run_image_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path,
    modes: int = 20,
    allow_low_confidence: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detection = FaceExtractor(model_path).detect(input_path)
    adapter = GNMAdapter()
    decoder = ExpressionDecoder("gnm/shape/data/semantic_sampler/expression_decoder_model.h5")
    rig = ControlRig(adapter, decoder)
    fitter = IdentityFitter(adapter, rig)
    fit = fitter.fit(
        detection.landmarks,
        detection.image_bgr.shape[:2],
        modes=modes,
        face_width=detection.face_width,
    )
    warnings = [IMAGE_CAVEAT, OCCLUSION_CAVEAT]
    if detection.mapped_in_bounds_fraction < .95:
        warnings.append("POSSIBLE_OCCLUSION")
    if detection.strong_expression_score > .70:
        warnings.append("STRONG_EXPRESSION")
    if fit.confidence == "rejected" or (fit.confidence == "low" and not allow_low_confidence):
        raise AutoAnimError(
            "FIT_REJECTED",
            f"Image fit confidence is {fit.confidence}",
            {
                "nme": fit.nme,
                "confidence": fit.confidence,
                "reasons": fit.confidence_reasons,
            },
        )
    if fit.confidence == "low":
        warnings.append("LOW_CONFIDENCE")
    expression = np.zeros(adapter.expression_dim, dtype=np.float32)
    mesh = adapter.mesh(identity=fit.identity, expression=expression)
    adapter.export_obj(output_dir / "fitted.obj", mesh)
    glb = export_gnm_glb(
        output_dir / "fitted.glb",
        adapter,
        mesh,
        mapping_path=output_dir / "fitted-glb-mapping.npz",
    )
    MeshRenderer(adapter).save_png(
        output_dir / "mesh-preview.png",
        mesh,
        adapter.landmarks(identity=fit.identity, expression=expression),
    )
    overlay = draw_overlay(detection.image_bgr, detection.landmarks, fit.fitted_landmarks)
    if not cv2.imwrite(str(output_dir / "overlay.png"), overlay):
        raise AutoAnimError("INTERNAL_ERROR", "Could not write fit overlay")
    write_npz(
        output_dir / "fit.npz",
        identity=fit.identity,
        expression=expression,
        camera=fit.camera,
        observed_landmarks=detection.landmarks.astype(np.float32),
        fitted_landmarks=fit.fitted_landmarks,
    )
    result = {
        "kind": "image_fit",
        "status": "succeeded",
        "model": {"gnm_version": "3.0", "identity_dim": adapter.identity_dim},
        "detection": {
            "faces": 1,
            "landmarks": len(detection.all_landmarks),
            "mapping": MAPPING_NAME,
            "face_width_px": detection.face_width,
            "mapped_in_bounds_fraction": detection.mapped_in_bounds_fraction,
            "strong_expression_score": detection.strong_expression_score,
            "yaw_deg": float(np.degrees(fit.camera[0])),
            "pitch_deg": float(np.degrees(fit.camera[1])),
            "roll_deg": float(np.degrees(fit.camera[2])),
        },
        "fit": {
            "modes": modes,
            "nme": fit.nme,
            "pixel_error": fit.pixel_error,
            "coefficient_bound_fraction": fit.saturation_fraction,
            "stability_rms": fit.stability_rms,
            "confidence": fit.confidence,
            "confidence_reasons": list(fit.confidence_reasons),
            "glb_vertices": glb.vertex_count,
            "glb_seam_duplicates": glb.seam_duplicates,
        },
        "artifacts": {
            "mesh": "fitted.obj",
            "glb": "fitted.glb",
            "glb_mapping": "fitted-glb-mapping.npz",
            "mesh_preview": "mesh-preview.png",
            "overlay": "overlay.png",
            "parameters": "fit.npz",
        },
        "warnings": warnings,
    }
    write_json(output_dir / "result.json", result)
    return result
