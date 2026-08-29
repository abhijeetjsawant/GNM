#!/usr/bin/env python3
"""Project one photographed face onto a fitted GNM identity for review."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from autoanim_gnm.gltf_export import export_gnm_glb
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.gnm_texture import build_gnm_texture_atlas
from autoanim_gnm.image import FaceExtractor
from autoanim_gnm.multiview import CameraIntrinsics, MultiViewIdentityFitter, MultiViewObservation, project_points
from autoanim_gnm.multiview_pipeline import _face_mask, texture_camera_from_fit
from autoanim_gnm.rig import ControlRig
from autoanim_gnm.semantic_decoder import ExpressionDecoder
from autoanim_gnm.serialization import write_json, write_npz
from autoanim_gnm.texture_baker import bake_multiview_texture


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.identity, allow_pickle=False) as values:
        identity = np.asarray(values["identity"], dtype=np.float32)
    adapter = GNMAdapter()
    if identity.shape != (adapter.identity_dim,):
        raise ValueError("identity artifact does not contain one GNM identity")
    detection = FaceExtractor(args.model).detect(args.image)
    height, width = detection.image_bgr.shape[:2]
    focal = 1.25 * max(height, width)
    intrinsics = CameraIntrinsics(focal, focal, 0.5 * (width - 1), 0.5 * (height - 1))
    observation = MultiViewObservation(
        detection.landmarks,
        (height, width),
        intrinsics=intrinsics,
        role="single_front_oblique",
        confidence=detection.mapped_in_bounds_fraction,
    )
    rig = ControlRig(
        adapter,
        ExpressionDecoder("gnm/shape/data/semantic_sampler/expression_decoder_model.h5"),
    )
    fitter = MultiViewIdentityFitter(adapter, rig)
    view = fitter._prepare_views((observation,))[0]
    compact = adapter.compact_template.astype(np.float64) + np.einsum(
        "i,ijk->jk", identity, adapter.compact_identity_basis, optimize=True
    )
    camera = fitter._initial_camera(view)
    camera = fitter._refine_camera(view, compact, camera, np.ones(68, dtype=np.float64))
    predicted = project_points(compact, camera)
    interocular = max(float(np.linalg.norm(detection.landmarks[36] - detection.landmarks[45])), 1.0)
    nme = float(np.mean(np.linalg.norm(predicted - detection.landmarks, axis=1)) / interocular)

    mesh = adapter.mesh(identity=identity)
    atlas = build_gnm_texture_atlas(adapter, args.size)
    image_rgb = cv2.cvtColor(detection.image_bgr, cv2.COLOR_BGR2RGB)
    baked = bake_multiview_texture(
        mesh,
        adapter.triangles,
        atlas.triangle_uvs,
        (image_rgb,),
        (texture_camera_from_fit(camera),),
        texture_size=args.size,
        masks=(_face_mask(detection.all_landmarks, (height, width)),),
        confidences=(detection.mapped_in_bounds_fraction,),
        generic_vertex_colors=atlas.generic_vertex_colors,
        mirror_fill=False,
        inpaint=True,
    )
    Image.fromarray(baked.rgba, mode="RGBA").save(output / "texture.png")
    write_npz(
        output / "texture-maps.npz",
        triangle_uvs=atlas.triangle_uvs,
        confidence=baked.confidence,
        source_view=baked.source_view,
        observed=baked.observed,
        inpainted=baked.inpainted,
        generic=baked.generic,
        atlas_mask=baked.atlas_mask,
    )
    export_gnm_glb(
        output / "neutral-textured.glb",
        adapter,
        mesh,
        texture_path=output / "texture.png",
        triangle_uvs=atlas.triangle_uvs,
        mapping_path=output / "neutral-textured-mapping.npz",
    )
    write_json(
        output / "result.json",
        {
            "schema_version": "autoanim.single-image-textured-identity/1.0",
            "status": "research_preview",
            "identity_modes": 170,
            "camera_reprojection_nme": nme,
            "texture_size": args.size,
            "texture_metrics": dict(baked.metrics),
            "production_validated": False,
            "limitations": [
                "Only camera-visible texels are photographic observations.",
                "Hidden and rear texels are inpainted or generic and are not Andrew Garfield measurements.",
                "The source frame is low-resolution, dramatically lit, and mildly expressive.",
                "Hair is not represented by the GNM head mesh.",
            ],
        },
    )
    print(output / "neutral-textured.glb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
