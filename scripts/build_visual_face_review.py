#!/usr/bin/env python3
"""Build a source-landmark-corrected, face-only GNM review payload."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
from gnm.shape.visualization import vertex_colors as vertex_colors_module

from autoanim_gnm.gltf_export import export_gnm_glb
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.image import MEDIAPIPE_TO_GNM68
from autoanim_gnm.serialization import write_json, write_npz
from autoanim_gnm.visual_face_retarget import solve_visual_face_track


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = _arguments()
    base_job = args.base_job.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    performance_path = base_job / "performance.npz"
    capture_path = base_job / "capture.npz"
    if not performance_path.is_file() or not capture_path.is_file():
        raise FileNotFoundError("base job must contain performance.npz and capture.npz")

    with np.load(performance_path, allow_pickle=False) as values:
        performance = {name: np.asarray(values[name]) for name in values.files}
    with np.load(capture_path, allow_pickle=False) as values:
        landmarks = np.asarray(values["landmarks_xyz"], dtype=np.float32)[
            :, MEDIAPIPE_TO_GNM68, :2
        ]
        frame_size = np.asarray(values["frame_size"], dtype=np.int32)
        detected = np.asarray(values["detected"], dtype=bool)
    observed_pixels = landmarks * np.asarray(
        (frame_size[0], frame_size[1]), dtype=np.float32
    )

    adapter = GNMAdapter()
    solved = solve_visual_face_track(
        adapter,
        identity=np.asarray(performance["identity"], dtype=np.float32),
        base_expression=np.asarray(performance["expression"], dtype=np.float32),
        observed_landmarks=observed_pixels,
        image_size=(int(frame_size[1]), int(frame_size[0])),
        detected=detected,
    )
    performance["expression"] = solved.expression
    write_npz(output / "performance.npz", **performance)
    write_npz(
        output / "visual-face-retarget.npz",
        cameras=solved.cameras,
        base_nme=solved.base_nme,
        corrected_nme=solved.corrected_nme,
        base_region_nme=solved.base_region_nme,
        corrected_region_nme=solved.corrected_region_nme,
        bound_fraction=solved.bound_fraction,
        observed_landmarks=observed_pixels,
    )
    report = {
        **solved.report,
        "bindings": {
            "base_performance_sha256": _file_sha256(performance_path),
            "capture_sha256": _file_sha256(capture_path),
        },
    }
    write_json(output / "visual-face-retarget.json", report)

    identity = np.asarray(performance["identity"], dtype=np.float32)
    neutral_mesh = adapter.mesh(identity=identity)
    export_gnm_glb(
        output / "neutral.glb",
        adapter,
        neutral_mesh,
        mapping_path=output / "neutral-glb-mapping.npz",
    )
    with np.load(output / "neutral-glb-mapping.npz", allow_pickle=False) as mapping:
        glb_vertex_to_gnm = np.asarray(mapping["glb_vertex_to_gnm_vertex"], dtype=np.int32)
    meshes = adapter.mesh(
        identity=np.broadcast_to(identity, (len(solved.expression), len(identity))),
        expression=solved.expression,
    )
    render_vertices = np.asarray(meshes[:, glb_vertex_to_gnm], dtype=np.float32)
    render_colors = np.asarray(
        vertex_colors_module.get_vertex_colors(adapter.model), dtype=np.float32
    )[glb_vertex_to_gnm]
    write_npz(
        output / "render-meshes.npz",
        vertices=render_vertices,
        colors=render_colors,
        timestamps_seconds=np.asarray(performance["timestamps_seconds"], dtype=np.float64),
        source_pts=np.asarray(performance["source_pts"], dtype=np.int64),
    )
    print(output / "visual-face-retarget.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
