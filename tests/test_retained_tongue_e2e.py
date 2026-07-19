"""Retained-real-audio regression coverage for the GNM tongue handoff.

This test intentionally uses a retained learned Audio2Face job instead of a
mock.  A checkout without that private/local artifact skips honestly; a job
that exists but is partial, corrupt, or no longer reconstructs against the
installed GNM asset fails.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from autoanim_gnm.gnm_adapter import GNMAdapter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETAINED_JOB = ROOT / "artifacts/jobs/01kxvctvcyh5nrat758makhchv"
REQUIRED_ARTIFACTS = {
    "normalized_audio": "normalized.wav",
    "arkit_controls": "arkit_controls.npz",
    "controls": "controls.npz",
    "glb": "animation.glb",
    "glb_mapping": "animation-glb-mapping.npz",
    "retarget_calibration": "retarget_calibration.npz",
}


def _retained_job() -> Path:
    override = os.environ.get("AUTOANIM_RETAINED_TONGUE_JOB")
    if override:
        path = Path(override).expanduser().resolve()
        assert path.is_dir(), (
            "AUTOANIM_RETAINED_TONGUE_JOB does not name a directory: "
            f"{path}"
        )
        return path
    if DEFAULT_RETAINED_JOB.is_dir():
        return DEFAULT_RETAINED_JOB

    candidates: list[Path] = []
    for result_path in (ROOT / "artifacts/jobs").glob("*/result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            result.get("status") == "succeeded"
            and result.get("analysis", {}).get("motion_backend") == "learned_a2f"
        ):
            candidates.append(result_path.parent)
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)
    pytest.skip(
        "retained learned-audio tongue job unavailable; set "
        "AUTOANIM_RETAINED_TONGUE_JOB to run this real-input regression"
    )


def _load_and_verify_job(job: Path) -> dict[str, object]:
    result_path = job / "result.json"
    assert result_path.is_file(), f"retained job is partial: missing {result_path.name}"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result.get("kind") == "audio_animation"
    assert result.get("status") == "succeeded"
    assert result.get("analysis", {}).get("motion_backend") == "learned_a2f"
    assert "gnm-dense-calibrated-v3" in result.get("analysis", {}).get("backend", "")

    missing = [name for name in REQUIRED_ARTIFACTS.values() if not (job / name).is_file()]
    assert not missing, f"retained job is partial: missing {missing}"

    manifest = result.get("artifacts")
    assert isinstance(manifest, dict), "retained job has no artifact manifest"
    for key, expected_name in REQUIRED_ARTIFACTS.items():
        entry = manifest.get(key)
        assert isinstance(entry, dict), f"manifest entry {key!r} is not integrity-bearing"
        assert entry.get("name") == expected_name
        expected_sha = entry.get("sha256")
        assert isinstance(expected_sha, str) and len(expected_sha) == 64
        artifact_path = job / expected_name
        assert artifact_path.stat().st_size == entry.get("bytes")
        actual_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"retained artifact hash mismatch: {expected_name}"
    return result


def _glb_chunks(path: Path) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    assert len(payload) >= 20
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    assert magic == b"glTF" and version == 2 and declared_length == len(payload)
    offset = 12
    chunks: dict[int, bytes] = {}
    while offset < len(payload):
        length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunks[chunk_type] = payload[offset : offset + length]
        offset += length
    assert offset == len(payload)
    assert 0x4E4F534A in chunks and 0x004E4942 in chunks
    return json.loads(chunks[0x4E4F534A]), chunks[0x004E4942]


def _float_accessor(document: dict[str, object], binary: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    assert accessor["componentType"] == 5126
    assert "sparse" not in accessor
    component_count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[
        accessor["type"]
    ]
    assert view.get("byteStride", component_count * 4) == component_count * 4
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    values = np.frombuffer(
        binary,
        dtype="<f4",
        count=int(accessor["count"]) * component_count,
        offset=offset,
    )
    return values.reshape(int(accessor["count"]), component_count).copy()


def test_retained_real_audio_drives_and_exports_dense_gnm_tongue() -> None:
    job = _retained_job()
    _load_and_verify_job(job)

    with np.load(job / "arkit_controls.npz", allow_pickle=False) as source:
        names = np.asarray(source["tongue_pose_names"]).astype(str)
        named_weights = np.asarray(source["conditioned_tongue_weights"], dtype=np.float32)
        source_timestamps = np.asarray(source["timestamps"], dtype=np.float32)
    assert named_weights.shape == (len(source_timestamps), len(names))
    assert len(names) == 16 and len(set(names.tolist())) == len(names)
    assert np.isfinite(named_weights).all()
    assert float(named_weights.min()) >= 0.0 and float(named_weights.max()) <= 1.0
    varying_named_controls = int(np.count_nonzero(np.ptp(named_weights, axis=0) > 1e-4))
    named_peak = float(np.max(named_weights, initial=0.0))
    assert named_peak > 0.05
    assert varying_named_controls >= 3

    with np.load(job / "controls.npz", allow_pickle=False) as controls:
        expression = np.asarray(controls["expression"], dtype=np.float32)
        rotations = np.asarray(controls["rotations"], dtype=np.float32)
        translation = np.asarray(controls["translation"], dtype=np.float32)
        timestamps = np.asarray(controls["timestamps"], dtype=np.float32)
    assert expression.shape == (len(timestamps), 383)
    assert rotations.shape == (len(timestamps), 4, 3)
    assert translation.shape == (len(timestamps), 3)
    assert len(source_timestamps) in (len(timestamps), len(timestamps) + 1)
    assert np.isfinite(expression).all()
    tongue_coefficients = expression[:, 350:382]
    dense_peak = float(np.max(np.abs(tongue_coefficients), initial=0.0))
    animated_dense_frames = int(
        np.count_nonzero(np.max(np.abs(tongue_coefficients), axis=1) > 1e-5)
    )
    assert dense_peak > 0.05
    assert animated_dense_frames > len(timestamps) // 2

    adapter = GNMAdapter()
    tongue_mask = adapter.vertex_group("tongue") > 0.5
    tongue_basis = np.asarray(
        adapter.model.expression_basis[350:382, tongue_mask], dtype=np.float32
    )
    isolated_track = np.einsum(
        "tc,cvj->tvj", tongue_coefficients, tongue_basis, optimize=True
    )
    isolated_norm = np.linalg.norm(isolated_track, axis=2)
    peak_frame = int(np.argmax(np.max(isolated_norm, axis=1)))
    tongue_displacement_max_m = float(np.max(isolated_norm[peak_frame], initial=0.0))
    tongue_displacement_p95_m = float(np.percentile(isolated_norm[peak_frame], 95))
    assert tongue_displacement_p95_m > 0.00025
    assert tongue_displacement_max_m > 0.00050

    without_tongue = expression[peak_frame].copy()
    without_tongue[350:382] = 0.0
    full_mesh = adapter.mesh(expression=expression[peak_frame])
    tongue_zeroed_mesh = adapter.mesh(expression=without_tongue)
    dense_delta = np.linalg.norm(full_mesh - tongue_zeroed_mesh, axis=1)
    assert float(np.max(dense_delta[tongue_mask], initial=0.0)) > 0.00050
    assert float(np.max(dense_delta[~tongue_mask], initial=0.0)) < 1e-7

    with np.load(job / "animation-glb-mapping.npz", allow_pickle=False) as mapping:
        glb_to_gnm = np.asarray(mapping["glb_vertex_to_gnm_vertex"], dtype=np.int32)
        morph_weights = np.asarray(mapping["morph_weights"], dtype=np.float32)
        mapping_timestamps = np.asarray(mapping["timestamps"], dtype=np.float32)
    np.testing.assert_allclose(mapping_timestamps, timestamps, rtol=0.0, atol=1e-7)
    assert morph_weights.shape[0] == len(timestamps)
    native_tongue_indices = np.flatnonzero(tongue_mask)
    mapped_tongue_mask = tongue_mask[glb_to_gnm]
    assert set(native_tongue_indices.tolist()).issubset(set(glb_to_gnm.tolist()))

    document, binary = _glb_chunks(job / "animation.glb")
    primitive = document["meshes"][0]["primitives"][0]
    base_positions = _float_accessor(document, binary, primitive["attributes"]["POSITION"])
    morph_positions = np.stack(
        [
            _float_accessor(document, binary, target["POSITION"])
            for target in primitive["targets"]
        ]
    )
    assert base_positions.shape == (len(glb_to_gnm), 3)
    assert morph_positions.shape == (morph_weights.shape[1], len(glb_to_gnm), 3)
    per_target_tongue_motion = np.max(
        np.linalg.norm(morph_positions[:, mapped_tongue_mask], axis=2), axis=1
    )
    assert np.count_nonzero(per_target_tongue_motion > 1e-8) > 0

    reconstructed = base_positions + np.einsum(
        "r,rvj->vj", morph_weights[peak_frame], morph_positions, optimize=True
    )
    direct = adapter.mesh(
        expression=expression[peak_frame],
        rotations=rotations[peak_frame],
        translation=translation[peak_frame],
    )[glb_to_gnm]
    reconstruction_error = np.linalg.norm(
        reconstructed[mapped_tongue_mask] - direct[mapped_tongue_mask], axis=1
    )
    reconstruction_p95_m = float(np.percentile(reconstruction_error, 95))
    reconstruction_max_m = float(np.max(reconstruction_error, initial=0.0))
    assert reconstruction_p95_m <= 0.00010
    assert reconstruction_max_m <= 0.00050

    print(
        json.dumps(
            {
                "job": job.name,
                "frames": len(timestamps),
                "named_tongue_peak": named_peak,
                "varying_named_tongue_controls": varying_named_controls,
                "dense_tongue_peak": dense_peak,
                "animated_dense_frames": animated_dense_frames,
                "peak_frame": peak_frame,
                "tongue_displacement_p95_mm": tongue_displacement_p95_m * 1000.0,
                "tongue_displacement_max_mm": tongue_displacement_max_m * 1000.0,
                "mapped_native_tongue_vertices": len(native_tongue_indices),
                "mapped_glb_tongue_vertices": int(np.count_nonzero(mapped_tongue_mask)),
                "animated_glb_tongue_targets": int(
                    np.count_nonzero(per_target_tongue_motion > 1e-8)
                ),
                "glb_tongue_reconstruction_p95_mm": reconstruction_p95_m * 1000.0,
                "glb_tongue_reconstruction_max_mm": reconstruction_max_m * 1000.0,
            },
            sort_keys=True,
        )
    )


def test_oral_geometry_diagnostic_writes_contact_sheet(tmp_path: Path) -> None:
    job = _retained_job()
    _load_and_verify_job(job)
    output = tmp_path / "oral-diagnostic.png"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_oral_diagnostic.py"),
            "--job",
            str(job),
            "--output",
            str(output),
            "--frames",
            "4",
            "--size",
            "240",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "DIAGNOSTIC ONLY" in completed.stdout
    assert output.is_file() and output.stat().st_size > 10_000
    import cv2

    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[0] > 240 and image.shape[1] >= 240
    assert int(np.ptp(image)) > 100
