#!/usr/bin/env python3
"""Fail-closed acceptance gate for a clean-room multiview comparison artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from autoanim_gnm.body import DETAILED_HUMANOID, forward_kinematics_positions
from autoanim_gnm.commercial_multiview import JOINT_INDEX


RETARGET_JOINTS = {
    "root": "Hips",
    "neck": "Neck",
    "left_shoulder": "LeftUpperArm",
    "left_elbow": "LeftLowerArm",
    "left_wrist": "LeftHand",
    "right_shoulder": "RightUpperArm",
    "right_elbow": "RightLowerArm",
    "right_wrist": "RightHand",
    "left_hip": "LeftUpperLeg",
    "left_knee": "LeftLowerLeg",
    "left_ankle": "LeftFoot",
    "right_hip": "RightUpperLeg",
    "right_knee": "RightLowerLeg",
    "right_ankle": "RightFoot",
}


class VerificationFailure(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationFailure(message)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationFailure(f"Cannot load JSON {path}: {error}") from error


def _video_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(completed.returncode == 0, f"ffprobe failed for {path}")
    try:
        return float(json.loads(completed.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise VerificationFailure(f"ffprobe returned no duration for {path}") from error


def verify(output: Path) -> dict[str, Any]:
    output = output.resolve(strict=True)
    report = _load_json(output / "run-report.json")
    _require(report.get("schema_version") == "autoanim.commercial-multiview/1.0", "Wrong report schema")
    _require(report.get("provider_id") == "autoanim_cleanroom_multiview", "Wrong provider")
    _require(report.get("camera_count") == 4, "Acceptance fixture must use all four cameras")
    _require(report.get("subject_count") == 2, "Acceptance fixture must reconstruct two subjects")
    frame_count = report.get("frame_count")
    _require(type(frame_count) is int and frame_count >= 150, "Acceptance window must contain at least 150 frames")
    # The pixel gates below were calibrated at a 1280-wide detector. Reported
    # residuals are in detector-native pixels, so they scale with detector width
    # and the same physical reconstruction would pass or fail depending on it.
    # Pin the fixture rather than let the unit drift silently.
    _require(report.get("detector_width") == 1280, "Acceptance fixture is calibrated at 1280 detector width")
    _require(report.get("frame_jpeg_quality") == 2, "Acceptance fixture requires pinned JPEG quality")
    for camera in ("A001", "B001", "C001", "D001"):
        stamp_path = output / "work" / "frames" / camera / "extraction.json"
        _require(stamp_path.is_file(), f"Camera {camera} has no frame extraction stamp")
        stamp = _load_json(stamp_path)
        _require(
            stamp.get("width") == report.get("detector_width")
            and stamp.get("jpeg_quality") == report.get("frame_jpeg_quality"),
            f"Camera {camera} frames were extracted under different settings than the report claims",
        )
        _require(
            stamp.get("source_sha256") == report.get("input_sha256", {}).get(camera),
            f"Camera {camera} frames were extracted from different footage than the report claims",
        )
    _require(float(report.get("valid_joint_fraction", 0.0)) >= 0.80, "Direct joint coverage is below 80%")
    _require(float(report.get("interpolated_joint_fraction", 1.0)) <= 0.20, "Interpolated joint fraction exceeds 20%")
    _require(float(report.get("median_reprojection_error_px", 1e9)) <= 6.0, "Median reprojection error exceeds 6 px")
    _require(float(report.get("p95_reprojection_error_px", 1e9)) <= 12.0, "P95 reprojection error exceeds 12 px")
    _require(float(report.get("maximum_reprojection_error_px", 1e9)) <= 25.0, "Maximum reprojection error exceeds 25 px")
    _require("association_objective_median" in report, "Association objective is not reported")
    _require("temporally_rejected_subject_frames" in report, "Temporal rejection count is not reported")
    dependencies = report.get("runtime_dependencies")
    _require(isinstance(dependencies, dict), "Runtime dependency declaration is absent")
    for prohibited in ("mamma", "mamma_outputs", "mamma_weights", "smplx_model"):
        _require(dependencies.get(prohibited) is False, f"Prohibited runtime dependency enabled: {prohibited}")
    _require(report.get("production_claim") is False, "Research fixture must not claim production qualification")

    for page in ("review.html", "only-3d-review.html"):
        _require((output / page).stat().st_size > 2_000, f"Missing or empty viewer: {page}")
    source_duration_s = _video_duration(output / "source.mp4")
    _require(4.8 <= source_duration_s <= 5.1, "Source comparison clip is not the expected five-second window")

    track_results: list[dict[str, Any]] = []
    all_retarget_errors_m: list[np.ndarray] = []
    for subject in range(2):
        prefix = output / f"subject-{subject:02d}"
        manifest = _load_json(prefix.with_suffix(".body-track.json"))
        _require(len(manifest.get("joint_names", [])) == 55, f"Subject {subject} JSON is not AutoAnim-55")
        with np.load(prefix.with_suffix(".body-track.npz"), allow_pickle=False) as archive:
            ticks = archive["ticks"]
            roots = archive["root_translation_m"]
            rotations = archive["local_rotations_xyzw"]
            contacts = archive["foot_contacts"]
            world = archive["triangulated_world_positions_z_up_m"]
        _require(ticks.shape == (frame_count,), f"Subject {subject} tick count differs")
        _require(roots.shape == (frame_count, 3), f"Subject {subject} root shape differs")
        _require(rotations.shape == (frame_count, 55, 4), f"Subject {subject} rotation shape differs")
        _require(contacts.shape == (frame_count, 2), f"Subject {subject} contact shape differs")
        _require(world.shape == (frame_count, 19, 3), f"Subject {subject} sparse-world shape differs")
        for name, array in (("roots", roots), ("rotations", rotations), ("world", world)):
            _require(np.isfinite(array).all(), f"Subject {subject} {name} contain nonfinite values")
        norms = np.linalg.norm(rotations, axis=2)
        _require(np.allclose(norms, 1.0, atol=2e-5), f"Subject {subject} quaternions are not normalized")
        _require(int(np.count_nonzero(contacts)) > 0, f"Subject {subject} has no reconstructed contacts")
        rig_positions = forward_kinematics_positions(
            roots,
            rotations,
            skeleton=DETAILED_HUMANOID,
        )
        target_positions = world[..., (0, 2, 1)].copy()
        target_positions[..., 2] *= -1.0
        subject_errors_m = np.concatenate(
            tuple(
                np.linalg.norm(
                    rig_positions[:, DETAILED_HUMANOID.index(rig_joint)]
                    - target_positions[:, JOINT_INDEX[source_joint]],
                    axis=1,
                )
                for source_joint, rig_joint in RETARGET_JOINTS.items()
            )
        )
        all_retarget_errors_m.append(subject_errors_m)
        glb = prefix.with_suffix(".glb")
        _require(glb.stat().st_size > 100_000, f"Subject {subject} GLB is unexpectedly small")
        _require(glb.read_bytes()[:4] == b"glTF", f"Subject {subject} GLB header is invalid")
        track_results.append(
            {
                "subject": subject,
                "frames": int(len(ticks)),
                "joints": int(rotations.shape[1]),
                "contact_samples": int(np.count_nonzero(contacts)),
                "retarget_endpoint_median_m": float(np.median(subject_errors_m)),
                "retarget_endpoint_p95_m": float(np.percentile(subject_errors_m, 95)),
                "glb_bytes": glb.stat().st_size,
            }
        )

    retarget_errors_m = np.concatenate(all_retarget_errors_m)
    retarget_median_m = float(np.median(retarget_errors_m))
    retarget_p95_m = float(np.percentile(retarget_errors_m, 95))
    # This is deliberately a loose sparse-baseline gate, not a production
    # body-fit claim. P3 in the plan replaces the fixed generic proportions.
    _require(retarget_median_m <= 0.18, "Sparse-to-rig median endpoint error exceeds 180 mm")
    _require(retarget_p95_m <= 0.40, "Sparse-to-rig P95 endpoint error exceeds 400 mm")

    return {
        "status": "pass",
        "artifact": str(output),
        "source_duration_s": source_duration_s,
        "metrics": {
            "valid_joint_fraction": report["valid_joint_fraction"],
            "interpolated_joint_fraction": report["interpolated_joint_fraction"],
            "median_reprojection_error_px": report["median_reprojection_error_px"],
            "p95_reprojection_error_px": report["p95_reprojection_error_px"],
            "maximum_reprojection_error_px": report["maximum_reprojection_error_px"],
            "association_objective_median": report["association_objective_median"],
            "temporally_rejected_subject_frames": report["temporally_rejected_subject_frames"],
            "retarget_endpoint_median_m": retarget_median_m,
            "retarget_endpoint_p95_m": retarget_p95_m,
        },
        "tracks": track_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args()
    try:
        result = verify(arguments.artifact)
    except (OSError, VerificationFailure) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
