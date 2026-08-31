#!/usr/bin/env python3
"""No gate a constant can pass -- applied to the ears themselves.

`HEAD_ORIENTATION_MEASURED.md` §4 finds Apple Vision's ear axis the best-conditioned
head input on this footage. Before that supports anything, it has to survive the
question the finger gate exists to ask: **is the ear an image measurement, or is it a
template hung off the face landmarks the detector already found?**

The null: per camera, per subject, predict each ear's 2D from `nose`, both eyes and
`neck` by least squares -- a pure face template with no ear evidence in it. Fit on even
frames, predict odd, and vice versa, so no frame is ever predicted by a model that saw
it. Then push the **predicted** ears through the identical triangulation and axis
statistics.

Two outcomes, and they mean opposite things:
  * the null reproduces the observed ear-axis stability -> the stability is inherited
    from the face landmarks, the ear adds nothing, and §4's evidence evaporates;
  * the null is markedly worse -> the detected ear carries information the face
    landmarks do not, which is what "image-measured" means operationally here.

A per-camera affine template is a *generous* null: it is allowed to learn a different
ear offset for every camera and every subject, which a real crop-conditioned detector
could not do. Beating a generous null is the strong direction.

Blind to: whether the ear is in the anatomically right place. A detector could measure
some real image feature that is not an ear and pass this.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import autoanim_gnm.commercial_multiview as cm  # noqa: E402
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_INDEX, load_camera_rig, load_observation_jsonl, triangulate_point,
)

CAMERAS = ("A001", "B001", "C001", "D001")
RIG = Path("artifacts/soma77-full/camera-rig.json")
AV = "artifacts/commercial-multiview-soma77/work/{cam}-observations.jsonl"
PREDICTORS = ("nose", "left_eye", "right_eye", "neck")
TARGETS = ("left_ear", "right_ear")


def fit_predict(features: np.ndarray, target: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Two-fold held-out affine prediction. Rows without full data stay NaN."""
    out = np.full_like(target, np.nan)
    design = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    parity = np.arange(len(features)) % 2
    for fold in (0, 1):
        train = ok & (parity != fold)
        test = ok & (parity == fold)
        if train.sum() < design.shape[1] + 2 or not test.any():
            continue
        coefficients, *_ = np.linalg.lstsq(design[train], target[train], rcond=None)
        out[test] = design[test] @ coefficients
    return out


def axis_stats(world: np.ndarray, mask: np.ndarray, ia: int, ib: int) -> dict:
    d = np.linalg.norm(world[mask, ia] - world[mask, ib], axis=1) * 1000.0
    vectors = world[:, ib] - world[:, ia]
    ok = mask & np.isfinite(vectors).all(axis=1)
    unit = np.full_like(vectors, np.nan)
    unit[ok] = vectors[ok] / np.linalg.norm(vectors[ok], axis=1, keepdims=True)
    index = np.flatnonzero(ok)
    pairs = index[1:][np.diff(index) == 1]
    ang = np.degrees(np.arccos(np.clip(
        np.einsum("ni,ni->n", unit[pairs], unit[pairs - 1]), -1.0, 1.0)))
    return {
        "frames": int(mask.sum()), "mean_mm": float(d.mean()), "sd_mm": float(d.std()),
        "sd_pct": float(100.0 * d.std() / d.mean()),
        "rot_median_deg": float(np.median(ang)) if ang.size else None,
        "rot_p95_deg": float(np.percentile(ang, 95)) if ang.size else None,
    }


def main() -> None:
    rig = {camera.name: camera for camera in load_camera_rig(RIG)}
    observations = [load_observation_jsonl(AV.format(cam=name)) for name in CAMERAS]
    frames = len(observations[0])
    cameras = [rig[name].scaled(observations[0][0]["width"], observations[0][0]["height"])
               for name in CAMERAS]

    calls: list[np.ndarray] = []
    real = cm.associate_frame_graph

    def recording(*args, **kwargs):
        associated, cost = real(*args, **kwargs)
        calls.append(np.array(associated, copy=True))
        return associated, cost

    cm.reconstruct_multiview(cameras, observations, subject_count=2,
                             sample_rate_hz=30, associator=recording)
    associated = np.stack(calls)  # [frame, subject, camera, joint, 3]

    report: dict = {}
    for subject in range(2):
        # --- build the null's 2D -------------------------------------------------
        null = associated[:, subject].copy()
        residuals: dict[str, list[float]] = {name: [] for name in TARGETS}
        for camera in range(len(CAMERAS)):
            features = np.concatenate(
                [associated[:, subject, camera, JOINT_INDEX[name], :2] for name in PREDICTORS],
                axis=1,
            )
            for name in TARGETS:
                target = associated[:, subject, camera, JOINT_INDEX[name], :2]
                ok = np.isfinite(features).all(axis=1) & np.isfinite(target).all(axis=1)
                predicted = fit_predict(features, target, ok)
                good = ok & np.isfinite(predicted).all(axis=1)
                residuals[name].extend(
                    np.linalg.norm(predicted[good] - target[good], axis=1).tolist())
                null[:, camera, JOINT_INDEX[name], :2] = predicted
                # The null keeps the real confidences, so triangulation gates it
                # exactly as it gates the observed ears.

        # --- triangulate observed and null ears the same way ---------------------
        def build(source: np.ndarray) -> np.ndarray:
            world = np.full((frames, len(JOINT_INDEX), 3), np.nan)
            for frame in range(frames):
                for name in TARGETS + ("left_shoulder", "right_shoulder"):
                    joint = JOINT_INDEX[name]
                    point = triangulate_point(
                        cameras, source[frame, :, joint, :2],
                        np.nan_to_num(source[frame, :, joint, 2]), pixel_scale=1.0)
                    if point is not None:
                        world[frame, joint] = point.position_world_m
            return world

        observed_world = build(associated[:, subject])
        null_world = build(null)

        needed = [JOINT_INDEX[name] for name in TARGETS]
        mask = (np.isfinite(observed_world[:, needed]).all(axis=(1, 2))
                & np.isfinite(null_world[:, needed]).all(axis=(1, 2)))
        report[f"subject_{subject:02d}"] = {
            "held_out_2d_residual_px": {
                name: {
                    "n": len(residuals[name]),
                    "median": float(np.median(residuals[name])),
                    "p95": float(np.percentile(residuals[name], 95)),
                } for name in TARGETS
            },
            "ear_axis_observed": axis_stats(
                observed_world, mask, JOINT_INDEX["right_ear"], JOINT_INDEX["left_ear"]),
            "ear_axis_null_template": axis_stats(
                null_world, mask, JOINT_INDEX["right_ear"], JOINT_INDEX["left_ear"]),
        }

    Path("artifacts/head-lane/ear-null-template.json").write_text(json.dumps(report, indent=2))
    for subject, block in report.items():
        print(f"\n=== {subject} ===")
        for name, row in block["held_out_2d_residual_px"].items():
            print(f"  held-out 2D residual, {name:10s}: median {row['median']:6.2f} px, "
                  f"p95 {row['p95']:7.2f} px  (n={row['n']})")
        for label in ("ear_axis_observed", "ear_axis_null_template"):
            row = block[label]
            print(f"  {label:24s} frames {row['frames']:3d}  mean {row['mean_mm']:6.1f} mm  "
                  f"sd {row['sd_mm']:6.1f} mm ({row['sd_pct']:5.1f}%)  "
                  f"rot med {row['rot_median_deg']:6.2f}  p95 {row['rot_p95_deg']:7.2f}")


if __name__ == "__main__":
    main()
