#!/usr/bin/env python3
"""The cross-detector head comparison, on ONE frame set.

`adapter_head_comparison.py` scored each detector on its own common frames --
91/81 for Apple Vision, which needs both ears, against 146/138 for SOMA-77. That
is the composition shift this lane has been fooled by twice
(`BODY_LANE_PLAN.md` §1), so nothing from it may be quoted across detectors.

Here every statistic is computed on the intersection: frames where **both**
detectors resolved **every** landmark either is scored on. Each detector still
carries its own body controls, computed on that same intersection, so a reader
can see whether a head axis is good in absolute terms or merely on easy frames.

Blind to: joint convention, as before -- an Apple Vision ear is a surface
landmark and a SOMA-77 `Head` is a joint centre. Stability is comparable across
conventions; position is not.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import autoanim_gnm.commercial_multiview as cm  # noqa: E402
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_INDEX, load_camera_rig, load_observation_jsonl,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triangulate_soma import triangulate  # noqa: E402

CAMERAS = ("A001", "B001", "C001", "D001")
RIG = Path("artifacts/soma77-full/camera-rig.json")
AV = "artifacts/commercial-multiview-soma77/work/{cam}-observations.jsonl"
CACHE = Path("artifacts/head-lane/raw-cache.npz")

SOMA = {"Chest": 3, "Neck1": 4, "Neck2": 5, "Head": 6, "HeadEnd": 7, "Jaw": 8,
        "LeftEye": 9, "RightEye": 10, "LeftArm": 12, "LeftForeArm": 13,
        "LeftHand": 14, "RightArm": 40, "LeftShin": 68, "LeftFoot": 69}
SOMA_SLOT = {name: i for i, name in enumerate(SOMA)}

AV_PAIRS = [
    ("AV   ear axis", "right_ear", "left_ear"),
    ("AV   eye baseline", "right_eye", "left_eye"),
    ("AV   CONTROL shoulders", "right_shoulder", "left_shoulder"),
    ("AV   CONTROL shin L", "left_knee", "left_ankle"),
]
SOMA_PAIRS = [
    ("SOMA skull axis Head->HeadEnd", "Head", "HeadEnd"),
    ("SOMA eye baseline", "RightEye", "LeftEye"),
    ("SOMA CONTROL shoulders", "RightArm", "LeftArm"),
    ("SOMA CONTROL shin L", "LeftShin", "LeftFoot"),
]


def build_cache() -> tuple[np.ndarray, np.ndarray]:
    if CACHE.exists():
        blob = np.load(CACHE)
        return blob["av"], blob["soma"]
    rig = {camera.name: camera for camera in load_camera_rig(RIG)}
    observations = [load_observation_jsonl(AV.format(cam=name)) for name in CAMERAS]
    cameras = [rig[name].scaled(observations[0][0]["width"], observations[0][0]["height"])
               for name in CAMERAS]
    _, _, _, av = cm.reconstruct_multiview(cameras, observations, subject_count=2, sample_rate_hz=30)
    soma, _, _ = triangulate([SOMA[name] for name in SOMA])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, av=av, soma=soma)
    return av, soma


def segment(world: np.ndarray, mask: np.ndarray, ia: int, ib: int) -> dict:
    d = np.linalg.norm(world[mask, ia] - world[mask, ib], axis=1) * 1000.0
    vectors = world[:, ib] - world[:, ia]
    unit = np.full_like(vectors, np.nan)
    ok = mask & np.isfinite(vectors).all(axis=1)
    unit[ok] = vectors[ok] / np.linalg.norm(vectors[ok], axis=1, keepdims=True)
    index = np.flatnonzero(ok)
    pairs = index[1:][np.diff(index) == 1]
    dots = np.einsum("ni,ni->n", unit[pairs], unit[pairs - 1])
    ang = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return {
        "mean_mm": float(d.mean()), "sd_mm": float(d.std()),
        "sd_pct": float(100.0 * d.std() / d.mean()),
        "rot_n": int(ang.size),
        "rot_median_deg": float(np.median(ang)) if ang.size else None,
        "rot_p95_deg": float(np.percentile(ang, 95)) if ang.size else None,
        "rot_max_deg": float(ang.max()) if ang.size else None,
    }


def main() -> None:
    av, soma = build_cache()
    report: dict = {}
    for subject in range(2):
        a, s = av[subject], soma[subject]
        av_needed = sorted({JOINT_INDEX[x] for _, p, q in AV_PAIRS for x in (p, q)})
        soma_needed = sorted({SOMA_SLOT[x] for _, p, q in SOMA_PAIRS for x in (p, q)})
        mask = (np.isfinite(a[:, av_needed]).all(axis=(1, 2))
                & np.isfinite(s[:, soma_needed]).all(axis=(1, 2)))
        rows = {}
        for label, p, q in AV_PAIRS:
            rows[label] = segment(a, mask, JOINT_INDEX[p], JOINT_INDEX[q])
        for label, p, q in SOMA_PAIRS:
            rows[label] = segment(s, mask, SOMA_SLOT[p], SOMA_SLOT[q])

        # Opposition, both detectors, same frames, each against its own shoulders.
        opposition = {}
        for tag, world, axis_pair, shoulder_pair, table in (
            ("AV ear", a, ("right_ear", "left_ear"), ("right_shoulder", "left_shoulder"), JOINT_INDEX),
            ("AV eye", a, ("right_eye", "left_eye"), ("right_shoulder", "left_shoulder"), JOINT_INDEX),
            ("SOMA eye", s, ("RightEye", "LeftEye"), ("RightArm", "LeftArm"), SOMA_SLOT),
            ("SOMA skull", s, ("Head", "HeadEnd"), ("RightArm", "LeftArm"), SOMA_SLOT),
        ):
            axis = world[:, table[axis_pair[1]]] - world[:, table[axis_pair[0]]]
            ref = world[:, table[shoulder_pair[1]]] - world[:, table[shoulder_pair[0]]]
            ok = mask & np.isfinite(axis).all(axis=1) & np.isfinite(ref).all(axis=1)
            dots = np.einsum(
                "ni,ni->n",
                axis[ok] / np.linalg.norm(axis[ok], axis=1, keepdims=True),
                ref[ok] / np.linalg.norm(ref[ok], axis=1, keepdims=True),
            )
            opposition[tag] = {"frames": int(ok.sum()), "opposing": int((dots < 0).sum())}
        report[f"subject_{subject:02d}"] = {
            "frames_scored": int(mask.sum()),
            "segments": rows,
            "opposes_own_shoulder_axis": opposition,
        }

    Path("artifacts/head-lane/same-denominator-head.json").write_text(json.dumps(report, indent=2))
    for subject, block in report.items():
        print(f"\n=== {subject} -- {block['frames_scored']} frames, both detectors, all landmarks ===")
        print(f"{'axis':32s} {'mean mm':>9s} {'sd mm':>8s} {'sd%':>7s}   "
              f"{'rot med':>8s} {'p95':>8s} {'max':>8s}")
        for label, row in block["segments"].items():
            print(f"{label:32s} {row['mean_mm']:9.1f} {row['sd_mm']:8.1f} {row['sd_pct']:6.1f}%   "
                  f"{row['rot_median_deg']:8.2f} {row['rot_p95_deg']:8.2f} {row['rot_max_deg']:8.2f}")
        print("  opposes own shoulders:", json.dumps(block["opposes_own_shoulder_axis"]))


if __name__ == "__main__":
    main()
