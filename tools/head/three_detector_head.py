#!/usr/bin/env python3
"""All three integrated detectors' head landmarks, on ONE frame set.

`docs/HEAD_FEET_HANDS_PLAN.md` §2: "Before proposing any new detector, enumerate
what the two already-integrated adapters emit ... Score them on **this footage,
same denominator**, against SOMA-77." This does that for all three:

| adapter | head landmarks it emits on this footage |
|---|---|
| `soma77_pose.py` | `Head`(6) `HeadEnd`(7) `Jaw`(8) `LeftEye`(9) `RightEye`(10). **No ears** -- SOMA-77 has none, and `left_ear`/`right_ear` are populated on 0 frames. |
| `apple_vision_pose.swift` | nose, both eyes, **both ears** (`.leftEar`/`.rightEar`, lines 59-60). Already run for this window: its detections are the boxes the SOMA-77 worker was driven from. |
| `mediapipe_pose.py` | nose, both eyes, **both ears** (indices 7/8). Run here for the first time on this footage, from the cached `pose_landmarker_heavy.task`. |

Every statistic is on the intersection: frames where **all three** detectors
resolved **every** landmark any of them is scored on. Each detector carries its
own body controls on that same set, and the head axis is reported as a ratio to
**both** controls -- shoulder width and shin -- because the two normalisations
disagree and quoting only the flattering one is this lane's recurring defect.

Two things this deliberately does NOT report:
  * the sign test of a skull long axis against the shoulder axis. Those axes are
    near-perpendicular, so the sign of their dot product is meaningless and an
    earlier run of this comparison produced a spurious "36/95 opposing". The
    skull axis is tested against the neck-up direction instead, where opposition
    really is impossible.
  * MediaPipe confidences as a quality channel. Its landmarker reports presence,
    saturated near 1.0, and it is not comparable to Apple Vision's or SOMA-77's.

Blind to: joint convention. An Apple Vision ear, a MediaPipe ear and a SOMA-77
`Head` are different points under different definitions. Stability compares
across conventions; position does not.
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
CACHE = Path("artifacts/head-lane/three-detector-cache.npz")
SOURCES = {
    "apple_vision": "artifacts/commercial-multiview-soma77/work/{cam}-observations.jsonl",
    "mediapipe": "artifacts/head-lane/mediapipe/{cam}-observations.jsonl",
}
SOMA = {"Neck2": 5, "Head": 6, "HeadEnd": 7, "LeftEye": 9, "RightEye": 10,
        "LeftArm": 12, "RightArm": 40, "LeftShin": 68, "LeftFoot": 69}
SOMA_SLOT = {name: i for i, name in enumerate(SOMA)}

ROWS = [
    # detector, label, a, b, kind
    ("apple_vision", "ear axis", "right_ear", "left_ear", "head"),
    ("apple_vision", "eye baseline", "right_eye", "left_eye", "head"),
    ("apple_vision", "shoulder width", "right_shoulder", "left_shoulder", "control"),
    ("apple_vision", "shin", "left_knee", "left_ankle", "control"),
    ("mediapipe", "ear axis", "right_ear", "left_ear", "head"),
    ("mediapipe", "eye baseline", "right_eye", "left_eye", "head"),
    ("mediapipe", "shoulder width", "right_shoulder", "left_shoulder", "control"),
    ("mediapipe", "shin", "left_knee", "left_ankle", "control"),
    ("soma77", "skull axis Head->HeadEnd", "Head", "HeadEnd", "head"),
    ("soma77", "eye baseline", "RightEye", "LeftEye", "head"),
    ("soma77", "shoulder width", "RightArm", "LeftArm", "control"),
    ("soma77", "shin", "LeftShin", "LeftFoot", "control"),
]


def build() -> dict[str, np.ndarray]:
    if CACHE.exists():
        blob = np.load(CACHE)
        return {key: blob[key] for key in blob.files}
    rig = {camera.name: camera for camera in load_camera_rig(RIG)}
    out: dict[str, np.ndarray] = {}
    for detector, template in SOURCES.items():
        observations = [load_observation_jsonl(template.format(cam=name)) for name in CAMERAS]
        cameras = [rig[name].scaled(observations[0][0]["width"], observations[0][0]["height"])
                   for name in CAMERAS]
        _, _, _, raw = cm.reconstruct_multiview(
            cameras, observations, subject_count=2, sample_rate_hz=30)
        out[detector] = raw
    out["soma77"] = triangulate([SOMA[name] for name in SOMA])[0]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def index_of(detector: str, name: str) -> int:
    return SOMA_SLOT[name] if detector == "soma77" else JOINT_INDEX[name]


def measure(world: np.ndarray, mask: np.ndarray, ia: int, ib: int) -> dict:
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
        "mean_mm": float(d.mean()), "sd_mm": float(d.std()),
        "sd_pct": float(100.0 * d.std() / d.mean()),
        "rot_median_deg": float(np.median(ang)), "rot_p95_deg": float(np.percentile(ang, 95)),
        "rot_max_deg": float(ang.max()),
    }


def main() -> None:
    worlds = build()
    report: dict = {}
    for subject in range(2):
        mask = np.ones(worlds["soma77"].shape[1], dtype=bool)
        for detector, _, a, b, _ in ROWS:
            world = worlds[detector][subject]
            for name in (a, b):
                mask &= np.isfinite(world[:, index_of(detector, name)]).all(axis=1)
        rows: dict[str, dict] = {}
        for detector, label, a, b, kind in ROWS:
            world = worlds[detector][subject]
            rows[f"{detector} :: {label}"] = measure(
                world, mask, index_of(detector, a), index_of(detector, b)) | {"kind": kind}
        # Normalise every head axis against BOTH of its own detector's controls.
        for detector in ("apple_vision", "mediapipe", "soma77"):
            shoulders = rows[f"{detector} :: shoulder width"]["sd_pct"]
            shin = rows[f"{detector} :: shin"]["sd_pct"]
            for key, row in rows.items():
                if key.startswith(detector) and row["kind"] == "head":
                    row["ratio_to_own_shoulder_control"] = row["sd_pct"] / shoulders
                    row["ratio_to_own_shin_control"] = row["sd_pct"] / shin

        # Sign sanity, each axis against a reference where opposition is truly
        # impossible: a lateral head axis against the shoulders, a skull LONG
        # axis against neck-up.
        sanity: dict[str, dict] = {}
        for detector, label, a, b, ref_a, ref_b in (
            ("apple_vision", "ear vs shoulders", "right_ear", "left_ear", "right_shoulder", "left_shoulder"),
            ("apple_vision", "eye vs shoulders", "right_eye", "left_eye", "right_shoulder", "left_shoulder"),
            ("mediapipe", "ear vs shoulders", "right_ear", "left_ear", "right_shoulder", "left_shoulder"),
            ("mediapipe", "eye vs shoulders", "right_eye", "left_eye", "right_shoulder", "left_shoulder"),
            ("soma77", "eye vs shoulders", "RightEye", "LeftEye", "RightArm", "LeftArm"),
            ("soma77", "skull axis vs neck-up", "Head", "HeadEnd", "Neck2", "Head"),
        ):
            world = worlds[detector][subject]
            axis = world[:, index_of(detector, b)] - world[:, index_of(detector, a)]
            ref = world[:, index_of(detector, ref_b)] - world[:, index_of(detector, ref_a)]
            ok = mask & np.isfinite(axis).all(axis=1) & np.isfinite(ref).all(axis=1)
            dots = np.einsum("ni,ni->n",
                             axis[ok] / np.linalg.norm(axis[ok], axis=1, keepdims=True),
                             ref[ok] / np.linalg.norm(ref[ok], axis=1, keepdims=True))
            sanity[f"{detector} :: {label}"] = {
                "frames": int(ok.sum()), "opposing": int((dots < 0).sum())}
        report[f"subject_{subject:02d}"] = {
            "frames_scored": int(mask.sum()), "segments": rows, "sign_sanity": sanity}

    Path("artifacts/head-lane/three-detector-head.json").write_text(json.dumps(report, indent=2))
    for subject, block in report.items():
        print(f"\n=== {subject} -- {block['frames_scored']} frames, all three detectors ===")
        print(f"{'detector :: axis':40s} {'mean mm':>8s} {'sd%':>7s} {'/shldr':>7s} {'/shin':>6s} "
              f"{'rot med':>8s} {'p95':>7s}")
        for label, row in block["segments"].items():
            rs = row.get("ratio_to_own_shoulder_control")
            rn = row.get("ratio_to_own_shin_control")
            print(f"{label:40s} {row['mean_mm']:8.1f} {row['sd_pct']:6.1f}% "
                  f"{'-' if rs is None else format(rs, '6.2f'):>7s} "
                  f"{'-' if rn is None else format(rn, '5.2f'):>6s} "
                  f"{row['rot_median_deg']:8.2f} {row['rot_p95_deg']:7.2f}")
        print("  sign sanity (opposing/frames):", ", ".join(
            f"{k.split(' :: ')[0][:4]}/{k.split(' :: ')[1]}={v['opposing']}/{v['frames']}"
            for k, v in block["sign_sanity"].items()))


if __name__ == "__main__":
    main()
