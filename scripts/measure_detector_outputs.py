#!/usr/bin/env python3
"""What each detector output is worth — corrected.

Supersedes the first pass in `measure_sigma_value.py`, which had three defects
found by adversarial review and confirmed independently:

* the noise model was fitted to `_epipolar_distance_px`, which returns the
  **symmetric** distance (the sum of two one-sided distances), as though it were
  one-sided. Measured over-statement: 1.962x. Corrected model below;
* the `both` arm was byte-identical to the `visible` arm, because
  `sigma_to_confidence` normalised by `weight.max()` and mapped every clean
  observation to the ceiling. Dropped rather than reported;
* `solve_sequence_positions` applied the weight *after* the soft-l1 compression,
  so a confidence could shrink an outlier's contribution but not change what
  counted as one. Both orderings are measured here.

Every arm shares one noise realisation, so comparisons are paired.

    python scripts/measure_detector_outputs.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers" / "commercial_multiview"))

from autoanim_gnm import commercial_multiview as cm
from soma77_pose import SOMA77_TO_AUTOANIM

CAMERAS = ("A001", "B001", "C001", "D001")
FLOOR, CEILING = 0.25, 0.99

# Fitted to the ONE-SIDED epipolar ladder measured on the reference fixture over
# 23,987 associated camera-pair observations: 0.53 / 1.37 / 3.11 / 6.04 / 11.49 /
# 32.79 px. Mean |log ratio| 0.043. A lognormal continuum on the same ladder scores
# 0.139 -- it cannot reach the measured p95 -- so the data does prefer a mixture.
CLEAN_PX, BAD_PX, BAD_FRACTION = 2.75, 36.0, 0.07
LOGNORMAL_MEDIAN_PX, LOGNORMAL_SIGMA = 2.75, 0.85


def load(fixture: Path):
    cameras = tuple(c.scaled(1280, 720) for c in cm.load_camera_rig(fixture / "camera-rig.json"))
    base = [[json.loads(line) for line in
             (fixture / "work" / f"{n}-observations.jsonl").read_text().splitlines() if line.strip()]
            for n in CAMERAS]
    subjects = np.load(fixture / "truth.npz", allow_pickle=True)["subjects_soma77_m"]
    names = [n for n in cm.JOINT_INDEX if n in SOMA77_TO_AUTOANIM]
    truth = np.stack([np.stack([subjects[s, :, SOMA77_TO_AUTOANIM[n]] for n in names], axis=1)
                      for s in range(subjects.shape[0])])
    return cameras, base, truth, names


def records(base, noise, confidence):
    out = []
    for camera_index, rows in enumerate(base):
        stream = []
        for frame, row in enumerate(rows):
            people = []
            for person_index, person in enumerate(row["people"]):
                joints = {}
                for joint_index, (name, value) in enumerate(person["joints"].items()):
                    dx, dy = noise[camera_index, frame, person_index, joint_index]
                    joints[name] = {"x": value["x"] + float(dx), "y": value["y"] + float(dy),
                                    "confidence": float(confidence[camera_index, frame, person_index, joint_index])}
                people.append({"index": person_index, "joints": joints})
            stream.append({**row, "people": people})
        out.append(stream)
    return out


def score(cameras, recs, truth, names, before_loss=False):
    _, _, positions, _ = cm.reconstruct_multiview(
        cameras, recs, subject_count=2, sample_rate_hz=30,
        minimum_confidence=FLOOR, weight_before_loss=before_loss,
    )
    positions = positions[:, :, [cm.JOINT_INDEX[n] for n in names], :]
    root = names.index("root")
    cost = np.asarray([[np.nanmedian(np.linalg.norm(positions[a, :, root] - truth[b, :, root], axis=1))
                        for b in range(2)] for a in range(2)])
    pair = (0, 1) if cost[0, 0] + cost[1, 1] <= cost[0, 1] + cost[1, 0] else (1, 0)
    errors = np.concatenate([np.linalg.norm(positions[a] - truth[pair[a]], axis=2).ravel()
                             for a in range(2)]) * 1000.0
    finite = errors[np.isfinite(errors)]
    return float(finite.mean()), float(len(finite) / len(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("artifacts/synthetic-truth"))
    parser.add_argument("--seeds", type=int, default=3)
    # The floor is both the gate and the bottom of the weight scale, so at 0.25 the
    # expressible weight ratio is only 2x while the fitted sigma ratio is 13x. A
    # sigma arm run at the default floor is still measuring the channel.
    parser.add_argument("--floor", type=float, default=0.25)
    args = parser.parse_args()
    global FLOOR
    FLOOR = args.floor
    cameras, base, truth, names = load(args.fixture)
    shape = (len(CAMERAS), len(base[0]), truth.shape[0], len(names))
    print(f"fixture {args.fixture}, {shape[1]} frames, {args.seeds} seeds")
    print(f"corrected noise: mixture {CLEAN_PX} px / {BAD_FRACTION*100:.0f}% at {BAD_PX} px, "
          f"and lognormal median {LOGNORMAL_MEDIAN_PX} px sigma_log {LOGNORMAL_SIGMA}\n")
    results: dict[str, dict[str, float]] = {}
    for model in ("mixture", "lognormal"):
        rows: dict[str, list[float]] = {}
        for seed in range(args.seeds):
            g = np.random.default_rng(20260829 + seed)
            if model == "mixture":
                bad = g.random(shape) < BAD_FRACTION
                sigma = np.where(bad, BAD_PX, CLEAN_PX)
            else:
                sigma = LOGNORMAL_MEDIAN_PX * np.exp(g.normal(0.0, LOGNORMAL_SIGMA, shape))
                # a continuum has no "bad" set; flag the same fraction by quantile
                bad = sigma >= np.quantile(sigma, 1.0 - BAD_FRACTION)
            noise = g.normal(0.0, 1.0, shape + (2,)) * sigma[..., None]
            weight = 1.0 / (1.0 + (sigma / CLEAN_PX) ** 2)
            informed = np.clip(CEILING * weight / float(weight.max()), FLOOR + 1e-6, CEILING)
            arms = {
                "no channel": (np.full(shape, CEILING), False),
                "sigma, weight after loss": (informed, False),
                "sigma, weight before loss": (informed, True),
                "visibility": (np.where(bad, FLOOR * 0.5, CEILING), False),
                "visibility, shuffled (control)": (
                    np.where(g.permutation(bad.ravel()).reshape(shape), FLOOR * 0.5, CEILING), False),
            }
            for label, (confidence, before) in arms.items():
                value, coverage = score(cameras, records(base, noise, confidence), truth, names, before)
                rows.setdefault(label, []).append(value)
        base_value = float(np.mean(rows["no channel"]))
        print(f"--- {model} ---")
        print(f"{'arm':>34} {'MPJPE':>10} {'sd':>6} {'buys':>9}")
        for label, values in rows.items():
            mean = float(np.mean(values))
            print(f"{label:>34} {mean:>7.2f} mm {np.std(values):>5.2f} {base_value - mean:>+8.2f}")
            results[f"{model}/{label}"] = {"mpjpe_mm": mean, "sd": float(np.std(values)),
                                           "buys_mm": base_value - mean}
        print()
    (args.fixture / "detector-outputs.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
