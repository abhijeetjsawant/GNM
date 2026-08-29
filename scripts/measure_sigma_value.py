#!/usr/bin/env python3
"""What is a per-observation sigma worth, in millimetres, against actual truth?

This is the number the strategy turns on. If a learned uncertainty channel buys
little, then MAMMA's advantage is elsewhere and Battle 4 loses its urgency; if it
buys a lot, the synthetic-detector campaign is earned with data rather than
inferred by analogy.

Three arms over the same noise realisation, so the comparison is paired:

* **(ii)** heteroscedastic noise, every observation declared equally reliable --
  the solver cannot tell the good from the bad;
* **(iii)** the same noise, with the true sigma handed to the solver;
* **G6**, the vacuity control -- arm (iii) re-run with a *constant* sigma equal to
  the population mean. It must land on arm (ii). If it does not, the (ii)->(iii)
  delta is a prior rebalance rather than a sigma effect, and the headline number
  means nothing.

The sigma channel is `confidence`, which `solve_sequence_positions` already
consumes as `sqrt(clip(confidence, 0, 1))` on the residual -- the same form as the
hand fit's inverse-variance weighting. No schema change is needed to measure this.

**One limitation, stated up front because it bounds the answer.**
`reconstruct_multiview` fixes `minimum_confidence` at 0.25 and does not expose it,
so confidences must stay above that floor or arm (iii) would also *drop*
observations arm (ii) keeps, confounding weighting with gating. Confidence is
therefore mapped into [0.26, 0.99], which is a weight range of only 0.51 to 0.995.
**The solver's weight model caps what any sigma can buy at about 2x.** A small
delta here is evidence about that cap as much as about sigma.

    python scripts/measure_sigma_value.py --seeds 5
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
WORKING_WIDTH, WORKING_HEIGHT = 1280, 720
CONFIDENCE_FLOOR, CONFIDENCE_CEILING = 0.26, 0.99

# Two components, because that is what the real detector does. Most observations
# are good; a minority are confidently wrong -- SOMA-77 reports peak confidences
# in [0.895, 0.992] whether or not the landmark is occluded, which is the whole
# reason a geometric substitute for visibility had to be invented.
#
# These are **fitted to our own detector**, not chosen. Cross-view epipolar
# disagreement was measured on the real fixture with subjects properly associated
# -- 23,987 camera-pair observations, deciles 1.06 / 2.75 / 6.15 / 11.65 / 22.79 /
# 63.33 px at 1280x720. Modelling a pair distance as |N(0,sa) + N(0,sb)| and
# fitting the mixture reproduces all six deciles to a mean |log ratio| of 0.051.
#
# The first draft used 2.0 px and 25% at 15 px, hand-set. That was optimistic by
# 2.8x on the bulk and 4.3x in the tail, which would have understated both the
# error and what sigma can do about it.
#
# Note the ratio: 11.8x between the two components, against a solver weight model
# that cannot express more than 1.95x. The real noise is far more heteroscedastic
# than the channel available to describe it.
CLEAN_SIGMA_PX = 5.5
BAD_SIGMA_PX = 65.0
BAD_FRACTION = 0.075


def sigma_to_confidence(sigma: np.ndarray, floor: float) -> np.ndarray:
    """Lorentzian inverse variance, as in the hand fit, scaled to the usable band.

    The solver reads `sqrt(confidence)` as the residual weight, so the expressible
    weight ratio is `sqrt(ceiling / floor)`. Raising the floor does not merely
    compress the scale -- it destroys the information, because two observations
    whose true sigmas differ elevenfold are handed weights that differ twofold.
    """

    weight = 1.0 / (1.0 + (sigma / CLEAN_SIGMA_PX) ** 2)
    confidence = CONFIDENCE_CEILING * weight / float(weight.max())
    return np.clip(confidence, floor + 1e-6, CONFIDENCE_CEILING)


def build_records(base, noise, confidence):
    """Rewrite the truth observations with noise and a confidence channel."""

    out = []
    for camera_index, rows in enumerate(base):
        stream = []
        for frame, row in enumerate(rows):
            people = []
            for person_index, person in enumerate(row["people"]):
                joints = {}
                for joint_index, (name, value) in enumerate(person["joints"].items()):
                    dx, dy = noise[camera_index, frame, person_index, joint_index]
                    joints[name] = {
                        "x": value["x"] + float(dx),
                        "y": value["y"] + float(dy),
                        "confidence": float(confidence[camera_index, frame, person_index, joint_index]),
                    }
                people.append({"index": person_index, "joints": joints})
            stream.append({**row, "people": people})
        out.append(stream)
    return out


def score(cameras, records, truth, scored_names, floor):
    tracks, diagnostics, positions, _ = cm.reconstruct_multiview(
        cameras, records, subject_count=2, sample_rate_hz=30,
        minimum_confidence=floor,
    )
    columns = [cm.JOINT_INDEX[name] for name in scored_names]
    positions = positions[:, :, columns, :]
    root = scored_names.index("root")
    cost = np.asarray([[np.nanmedian(np.linalg.norm(positions[a, :, root] - truth[b, :, root], axis=1))
                        for b in range(2)] for a in range(2)])
    pair = (0, 1) if cost[0, 0] + cost[1, 1] <= cost[0, 1] + cost[1, 0] else (1, 0)
    errors = np.concatenate([np.linalg.norm(positions[a] - truth[pair[a]], axis=2).ravel()
                             for a in range(2)]) * 1000.0
    finite = errors[np.isfinite(errors)]
    # Fixed denominator: unrecovered joints are failures, not absences. Battle 0's
    # lesson was that a threshold which discards the hard cases makes the survivors
    # look flat, and the mean improves as the noise grows.
    return {"mpjpe_mm": float(finite.mean()), "median_mm": float(np.median(finite)),
            "p95_mm": float(np.percentile(finite, 95)),
            "coverage": float(len(finite) / len(errors))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("artifacts/synthetic-truth"))
    parser.add_argument("--seeds", type=int, default=5)
    # The floor is both the gate and the bottom of the weight scale. Sweeping it
    # separates "sigma is worth little" from "sigma cannot be expressed".
    parser.add_argument("--floors", type=str, default="0.25,0.05,0.01,0.001")
    args = parser.parse_args()

    rig = cm.load_camera_rig(args.fixture / "camera-rig.json")
    cameras = tuple(camera.scaled(WORKING_WIDTH, WORKING_HEIGHT) for camera in rig)
    base = [[json.loads(line) for line in
             (args.fixture / "work" / f"{name}-observations.jsonl").read_text().splitlines() if line.strip()]
            for name in CAMERAS]
    subjects = np.load(args.fixture / "truth.npz", allow_pickle=True)["subjects_soma77_m"]
    scored_names = [n for n in cm.JOINT_INDEX if n in SOMA77_TO_AUTOANIM]
    truth = np.stack([np.stack([subjects[s, :, SOMA77_TO_AUTOANIM[n]] for n in scored_names], axis=1)
                      for s in range(subjects.shape[0])])

    frames = len(base[0])
    people = max(len(r["people"]) for rows in base for r in rows)
    joints = len(scored_names)
    shape = (len(CAMERAS), frames, people, joints)
    print(f"fixture: {frames} frames, {people} subjects, {joints} joints, "
          f"{args.seeds} seeds\nnoise: {CLEAN_SIGMA_PX} px clean / {BAD_SIGMA_PX} px on "
          f"{BAD_FRACTION*100:.0f}% of observations\n")

    floors = [float(v) for v in args.floors.split(",")]
    table = {}
    for floor in floors:
        results = {"ii": [], "iii": [], "g6": []}
        for seed in range(args.seeds):
            generator = np.random.default_rng(20260829 + seed)
            bad = generator.random(shape) < BAD_FRACTION
            sigma = np.where(bad, BAD_SIGMA_PX, CLEAN_SIGMA_PX)
            noise = generator.normal(0.0, 1.0, shape + (2,)) * sigma[..., None]
            uniform = np.full(shape, CONFIDENCE_CEILING)
            informed = sigma_to_confidence(sigma, floor)
            constant = np.full(shape, float(informed.mean()))
            for key, confidence in (("ii", uniform), ("iii", informed), ("g6", constant)):
                results[key].append(
                    score(cameras, build_records(base, noise, confidence), truth, scored_names, floor)
                )
        table[floor] = results
        ratio = (CONFIDENCE_CEILING / max(floor, 1e-9)) ** 0.5
        m2 = np.mean([r["mpjpe_mm"] for r in results["ii"]])
        m3 = np.mean([r["mpjpe_mm"] for r in results["iii"]])
        m6 = np.mean([r["mpjpe_mm"] for r in results["g6"]])
        s2 = np.std([r["mpjpe_mm"] for r in results["ii"]])
        print(f"floor {floor:<6} weight ratio up to {ratio:6.1f}x | "
              f"(ii) {m2:6.2f}  (iii) {m3:6.2f}  G6 {m6:6.2f} | "
              f"sigma buys {m2-m3:+6.2f} mm ({(m2-m3)/m2*100:+5.1f}%) | "
              f"G6 {'PASS' if abs(m6-m2)<=max(2*s2,0.5) else 'FAIL'}", flush=True)
    results = table[floors[0]]

    def summary(key):
        v = np.asarray([r["mpjpe_mm"] for r in results[key]])
        return float(v.mean()), float(v.std())

    m2, s2 = summary("ii"); m3, s3 = summary("iii"); m6, s6 = summary("g6")
    print(f"\n{'arm':>28} {'MPJPE':>10} {'sd':>7}")
    print(f"{'(ii)  uniform confidence':>28} {m2:>7.2f} mm {s2:>6.2f}")
    print(f"{'(iii) true sigma':>28} {m3:>7.2f} mm {s3:>6.2f}")
    print(f"{'G6   constant sigma':>28} {m6:>7.2f} mm {s6:>6.2f}")
    print(f"\nWHAT SIGMA BUYS: {m2 - m3:+.2f} mm  ({(m2-m3)/m2*100:+.1f}%)")
    spread = max(s2, s6)
    g6_ok = abs(m6 - m2) <= max(2.0 * spread, 0.5)
    print(f"G6 vacuity control: |constant - uniform| = {abs(m6-m2):.2f} mm against "
          f"a {max(2.0*spread,0.5):.2f} mm tolerance -> {'PASS' if g6_ok else 'FAIL'}")
    if not g6_ok:
        print("  The delta is a prior rebalance, not a sigma effect. Headline is void.")
    print(f"\nCap: confidence is confined to [{CONFIDENCE_FLOOR}, {CONFIDENCE_CEILING}] by the "
          f"solver's fixed 0.25 gate, so the weight ratio cannot exceed "
          f"{(CONFIDENCE_CEILING/CONFIDENCE_FLOOR)**0.5:.2f}x. Read the delta as a floor.")
    (args.fixture / "sigma-value.json").write_text(json.dumps({
        "noise_model": {"clean_sigma_px": CLEAN_SIGMA_PX, "bad_sigma_px": BAD_SIGMA_PX,
                        "bad_fraction": BAD_FRACTION},
        "seeds": args.seeds, "results": results,
        "arm_ii_mpjpe_mm": m2, "arm_iii_mpjpe_mm": m3, "g6_mpjpe_mm": m6,
        "sigma_value_mm": m2 - m3, "g6_pass": bool(g6_ok),
        "weight_ratio_cap": (CONFIDENCE_CEILING / CONFIDENCE_FLOOR) ** 0.5,
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
