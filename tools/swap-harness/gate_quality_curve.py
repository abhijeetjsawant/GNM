#!/usr/bin/env python3
"""How good must a visibility head be before gating pays?

`gate_vs_weight.py` establishes the two endpoints on this fixture: a perfect
oracle gate buys +3.09 mm, and a gate with the same drop count but shuffled
labels costs -3.35 mm. Battle 4 has to ship a head somewhere between them, so the
question that matters is not "is visibility worth something" but **where the
curve crosses zero.** Below that line a visibility head makes the reconstruction
worse than having no channel at all.

Two sweeps, because one number cannot express it:

* **recall** -- what fraction of the truly-bad observations the head flags, at
  zero false positives. This is the friendly axis: missing a bad observation
  costs only what that observation costs.
* **false-positive rate** -- what fraction of *good* observations the head
  wrongly drops, at full recall. This is the dangerous axis: every false positive
  throws away real evidence, and at 4 cameras a joint has little to spare.

No gate a constant can pass: recall 0.0 with zero false positives is exactly the
"no channel" arm and must land on it, and the shuffled arm from
`gate_vs_weight.py` is the other end.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workers" / "commercial_multiview"))
from autoanim_gnm import commercial_multiview as cm
from soma77_pose import SOMA77_TO_AUTOANIM

CAMERAS = ("A001", "B001", "C001", "D001")
FLOOR, CEILING, DROP = 0.25, 0.99, 0.125      # DROP is below FLOOR: an eligibility drop
CLEAN_PX, BAD_PX, BAD_FRACTION = 2.75, 36.0, 0.07
FIX = ROOT / "artifacts/synthetic-truth"
SEEDS = 3

cameras = tuple(c.scaled(1280, 720) for c in cm.load_camera_rig(FIX / "camera-rig.json"))
base = [[json.loads(l) for l in (FIX / "work" / f"{n}-observations.jsonl").read_text().splitlines() if l.strip()]
        for n in CAMERAS]
subjects = np.load(FIX / "truth.npz", allow_pickle=True)["subjects_soma77_m"]
names = [n for n in cm.JOINT_INDEX if n in SOMA77_TO_AUTOANIM]
truth = np.stack([np.stack([subjects[s, :, SOMA77_TO_AUTOANIM[n]] for n in names], axis=1)
                  for s in range(subjects.shape[0])])
shape = (len(CAMERAS), len(base[0]), subjects.shape[0], len(names))

def records(conf, noise):
    out = []
    for ci, rows in enumerate(base):
        stream = []
        for f, row in enumerate(rows):
            people = []
            for pi, person in enumerate(row["people"]):
                joints = {}
                for ji, (nm, v) in enumerate(person["joints"].items()):
                    dx, dy = noise[ci, f, pi, ji]
                    joints[nm] = {"x": v["x"] + float(dx), "y": v["y"] + float(dy),
                                  "confidence": float(conf[ci, f, pi, ji])}
                people.append({"index": pi, "joints": joints})
            stream.append({**row, "people": people})
        out.append(stream)
    return out

def score(conf, noise):
    _, _, positions, _ = cm.reconstruct_multiview(
        cameras, records(conf, noise), subject_count=2, sample_rate_hz=30,
        minimum_confidence=FLOOR)
    positions = positions[:, :, [cm.JOINT_INDEX[n] for n in names], :]
    root = names.index("root")
    cost = np.asarray([[np.nanmedian(np.linalg.norm(positions[a, :, root] - truth[b, :, root], axis=1))
                        for b in range(2)] for a in range(2)])
    pair = (0, 1) if cost[0, 0] + cost[1, 1] <= cost[0, 1] + cost[1, 0] else (1, 0)
    err = np.concatenate([np.linalg.norm(positions[a] - truth[pair[a]], axis=2).ravel()
                          for a in range(2)]) * 1000.0
    fin = err[np.isfinite(err)]
    return float(fin.mean()), float(len(fin) / len(err))

RECALLS = [0.0, 0.25, 0.50, 0.75, 1.0]
FPRS    = [0.0, 0.02, 0.05, 0.10, 0.20]
acc, t0 = {}, time.time()
for seed in range(SEEDS):
    g = np.random.default_rng(20260829 + seed)
    bad = g.random(shape) < BAD_FRACTION
    sigma = np.where(bad, BAD_PX, CLEAN_PX)
    noise = g.normal(0.0, 1.0, shape + (2,)) * sigma[..., None]
    acc.setdefault("no channel", []).append(score(np.full(shape, CEILING), noise))
    for r in RECALLS:
        flag = bad & (g.random(shape) < r)                 # recall r, zero false positives
        acc.setdefault(f"recall {r:.2f}, fpr 0.00", []).append(
            score(np.where(flag, DROP, CEILING), noise))
    for p in FPRS[1:]:
        flag = bad | (~bad & (g.random(shape) < p))        # full recall, false-positive rate p
        acc.setdefault(f"recall 1.00, fpr {p:.2f}", []).append(
            score(np.where(flag, DROP, CEILING), noise))
    print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

b = float(np.mean([v[0] for v in acc["no channel"]]))
print(f"\nfixture {FIX.name}: {shape[1]} frames, 2 subjects, {shape[3]} joints, "
      f"4 cameras, {SEEDS} seeds, {shape[0]*shape[1]*shape[2]*shape[3]} observations per arm")
print(f"noise: mixture {CLEAN_PX} px clean / {BAD_FRACTION*100:.0f}% bad at {BAD_PX} px")
print(f"gate: dropped observations set to {DROP}, below minimum_confidence={FLOOR}\n")
print(f"{'arm':>28} {'MPJPE':>9} {'sd':>6} {'buys':>8} {'coverage':>9}")
print("-" * 66)
out = {}
for label, v in acc.items():
    m = float(np.mean([x[0] for x in v])); s = float(np.std([x[0] for x in v]))
    c = float(np.mean([x[1] for x in v]))
    out[label] = {"mpjpe_mm": m, "sd": s, "buys_mm": b - m, "coverage": c}
    mark = "  <-- baseline" if label == "no channel" else ""
    print(f"{label:>28} {m:>6.2f} mm {s:>5.2f} {b-m:>+7.2f} {c*100:>8.1f}%{mark}")
(FIX / "gate-quality-curve.json").write_text(json.dumps(out, indent=1))
print("\n'recall 0.00, fpr 0.00' flags nothing and must land on 'no channel'.")
print("That is the degenerate check: if it does not, the harness is measuring seed noise.")
