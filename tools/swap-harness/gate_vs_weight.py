#!/usr/bin/env python3
"""Gate, or weight? Five seeds, paired, with the control that can fail.

The published sigma and visibility arms differ in ONE thing that is not the
channel: whether the bad observations land below `minimum_confidence` and are
dropped from `triangulate_point`, or above it where they can only re-weight.
Same information, two mechanisms. This holds the information fixed and moves
only the gate.

Controls, because no gate a constant can pass:
* SHUFFLED below gate -- drops the same NUMBER of observations, chosen at
  random. If this scores like the informed drop, the effect is dropping, not
  knowing what to drop.
* `recovered` is reported for every arm. It is the fraction of slots where the
  least-squares solution is actually used; `solve_sequence_positions` returns
  triangulation everywhere else, so it bounds what any weight can reach.
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
FLOOR, CEILING = 0.25, 0.99
CLEAN_PX, BAD_PX, BAD_FRACTION = 2.75, 36.0, 0.07
FIX = ROOT / "artifacts/synthetic-truth"
SEEDS = 5

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

rec_seen = []
original = cm.solve_sequence_positions
def spy(*a, **k):
    out, rec = original(*a, **k)
    rec_seen.append(float(rec.mean()))
    return out, rec
cm.solve_sequence_positions = spy

def score(conf, noise):
    rec_seen.clear()
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
    # Fixed denominator: unrecovered slots are failures, not absences.
    return (float(fin.mean()), float(np.mean(rec_seen)),
            float(len(fin) / len(err)), int((conf < FLOOR).sum()))

ORDER = ["no channel",
         "informed drop, bad=0.125  BELOW gate",
         "informed weight, bad=0.2501 ABOVE gate",
         "sigma, published arm (above gate)",
         "SHUFFLED drop, below gate (control)"]
acc = {k: [] for k in ORDER}
t0 = time.time()
for seed in range(SEEDS):
    g = np.random.default_rng(20260829 + seed)
    bad = g.random(shape) < BAD_FRACTION
    sigma = np.where(bad, BAD_PX, CLEAN_PX)
    noise = g.normal(0.0, 1.0, shape + (2,)) * sigma[..., None]
    w = 1.0 / (1.0 + (sigma / CLEAN_PX) ** 2)
    informed = np.clip(CEILING * w / float(w.max()), FLOOR + 1e-6, CEILING)
    shuffled = g.permutation(bad.ravel()).reshape(shape)
    arms = {
        ORDER[0]: np.full(shape, CEILING),
        ORDER[1]: np.where(bad, FLOOR * 0.5, CEILING),
        ORDER[2]: np.where(bad, FLOOR + 1e-4, CEILING),
        ORDER[3]: informed,
        ORDER[4]: np.where(shuffled, FLOOR * 0.5, CEILING),
    }
    for label in ORDER:
        acc[label].append(score(arms[label], noise))
    print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

b = float(np.mean([v[0] for v in acc[ORDER[0]]]))
print(f"\nfixture {FIX.name}: {shape[1]} frames, 2 subjects, {shape[3]} joints, "
      f"4 cameras, {SEEDS} seeds")
print(f"noise: mixture {CLEAN_PX} px clean / {BAD_FRACTION*100:.0f}% bad at {BAD_PX} px\n")
print(f"{'arm':>40} {'MPJPE':>9} {'sd':>6} {'buys':>8} {'dropped':>8} {'recov':>7} {'cover':>7}")
print("-" * 92)
out = {}
for label in ORDER:
    v = acc[label]
    m = float(np.mean([x[0] for x in v])); s = float(np.std([x[0] for x in v]))
    r = float(np.mean([x[1] for x in v])); c = float(np.mean([x[2] for x in v]))
    d = int(np.mean([x[3] for x in v]))
    out[label] = {"mpjpe_mm": m, "sd": s, "buys_mm": b - m, "recovered_fraction": r,
                  "coverage": c, "observations_dropped": d}
    print(f"{label:>40} {m:>6.2f} mm {s:>5.2f} {b-m:>+7.2f} {d:>8} {r*100:>6.2f}% {c*100:>6.1f}%")
print(f"\ntotal observations per arm: {shape[0]*shape[1]*shape[2]*shape[3]}")
(FIX / "gate-vs-weight.json").write_text(json.dumps(out, indent=1))
