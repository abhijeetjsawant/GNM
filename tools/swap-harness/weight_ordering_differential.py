#!/usr/bin/env python3
"""Does weight_before_loss change anything at all?

Minimal differential: one noise realisation, one confidence field, the flag
flipped. Everything else byte-identical. Plus a branch counter, so a null result
can be told apart from a branch that never executed -- which is the whole
question.
"""
import json, sys
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

cameras = tuple(c.scaled(1280, 720) for c in cm.load_camera_rig(FIX / "camera-rig.json"))
base = [[json.loads(l) for l in (FIX / "work" / f"{n}-observations.jsonl").read_text().splitlines() if l.strip()]
        for n in CAMERAS]
subjects = np.load(FIX / "truth.npz", allow_pickle=True)["subjects_soma77_m"]
names = [n for n in cm.JOINT_INDEX if n in SOMA77_TO_AUTOANIM]
shape = (len(CAMERAS), len(base[0]), subjects.shape[0], len(names))

g = np.random.default_rng(20260829)
bad = g.random(shape) < BAD_FRACTION
sigma = np.where(bad, BAD_PX, CLEAN_PX)
noise = g.normal(0.0, 1.0, shape + (2,)) * sigma[..., None]
weight = 1.0 / (1.0 + (sigma / CLEAN_PX) ** 2)
informed = np.clip(CEILING * weight / float(weight.max()), FLOOR + 1e-6, CEILING)

print(f"confidence field: min {informed.min():.6f} max {informed.max():.6f} "
      f"unique {len(np.unique(informed))}")
print(f"residual weights sqrt(c): {np.sqrt(informed.min()):.4f} .. {np.sqrt(informed.max()):.4f} "
      f"-> expressible ratio {np.sqrt(informed.max()/informed.min()):.3f}x")
print(f"observations below the {FLOOR} gate: {(informed < FLOOR).sum()} of {informed.size}\n")

def records(conf):
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

# Branch counter: wrap solve_sequence_positions to record the flag it actually receives.
seen = []
original = cm.solve_sequence_positions
def spy(*a, **k):
    seen.append(k.get("weight_before_loss", "NOT-PASSED"))
    return original(*a, **k)
cm.solve_sequence_positions = spy

recs = records(informed)
out = {}
for flag in (False, True):
    seen.clear()
    _, _, positions, _ = cm.reconstruct_multiview(
        cameras, recs, subject_count=2, sample_rate_hz=30,
        minimum_confidence=FLOOR, weight_before_loss=flag,
    )
    out[flag] = positions
    print(f"flag={flag!s:<5} solver received: {sorted(set(map(str, seen)))} "
          f"({len(seen)} calls)")

a, b = out[False], out[True]
finite = np.isfinite(a) & np.isfinite(b)
delta = np.abs(a[finite] - b[finite])
print(f"\nidentical arrays: {np.array_equal(a[finite], b[finite])}")
print(f"max |delta|: {delta.max()*1000:.9f} mm   mean {delta.mean()*1000:.9f} mm")
print(f"nan pattern identical: {np.array_equal(np.isnan(a), np.isnan(b))}")
