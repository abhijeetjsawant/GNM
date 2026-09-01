#!/usr/bin/env python3
"""Does native detector width fix the head landmarks? Pre-registered in §6s.

§6q measured the input limit -- the head is 30.6 px at width 1280 -- and §6s pre-registered
this run with its own falsification: if within-head scatter does NOT fall, the thirty-pixel
argument is withdrawn.

Measures the INPUT only. No gate, no reference. Within-head distances are constant by
construction, so their standard deviation is pure measurement error.

Both arms go through the REAL pipeline: `associate.recover()` wraps `reconstruct_multiview`
with a recording associator, and `triangulate_soma.triangulate` uses `triangulate_point` at
production settings. Nothing here is a reimplementation -- the module globals are repointed
at the native artifacts and the same functions run. CLAUDE.md's rule, and this session has
already broken it once.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import associate  # noqa: E402
import triangulate_soma  # noqa: E402

NAMES = ("Head", "HeadEnd", "Jaw", "LeftEye", "RightEye")
IDX = [6, 7, 8, 9, 10]
# (rig, work, observation-file suffix, TRACKS) -- TRACKS matters: associate.recover()
# gates the replay against the retained track at 0.0 mm, so pointing it at the wrong
# width's tracks correctly refuses to run. That gate fired on the first attempt here.
ARMS = {
    "width_1280": (Path("artifacts/soma77-full/camera-rig.json"),
                   Path("artifacts/soma77-full/work"), "-observations.jsonl",
                   Path("artifacts/commercial-multiview-soma77")),
    "width_3840": (Path("artifacts/commercial-multiview-native/camera-rig.json"),
                   Path("artifacts/commercial-multiview-native/work"), "-soma77-observations.jsonl",
                   Path("artifacts/commercial-multiview-native")),
}


def sigma_for(rig: Path, work: Path, suffix: str, tracks: Path, out: Path) -> dict:
    associate.RIG, associate.WORK, associate.OUT = rig, work, out
    associate.TRACKS = tracks
    # the observation filename differs between the two artifact layouts
    original = associate.load

    def load():
        from autoanim_gnm.commercial_multiview import load_camera_rig, load_observation_jsonl
        r = {c.name: c for c in load_camera_rig(associate.RIG)}
        obs = [load_observation_jsonl(associate.WORK / f"{n}{suffix}") for n in associate.CAMERAS]
        w, h = obs[0][0]["width"], obs[0][0]["height"]
        return [r[n].scaled(w, h) for n in associate.CAMERAS], obs

    associate.load = load
    triangulate_soma.load = load
    triangulate_soma.OUT = out
    out.mkdir(parents=True, exist_ok=True)
    assignment, used, report = associate.recover()
    np.savez(out / "association.npz", assignment=assignment, used=used)
    positions, support, used_arr = triangulate_soma.triangulate(IDX)
    associate.load = original

    result = {"association": {k: report[k] for k in list(report)[:4]}}
    for subject in range(positions.shape[0]):
        pos = positions[subject]
        common = np.isfinite(pos).all(axis=(1, 2)) & used_arr[:, subject]
        sig = {}
        for i, a in enumerate(NAMES):
            sds = [float((np.linalg.norm(pos[common, i] - pos[common, j], axis=1) * 1000.0).std())
                   for j in range(len(NAMES)) if j != i]
            sig[a] = round(float(np.median(sds)), 1)
        result[f"subject_{subject:02d}"] = {"frames": int(common.sum()), "sigma_mm": sig,
                                            "median_sigma_mm": round(float(np.median(list(sig.values()))), 1)}
    return result


def main() -> None:
    out = {}
    for arm, (rig, work, suffix, tracks) in ARMS.items():
        print(f"--- {arm} ---")
        out[arm] = sigma_for(rig, work, suffix, tracks, Path(f"artifacts/head-lane/native-{arm}"))
        for s in ("subject_00", "subject_01"):
            b = out[arm][s]
            print(f"  {s}: {b['frames']:3d} frames  median sigma {b['median_sigma_mm']:6.1f} mm   {b['sigma_mm']}")
    print()
    print(f"{'':12s} {'subject 0':>12s} {'subject 1':>12s}")
    for arm in ARMS:
        print(f"{arm:12s} {out[arm]['subject_00']['median_sigma_mm']:11.1f}mm "
              f"{out[arm]['subject_01']['median_sigma_mm']:11.1f}mm")
    for s in ("subject_00", "subject_01"):
        a = out["width_1280"][s]["median_sigma_mm"]; b = out["width_3840"][s]["median_sigma_mm"]
        print(f"{s}: ratio 1280/3840 = {a/b:.2f}x  (pre-registered expectation ~3x if depth-limited)")
    Path("artifacts/head-lane/native-width-sigma.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
