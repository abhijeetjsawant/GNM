"""D4 pre-card: the fitted MHR body against the D7c rig delivery on the SAME masks, per frame,
paired, with the moving-block bootstrap the lane uses. Reads silhouette.py's own rasteriser,
scorer and mask cache; scores nothing the instrument would not.

    .venv/bin/python tools/compare/d4_silhouette_paired.py --out artifacts/compare/d4-body/paired.json

Arms (each a delivered-mesh.npz written by Blender from a GLB, the capture's Z-up world):
  baseline   artifacts/compare/i6/delivered-mesh.npz            the D7c rig delivery
  fitted     artifacts/compare/d4-body/work-fitted/...          MHR fitted, pinned offsets
  mean_body  artifacts/compare/d4-body/work-mean-body/...       MHR mean body, pose tracked (control)
Populations: whole take, per performer, the four cameras pooled (600 frame-camera cells) and per camera.
The tracklet map is the committed instrument's (identity resolved by MAMMA joint containment, never IoU).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/compare"))
import silhouette as si  # noqa: E402
load_camera_rig = si.load_camera_rig

ARMS = {"baseline_D7c_rig": ROOT / "artifacts/compare/i6/delivered-mesh.npz",
        "fitted_MHR": ROOT / "artifacts/compare/d4-body/work-fitted/delivered-mesh.npz",
        "control_MHR_mean_body": ROOT / "artifacts/compare/d4-body/work-mean-body/delivered-mesh.npz"}
TRACKLET = json.load(open(ROOT / "artifacts/compare/i6/silhouette-1920x1080-A001.json"))["identity"]["our_subject_to_mask_tracklet"]


def per_frame_iou(mesh, masks, scaled, shape):
    out = {}
    for s in (0, 1):
        verts, faces = mesh[f"verts_{s:02d}"], mesh[f"faces_{s:02d}"]
        for cam in si.CAMERAS:
            m = masks.get(cam, int(TRACKLET[cam][f"subject_{s:02d}"]))
            ious = np.full(si.FRAMES, np.nan)
            for f in range(min(si.FRAMES, verts.shape[0])):
                if np.count_nonzero(m[f]) < si.MIN_MASK_PX:
                    continue
                ious[f] = si.score(si.rasterise(verts[f], faces, scaled[cam], shape), m[f])[2]
            out[(s, cam)] = ious
    return out


def paired(a, b, rng):
    keep = np.isfinite(a) & np.isfinite(b)
    return si.moving_block_bootstrap(a[keep], b[keep], rng), int(keep.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    shape = (si.NATIVE[0] // 4, si.NATIVE[1] // 4)
    rig = {c.name: c for c in load_camera_rig(si.RIG_PATH)}
    scaled = {n: c.scaled(*shape) for n, c in rig.items()}
    si.WORK = ROOT / "artifacts/compare/i6"
    masks = si.MaskStore(4, si.CAMERAS)
    per = {name: per_frame_iou(dict(np.load(p)), masks, scaled, shape) for name, p in ARMS.items()}
    report = {"pre_registration": "artifacts/compare/d4-body/PREREGISTRATION.md", "arms": {}, "paired": {}}
    for name, d in per.items():
        report["arms"][name] = {f"subject_{s:02d}": {
            "pooled_median_iou": round(float(np.nanmedian(np.concatenate([d[(s, c)] for c in si.CAMERAS]))), 4),
            **{c: round(float(np.nanmedian(d[(s, c)])), 4) for c in si.CAMERAS}} for s in (0, 1)}
    for cand in ("fitted_MHR", "control_MHR_mean_body"):
        for s in (0, 1):
            rng = np.random.default_rng(20260921)
            a = np.concatenate([per[cand][(s, c)] for c in si.CAMERAS])
            b = np.concatenate([per["baseline_D7c_rig"][(s, c)] for c in si.CAMERAS])
            boot, n = paired(a, b, rng)
            rng2 = np.random.default_rng(20260921)
            a2 = np.concatenate([per["fitted_MHR"][(s, c)] for c in si.CAMERAS])
            b2 = np.concatenate([per["control_MHR_mean_body"][(s, c)] for c in si.CAMERAS])
            boot2, _ = paired(a2, b2, rng2)
            report["paired"][f"{cand}_minus_baseline_subject_{s:02d}"] = {**boot, "n": n}
            report["paired"][f"fitted_minus_mean_body_subject_{s:02d}"] = boot2
    verdict = {}
    for s in (0, 1):
        r = report["paired"][f"fitted_MHR_minus_baseline_subject_{s:02d}"]
        verdict[f"subject_{s:02d}"] = "PASS" if (r["ci95_of_the_median_difference"][0] > 0) else "FAIL"
    report["verdict"] = {"rule": "fitted beats the D7c rig on BOTH performers, paired moving-block CI clear of zero",
                         **verdict, "route": "PROCEED" if all(v == "PASS" for v in verdict.values()) else "WRONG, not unfinished"}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["arms"], indent=1)); print(json.dumps(report["paired"], indent=1)); print(report["verdict"])


if __name__ == "__main__":
    main()
