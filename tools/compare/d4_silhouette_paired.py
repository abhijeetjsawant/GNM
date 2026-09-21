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

# The pre-card's three arms stay the default so the pre-card reproduces by running this with
# no --arm at all; D4 names its own with --arm NAME=PATH (repeatable) and --pair CAND,REF.
ARMS = {"baseline_D7c_rig": ROOT / "artifacts/compare/i6/delivered-mesh.npz",
        "fitted_MHR": ROOT / "artifacts/compare/d4-body/work-fitted/delivered-mesh.npz",
        "control_MHR_mean_body": ROOT / "artifacts/compare/d4-body/work-mean-body/delivered-mesh.npz"}
DEFAULT_PAIRS = (("fitted_MHR", "baseline_D7c_rig"),
                 ("control_MHR_mean_body", "baseline_D7c_rig"),
                 ("fitted_MHR", "control_MHR_mean_body"))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", default=[],
                    help="NAME=PATH of a delivered-mesh.npz. Replaces the default arm set.")
    ap.add_argument("--pair", action="append", default=[],
                    help="CANDIDATE,REFERENCE to bootstrap. Replaces the default pairs.")
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    arms_in = ({name: Path(path) for name, path in (a.split("=", 1) for a in args.arm)}
               if args.arm else dict(ARMS))
    pairs = ([tuple(p.split(",", 1)) for p in args.pair] if args.pair else list(DEFAULT_PAIRS))
    for cand, ref in pairs:
        missing = [n for n in (cand, ref) if n not in arms_in]
        if missing:
            raise SystemExit(f"--pair names an arm that was not given: {missing}")
    shape = (si.NATIVE[0] // 4, si.NATIVE[1] // 4)
    rig = {c.name: c for c in load_camera_rig(si.RIG_PATH)}
    scaled = {n: c.scaled(*shape) for n, c in rig.items()}
    si.WORK = ROOT / "artifacts/compare/i6"
    masks = si.MaskStore(4, si.CAMERAS)
    per = {name: per_frame_iou(dict(np.load(p)), masks, scaled, shape) for name, p in arms_in.items()}
    report = {"pre_registration": "artifacts/compare/d4-body/PREREGISTRATION.md",
              "arm_files": {n: str(p) for n, p in arms_in.items()},
              "arm_file_sha256": {n: si.sha256(Path(p)) for n, p in arms_in.items()},
              "seed": args.seed, "arms": {}, "paired": {}}
    for name, d in per.items():
        report["arms"][name] = {f"subject_{s:02d}": {
            "pooled_median_iou": round(float(np.nanmedian(np.concatenate([d[(s, c)] for c in si.CAMERAS]))), 4),
            **{c: round(float(np.nanmedian(d[(s, c)])), 4) for c in si.CAMERAS}} for s in (0, 1)}
    for cand, ref in pairs:
        for s in (0, 1):
            rng = np.random.default_rng(args.seed)
            a = np.concatenate([per[cand][(s, c)] for c in si.CAMERAS])
            b = np.concatenate([per[ref][(s, c)] for c in si.CAMERAS])
            boot, n = paired(a, b, rng)
            report["paired"][f"{cand}_minus_{ref}_subject_{s:02d}"] = {**boot, "n": n}
    # Derived, never a literal: each pair's lower CI bound read against zero. The BAND lives in
    # the gate (tools/compare/d4_body_gate.py); this is the same fact the pre-card printed.
    report["lower_ci_above_zero"] = {
        key: bool(value["ci95_of_the_median_difference"][0] > 0)
        for key, value in report["paired"].items()}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["arms"], indent=1)); print(json.dumps(report["paired"], indent=1))


if __name__ == "__main__":
    main()
