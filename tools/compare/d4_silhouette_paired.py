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
LANDMARKS = ROOT / "artifacts/compare/d4-body/delivery"
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
    ap.add_argument("--parts", help="write B1's part-wise and bent-tercile report here instead "
                                    "of the paired bootstrap")
    ap.add_argument("--landmarks", help="the delivery whose captured landmarks define the part "
                                        "partition and the bent terciles")
    args = ap.parse_args()
    arms_in = ({name: Path(path) for name, path in (a.split("=", 1) for a in args.arm)}
               if args.arm else dict(ARMS))
    if args.landmarks:
        global LANDMARKS       # noqa: PLW0603
        LANDMARKS = Path(args.landmarks)
    if args.parts:
        parts_and_terciles(arms_in, Path(args.parts))
        return
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




# ----------------------------------------------------------------- D4 B1's reported diagnostics
# The card reports part-wise IoU with precision and recall beside it, and bent-tercile cells.
# The SAM2 masks carry no part labels, so a part is a REGION OF THE IMAGE, defined from the
# CAPTURED landmarks -- the same array both arms were built from, so the partition is identical
# for every arm and the denominator is shared. Every foreground pixel (in the mask or in any
# arm's render) is assigned to the segment of the projected 19-joint skeleton it is nearest to,
# and each segment belongs to `arms` or to `torso_legs`. No radius, no free parameter: the
# partition is exhaustive and disjoint by construction.
NAMES19 = ("nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
           "left_elbow", "left_wrist", "root", "right_hip", "right_knee", "right_ankle",
           "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear")
ARM_SEGMENTS = (("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"))
TORSO_LEG_SEGMENTS = (("neck", "left_shoulder"), ("neck", "right_shoulder"), ("neck", "root"),
                      ("root", "left_hip"), ("root", "right_hip"),
                      ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                      ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
                      ("nose", "neck"))


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    denom = float(d @ d)
    if denom < 1e-12:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip((points - a) @ d / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * d), axis=1)


def part_labels(landmarks: np.ndarray, camera, pixels: np.ndarray) -> np.ndarray:
    """`pixels` as (N, 2) float (x, y). Returns 0 = torso+legs, 1 = arms, -1 = undecidable."""
    uv, depth = camera.project(np.asarray(landmarks, np.float64))
    best = np.full(len(pixels), np.inf)
    label = np.full(len(pixels), -1, np.int8)
    for part, segments in ((0, TORSO_LEG_SEGMENTS), (1, ARM_SEGMENTS)):
        for first, second in segments:
            i, j = NAMES19.index(first), NAMES19.index(second)
            if not (np.isfinite(uv[i]).all() and np.isfinite(uv[j]).all()
                    and depth[i] > 1e-6 and depth[j] > 1e-6):
                continue
            distance = _segment_distance(pixels, uv[i], uv[j])
            closer = distance < best
            best[closer] = distance[closer]
            label[closer] = part
    return label


def trunk_lean_deg(landmarks: np.ndarray) -> float:
    """Angle of root->neck from the capture's vertical, in degrees. NaN when either is missing."""
    root, neck = landmarks[NAMES19.index("root")], landmarks[NAMES19.index("neck")]
    if not (np.isfinite(root).all() and np.isfinite(neck).all()):
        return float("nan")
    axis = neck - root
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(axis[2] / norm, -1.0, 1.0))))


def parts_and_terciles(arms_in: dict, out: Path) -> None:
    """B1's REPORTED diagnostics: per-part IoU/precision/recall and bent-tercile cells.

    Every arm is scored on the SAME image partition (built from the captured landmarks both arms
    consumed) and on the SAME frames, so the populations are identical across arms.
    """
    shape = (si.NATIVE[0] // 4, si.NATIVE[1] // 4)
    rig = {c.name: c for c in load_camera_rig(si.RIG_PATH)}
    scaled = {n: c.scaled(*shape) for n, c in rig.items()}
    si.WORK = ROOT / "artifacts/compare/i6"
    masks = si.MaskStore(4, si.CAMERAS)
    meshes = {name: dict(np.load(path)) for name, path in arms_in.items()}
    landmarks = {s: np.load(LANDMARKS / f"subject-{s:02d}.body-track.npz",
                            allow_pickle=True)["triangulated_world_positions_z_up_m"]
                 for s in (0, 1)}
    lean = {s: np.asarray([trunk_lean_deg(landmarks[s][f]) for f in range(si.FRAMES)])
            for s in (0, 1)}
    terciles = {}
    for s in (0, 1):
        finite = lean[s][np.isfinite(lean[s])]
        low, high = np.percentile(finite, [100 / 3, 200 / 3])
        terciles[s] = (float(low), float(high))
    totals = {name: {(s, part, cam): np.zeros(3) for s in (0, 1) for part in (0, 1)
                     for cam in si.CAMERAS} for name in arms_in}
    tercile_iou = {name: {(s, t, cam): [] for s in (0, 1) for t in (0, 1, 2)
                          for cam in si.CAMERAS} for name in arms_in}
    for s in (0, 1):
        for cam in si.CAMERAS:
            mask = masks.get(cam, int(TRACKLET[cam][f"subject_{s:02d}"]))
            for f in range(si.FRAMES):
                if np.count_nonzero(mask[f]) < si.MIN_MASK_PX:
                    continue
                renders = {}
                for name in arms_in:
                    verts, faces = meshes[name][f"verts_{s:02d}"], meshes[name][f"faces_{s:02d}"]
                    if f >= verts.shape[0]:
                        continue
                    renders[name] = si.rasterise(verts[f], faces, scaled[cam], shape)
                union = mask[f].copy()
                for render in renders.values():
                    union |= render
                ys, xs = np.nonzero(union)
                if not len(xs):
                    continue
                label = part_labels(landmarks[s][f], scaled[cam],
                                    np.stack([xs, ys], axis=1).astype(np.float64))
                value = lean[s][f]
                tier = (0 if value <= terciles[s][0] else 1 if value <= terciles[s][1] else 2) \
                    if np.isfinite(value) else None
                for name, render in renders.items():
                    r = render[ys, xs]
                    m = mask[f][ys, xs]
                    for part in (0, 1):
                        take = label == part
                        inter = float(np.count_nonzero(r & m & take))
                        totals[name][(s, part, cam)] += (inter,
                                                         float(np.count_nonzero(r & take)),
                                                         float(np.count_nonzero(m & take)))
                    if tier is not None:
                        totals_i = float(np.count_nonzero(r & m))
                        union_i = float(np.count_nonzero(r | m))
                        tercile_iou[name][(s, tier, cam)].append(
                            totals_i / union_i if union_i else np.nan)
            print(f"  parts: subject {s} {cam} done", flush=True)
    report = {"partition": "every foreground pixel assigned to the nearest projected segment of "
                           "the CAPTURED 19-joint skeleton; arms = shoulder-elbow-wrist both "
                           "sides, torso_legs = the rest. Identical for every arm.",
              "landmark_source": str(LANDMARKS),
              "tercile_edges_trunk_lean_deg": {f"subject_{s:02d}": terciles[s] for s in (0, 1)},
              "parts": {}, "bent_terciles": {}}
    for name in arms_in:
        report["parts"][name] = {}
        for s in (0, 1):
            for part, label in ((0, "torso_legs"), (1, "arms")):
                inter = sum(totals[name][(s, part, cam)][0] for cam in si.CAMERAS)
                rendered = sum(totals[name][(s, part, cam)][1] for cam in si.CAMERAS)
                masked = sum(totals[name][(s, part, cam)][2] for cam in si.CAMERAS)
                union = rendered + masked - inter
                report["parts"][name][f"subject_{s:02d}_{label}"] = {
                    "iou": round(inter / union, 4) if union else None,
                    "precision": round(inter / rendered, 4) if rendered else None,
                    "recall": round(inter / masked, 4) if masked else None,
                    "pixels_rendered": int(rendered), "pixels_in_mask": int(masked)}
        report["bent_terciles"][name] = {
            f"subject_{s:02d}_tercile_{t}": round(float(np.nanmedian(np.concatenate(
                [tercile_iou[name][(s, t, cam)] for cam in si.CAMERAS]))), 4)
            for s in (0, 1) for t in (0, 1, 2)}
    Path(out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["parts"], indent=1))
    print(json.dumps(report["bent_terciles"], indent=1))


if __name__ == "__main__":
    main()
