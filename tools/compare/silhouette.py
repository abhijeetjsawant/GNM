#!/usr/bin/env python3
"""THE SURFACE INSTRUMENT (ladder step I6). Our delivered mesh against MAMMA's masks.

Every other rung of the substitution ladder scores a *model* against a *model*: our
joints against MAMMA's `pred_joints`, our triangulation against its `triangulated_3d_pts`.
Both sides are fits, neither is truth, and a change that improves agreement may be
moving toward the reference's error. `ma_masks` is the one retained MAMMA artifact that
is **not model-mediated**: SAM2 segmented the actual pixels of the actual footage. A
silhouette scored against it is the closest thing this fixture has to an observation.

    .venv/bin/python tools/compare/silhouette.py
    .venv/bin/python tools/compare/silhouette.py --ids-only     # id resolution, then stop
    .venv/bin/python tools/compare/silhouette.py --scale 2      # 1920x1080 fidelity check

It runs `tools/compare/blender_export_mesh.py` itself when the posed-mesh cache is
missing or older than the GLBs (Blender's Python has no cv2 or scipy, so the render
path and the scoring path cannot share a process).

WHAT THIS IS BLIND TO, and it is on the report as `blind_to`: depth (a silhouette is a
projection; the subject can be nearer or farther along the ray and still fill the same
outline -- only the four cameras disagreeing constrains it); left/right mirroring of a
fore-aft symmetric pose; and *everything inside the outline*, which is most of a body.
It cannot separate a shape error from a pose error. The front/back frame list exists
because the one thing it is NOT blind to, on the frames where the body is fore-aft
asymmetric in a camera, is a 180-degree facing error -- which is what step D1 fixes.

THREE ID SPACES, AND NONE OF THEM IS RESOLVED BY IoU.
  * SAM2 mask tracklet id -- the `_01` / `_02` suffix on `mask_NNNN_XX.png`, stable
    within a camera and meaningless across cameras.
  * MAMMA `body_id` -- `verts_joints_body_id-00.npz` / `-01.npz`.
  * our subject index -- `subject-00.glb` / `subject-01.glb`.
Tracklet <-> body_id is resolved by projecting MAMMA's own `pred_joints` into each
camera and counting which tracklet's mask contains them. body_id <-> our subject is
resolved by 3D pelvis agreement (`tools/head/subject_map.py`). Resolving either by
silhouette overlap would make the instrument score the assumption it was built on.

CONTROLS. `precision` and `recall` are reported separately and never collapsed to IoU
alone, because the degenerate a mesh instrument actually produces -- something too big
-- buys recall with precision. The dilation sweep makes that trade a curve, not a point.

`np.load(..., allow_pickle=True)` is required by the SMPL-X model file and MAMMA's own
npz outputs (both hold object arrays; CLAUDE.md). Every such file is a retained local
artifact of this project, not third-party input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))
from autoanim_gnm.commercial_multiview import JOINT_INDEX, load_camera_rig  # noqa: E402
from subject_map import mamma_index_for  # noqa: E402

FIXTURE = "pushing_and_lifting_from_ground"
MAMMA_OUT = ROOT / "artifacts/mamma/mamma-4cam-five-second-v2/output"
MASKS = MAMMA_OUT / "ma_masks/ma_cap" / FIXTURE
MA3D = MAMMA_OUT / "ma_3d" / FIXTURE
RIG_PATH = ROOT / "artifacts/soma77-full/camera-rig.json"
SMPLX = ROOT / ".cache/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz"
DELIVERY = ROOT / "artifacts/commercial-multiview-soma77"
WORK = ROOT / "artifacts/compare/i6"
REPORT = ROOT / "artifacts/compare/silhouette.json"
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"

CAMERAS = ("A001", "B001", "C001", "D001")
TRACKLETS = (1, 2)
FRAMES = 150
NATIVE = (3840, 2160)
# The take's frames [60, 210) are MAMMA's 150; every index in this file is 0..149 of
# that window, on every side. Verified: the delivered GLB's action spans 0..149 once
# the scene fps is 30 (see blender_export_mesh.py) and ma_masks holds 150 x 2 PNGs.
FRAME_WINDOW = (60, 210)

# id-resolution probes: MAMMA `pred_joints` indices (SMPL-X body joints)
PROBE_JOINTS = {"pelvis": 0, "left_hip": 1, "right_hip": 2, "neck": 12, "head": 15,
                "left_shoulder": 16, "right_shoulder": 17, "left_knee": 4, "right_knee": 5}
PELVIS = 0
# body frame for the camera-independent fore-aft asymmetry figure
NECK, LHIP, RHIP = 12, 1, 2
ROOT_JOINT = JOINT_INDEX["root"]   # our 19-joint contract, for the ours-turned-round arm
LIMB_PAIRS = ((20, 21), (18, 19), (4, 5), (7, 8))  # wrists, elbows, knees, ankles

MIN_MASK_PX = 200            # at render resolution; below this the mask is a fragment
DILATION_RADII = (3, 9, 25)  # render px
# Front/back visibility is graded, not binary. The absolute thresholds are a menu for
# step D1 to choose a strictness from; the figures are split at the per-(camera,subject)
# MEDIAN of the same quantity, which needs no threshold and keeps 75 frames on each side
# -- on this take every frame clears even the loosest absolute threshold, so an absolute
# split would put the whole population on one side and prove nothing.
ASYMMETRY_THRESHOLDS = (0.30, 0.50, 0.70)
BLOCK = 15                   # moving-block bootstrap block length (lag-1 autocorr ~0.99)
RESAMPLES = 2000
SEED = 20260902

BASE_ARMS = ("ours_delivered", "ORACLE_mamma_mesh", "control_mean_body",
             "control_frozen_pose_tracked", "control_frozen_pose_static",
             "control_shuffled_subject", "control_oracle_yaw180_facing",
             "control_ours_yaw180_facing")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- rasteriser

def rasterise(verts: np.ndarray, faces: np.ndarray, camera, shape: tuple[int, int]) -> np.ndarray:
    """Binary silhouette of a triangle soup. No z-buffer: a silhouette is a union.

    `shape` is (width, height) and `camera` must already be `.scaled()` to it. Faces
    with any vertex at or behind the principal plane are dropped -- they project to
    garbage. Subpixel coverage comes from cv2's 4-bit `shift`, so a rasterised edge sits
    within 1/16 px of the true projected edge and no figure depends on integer rounding.

    **One `cv2.fillPoly` call per triangle, not one call for all of them.** Handing
    fillPoly an array of N polygons does not paint their union: it scanline-fills the
    region bounded by all N contours together under an even-odd rule, so overlapping
    polygons CANCEL -- the same square passed twice renders as its outline and nothing
    else (verified). A projected body self-overlaps everywhere the front of the torso
    covers the back, so the batched call quietly loses pixels: on a real frame here it
    disagreed with the correct fill on 190 of ~7000 px, small enough to look plausible
    and large enough to matter. `fillConvexPoly` per triangle is correct by construction
    and, on this mesh, 17 ms against 13 ms.
    """
    width, height = shape
    uv, depth = camera.project(np.asarray(verts, dtype=np.float64))
    keep = (depth[faces] > 1e-6).all(axis=1)
    image = np.zeros((height, width), dtype=np.uint8)
    if not keep.any():
        return image.astype(bool)
    # A triangle far outside the frame contributes nothing but could overflow the
    # shifted int32; clamping at 100x the frame keeps every on-screen edge exact.
    tri = np.clip(uv[faces[keep]], -100.0 * width, 100.0 * width)
    for triangle in np.round(tri * 16.0).astype(np.int32):
        cv2.fillConvexPoly(image, triangle, 1, lineType=cv2.LINE_8, shift=4)
    return image.astype(bool)


def score(render: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """precision, recall, IoU. An empty render scores precision 0, not NaN."""
    inter = float(np.count_nonzero(render & mask))
    r = float(np.count_nonzero(render))
    m = float(np.count_nonzero(mask))
    union = r + m - inter
    return (inter / r if r else 0.0, inter / m if m else 0.0,
            inter / union if union else 0.0)


# ------------------------------------------------------------------------------- SMPL-X

class SMPLXMesh:
    """Full SMPL-X forward (shape + pose blendshapes + LBS), enough to make a MESH.

    `tools/compare/smplx_body.py` is joints-only by design. The mean-body CONTROL needs
    vertices, and it needs them to differ from the oracle by *only* the shape term, so
    the control is emitted as `pred_vertices + (forward(betas=0) - forward(betas))`:
    any residual between this forward and MAMMA's own cancels exactly, and what is left
    is the shape displacement and nothing else.

    Instrument only. SMPL-X is MPI research-licensed and never enters a delivered
    artifact (CLAUDE.md); the delivery's `run-report.json` declares `smplx_model: false`.
    """

    def __init__(self, path: Path = SMPLX) -> None:
        data = np.load(path, allow_pickle=True)
        self.v_template = data["v_template"].astype(np.float64)
        self.shapedirs = data["shapedirs"].astype(np.float64)
        posedirs = data["posedirs"].astype(np.float64)
        self.posedirs = posedirs.reshape(posedirs.shape[0] * 3, -1)
        self.weights = data["weights"].astype(np.float64)
        self.j_regressor = data["J_regressor"].astype(np.float64)
        self.faces = data["f"].astype(np.int32)
        parents = data["kintree_table"].astype(np.int64)[0].copy()
        parents[0] = -1
        self.parents = parents
        self.n_joints = self.weights.shape[1]

    def forward(self, betas: np.ndarray, pose: np.ndarray, translation: np.ndarray) -> np.ndarray:
        betas = np.asarray(betas, dtype=np.float64)
        v_shaped = self.v_template + self.shapedirs[:, :, : len(betas)] @ betas
        joints = self.j_regressor @ v_shaped
        rot = Rotation.from_rotvec(np.asarray(pose, dtype=np.float64).reshape(-1, 3)).as_matrix()
        v_posed = v_shaped + (self.posedirs @ (rot[1:] - np.eye(3)).reshape(-1)).reshape(-1, 3)
        globals_ = np.zeros((self.n_joints, 4, 4))
        for j in range(self.n_joints):
            local = np.eye(4)
            local[:3, :3] = rot[j]
            parent = self.parents[j]
            local[:3, 3] = joints[j] - (joints[parent] if parent >= 0 else 0.0)
            globals_[j] = local if parent < 0 else globals_[parent] @ local
        relative = globals_.copy()
        for j in range(self.n_joints):
            relative[j, :3, 3] -= globals_[j, :3, :3] @ joints[j]
        skin = np.einsum("vj,jab->vab", self.weights, relative)
        homogeneous = np.concatenate([v_posed, np.ones((len(v_posed), 1))], axis=1)
        return np.einsum("vab,vb->va", skin, homogeneous)[:, :3] + np.asarray(translation, float)


# --------------------------------------------------------------------------------- data

def delivered_mesh() -> dict[str, np.ndarray]:
    out = WORK / "delivered-mesh.npz"
    glbs = [DELIVERY / f"subject-{s:02d}.glb" for s in (0, 1)]
    if not out.exists() or any(out.stat().st_mtime < g.stat().st_mtime for g in glbs):
        WORK.mkdir(parents=True, exist_ok=True)
        print("exporting the delivered mesh through Blender (fps 30 BEFORE import) ...")
        subprocess.run([BLENDER, "--background", "--python",
                        str(ROOT / "tools/compare/blender_export_mesh.py"), "--",
                        str(out), str(DELIVERY), "30"], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    return dict(np.load(out))


class MaskStore:
    """The SAM2 masks at render resolution, held packed one bit per pixel.

    Downsampled from the native 3840x2160 binary PNGs with INTER_AREA and thresholded at
    0.5 -- a render pixel is foreground when more than half its native footprint was.
    Cached on disk because decoding 1200 8.3 MP PNGs takes minutes, and kept packed in
    memory because 1200 frames of bool at 960x540 is 620 MB.
    """

    def __init__(self, scale: int, cams: tuple[str, ...] = CAMERAS) -> None:
        self.width, self.height = NATIVE[0] // scale, NATIVE[1] // scale
        self.cams = cams
        cache = WORK / f"masks-{self.width}x{self.height}-{'_'.join(cams)}.npz"
        if cache.exists():
            self.packed = np.load(cache)["packed"]
            return
        WORK.mkdir(parents=True, exist_ok=True)
        packed = np.empty((len(cams), len(TRACKLETS), FRAMES,
                           self.height, (self.width + 7) // 8), dtype=np.uint8)
        for c, cam in enumerate(cams):
            print(f"caching masks {cam} at {self.width}x{self.height} ...")
            for t, tracklet in enumerate(TRACKLETS):
                for f in range(FRAMES):
                    png = MASKS / cam / "masks" / f"mask_{f:04d}_{tracklet:02d}.png"
                    native = np.asarray(Image.open(png), dtype=np.uint8)
                    small = cv2.resize(native, (self.width, self.height),
                                       interpolation=cv2.INTER_AREA)
                    packed[c, t, f] = np.packbits(small > 127, axis=-1)
        np.savez_compressed(cache, packed=packed)
        self.packed = packed

    def get(self, camera: str, tracklet: int) -> np.ndarray:
        """[frame, H, W] bool for one camera and one SAM2 tracklet id."""
        c, t = self.cams.index(camera), TRACKLETS.index(tracklet)
        return np.unpackbits(self.packed[c, t], axis=-1, count=self.width).astype(bool)


# ------------------------------------------------------------------------ id resolution

def resolve_tracklets(cameras: dict, pred_joints: dict) -> dict[str, Any]:
    """Which SAM2 tracklet is which MAMMA `body_id`, per camera, with the evidence.

    Containment of MAMMA's own projected joints in the native-resolution masks. Never
    IoU: IoU is the quantity this instrument reports, and resolving identity with it
    would make every figure a restatement of the assumption it was built on.
    """
    evidence: dict[str, Any] = {}
    for cam in CAMERAS:
        counts = np.zeros((2, 2), dtype=int)      # [body_id, tracklet]
        ambiguous = np.zeros(2, dtype=int)        # joints inside BOTH masks: resolve nothing
        per_frame = []
        camera = cameras[cam]
        for f in range(FRAMES):
            native = {t: np.asarray(Image.open(
                MASKS / cam / "masks" / f"mask_{f:04d}_{t:02d}.png"), dtype=np.uint8) > 127
                for t in TRACKLETS}
            row = {}
            for body in (0, 1):
                points = np.stack([pred_joints[body][f, i] for i in PROBE_JOINTS.values()])
                uv, depth = camera.project(points)
                u = np.clip(np.round(uv[:, 0]).astype(int), 0, NATIVE[0] - 1)
                v = np.clip(np.round(uv[:, 1]).astype(int), 0, NATIVE[1] - 1)
                on = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < NATIVE[0]) \
                    & (uv[:, 1] >= 0) & (uv[:, 1] < NATIVE[1])
                inside = {t: on & native[t][v, u] for t in TRACKLETS}
                ambiguous[body] += int(np.count_nonzero(inside[1] & inside[2]))
                only = {t: int(np.count_nonzero(inside[t] & ~inside[3 - t])) for t in TRACKLETS}
                for t in TRACKLETS:
                    counts[body, t - 1] += only[t]
                row[body] = only
            per_frame.append({
                "frame": f,
                "body_00_only_in_tracklet_01_02": [row[0][1], row[0][2]],
                "body_01_only_in_tracklet_01_02": [row[1][1], row[1][2]],
                "mask_overlap_px_native": int(np.count_nonzero(native[1] & native[2])),
            })
        straight, crossed = int(counts[0, 0] + counts[1, 1]), int(counts[0, 1] + counts[1, 0])
        mapping = {0: 1, 1: 2} if straight >= crossed else {0: 2, 1: 1}
        # A frame where a body's joints land mostly in the OTHER body's mask is
        # occlusion, not an id swap -- SAM2 gives the occluding person those pixels.
        # Listed so a constant-per-camera mapping is audited rather than assumed.
        contested = [p["frame"] for p in per_frame
                     if p["body_00_only_in_tracklet_01_02"][mapping[0] - 1]
                     < p["body_00_only_in_tracklet_01_02"][mapping[1] - 1]
                     or p["body_01_only_in_tracklet_01_02"][mapping[1] - 1]
                     < p["body_01_only_in_tracklet_01_02"][mapping[0] - 1]]
        evidence[cam] = {
            "mapping_body_id_to_tracklet": {f"body_id-{b:02d}": f"{t:02d}"
                                            for b, t in mapping.items()},
            "containment_counts_body_by_tracklet": counts.tolist(),
            "probe_joints": list(PROBE_JOINTS),
            "joint_tests_per_body": FRAMES * len(PROBE_JOINTS),
            "straight_total": straight, "crossed_total": crossed,
            "margin_x": round(straight / max(crossed, 1), 1),
            "joints_inside_both_masks": ambiguous.tolist(),
            "contested_frames": contested,
            "contested_frames_reading":
                "on these frames the other subject's SAM2 mask covers the probed joints, "
                "which is occlusion and not an id swap: the subject whose joints move are "
                "the occluded one, the mask pair barely overlaps, and the opposite body "
                "stays 100% in its own tracklet. The mapping is constant per camera and "
                "these frames stay in the scored population, where a truncated mask caps "
                "recall identically for every arm.",
            "per_frame": per_frame,
        }
        print(f"{cam}: body_id-00 -> tracklet {mapping[0]:02d}, "
              f"body_id-01 -> tracklet {mapping[1]:02d}   "
              f"straight {straight} vs crossed {crossed} "
              f"({straight / max(crossed, 1):.0f}x), ambiguous {ambiguous.tolist()}, "
              f"contested frames {len(contested)}")
    return evidence


# ------------------------------------------------------------------------------- meshes

def build_arms(mesh: dict, pred_vertices: dict, pred_joints: dict, smplx_faces: np.ndarray,
               subject_to_body: dict[int, int]) -> dict:
    """Every arm as {subject: (verts[frame or 1], faces)} in the capture's Z-up world."""
    body = SMPLXMesh()
    arms: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    ours = {s: (mesh[f"verts_{s:02d}"].astype(np.float32), mesh[f"faces_{s:02d}"])
            for s in (0, 1)}
    arms["ours_delivered"] = ours
    arms["ORACLE_mamma_mesh"] = {s: (pred_vertices[subject_to_body[s]].astype(np.float32),
                                     smplx_faces) for s in (0, 1)}

    # control: MAMMA's own pose carrying the MEAN body. Differs from the oracle by the
    # shape term alone, so if this does not fail, the reading is that the silhouette at
    # this resolution cannot see body shape -- which is a finding D6 has to know.
    mean_body = {}
    for s in (0, 1):
        b = subject_to_body[s]
        cache = WORK / f"mean-body-{b:02d}.npy"
        if cache.exists():
            mean_body[s] = (np.load(cache), smplx_faces)
            continue
        params = np.load(MA3D / f"smplx_params_body_id-{b:02d}.npz", allow_pickle=True)
        betas = params["smplx_betas"][0].astype(np.float64)
        verts = np.empty_like(pred_vertices[b], dtype=np.float32)
        for f in range(FRAMES):
            pose = params["smplx_pose"][f].astype(np.float64)
            translation = params["smplx_translation"][f].astype(np.float64)
            shift = body.forward(np.zeros_like(betas), pose, translation) \
                - body.forward(betas, pose, translation)
            verts[f] = pred_vertices[b][f] + shift
        WORK.mkdir(parents=True, exist_ok=True)
        np.save(cache, verts)
        mean_body[s] = (verts, smplx_faces)
    arms["control_mean_body"] = mean_body

    # control: one frame's pose on every frame, translated so the mesh centroid still
    # tracks. The static version is trivially defeated by the metres of locomotion in
    # this take and teaches nothing; the tracked one is the degenerate a pose solver
    # actually produces. Both are reported.
    frozen, frozen_static = {}, {}
    for s in (0, 1):
        verts, faces = ours[s]
        centroids = verts.mean(axis=1)
        frozen[s] = ((verts[0][None] - centroids[0] + centroids[:, None, :]).astype(np.float32),
                     faces)
        frozen_static[s] = (verts[:1].copy(), faces)
    arms["control_frozen_pose_tracked"] = frozen
    arms["control_frozen_pose_static"] = frozen_static

    # control: our own mesh scored against the OTHER subject's mask.
    arms["control_shuffled_subject"] = {s: ours[1 - s] for s in (0, 1)}

    # control AND probe: the oracle turned 180 degrees about its own pelvis vertical.
    # Scored against the masks it is the self-consistent-but-backwards human that step
    # D1's gate must reject; compared against the oracle's own silhouette it says which
    # frames can see a facing error at all.
    yaw = {}
    for s in (0, 1):
        b = subject_to_body[s]
        verts = pred_vertices[b].astype(np.float64)
        pelvis = pred_joints[b][:, PELVIS]
        turned = verts - pelvis[:, None, :]
        turned = np.stack([-turned[..., 0], -turned[..., 1], turned[..., 2]], axis=-1)
        yaw[s] = ((turned + pelvis[:, None, :]).astype(np.float32), smplx_faces)
    arms["control_oracle_yaw180_facing"] = yaw

    # The same turn applied to OUR delivered mesh, about our own triangulated pelvis.
    # This is step D1's question asked of the pixels: if the delivery ships a 180-degree
    # facing error, turning it round must IMPROVE its agreement with the footage. It is a
    # control in the sense that one of the two orientations has to lose.
    ours_yaw = {}
    for s in (0, 1):
        verts, faces = ours[s]
        root = np.load(DELIVERY / f"subject-{s:02d}.body-track.npz")[
            "triangulated_world_positions_z_up_m"][:, ROOT_JOINT].astype(np.float64)
        # a frame with no triangulated root falls back to the mesh centroid
        centroid = verts.mean(axis=1).astype(np.float64)
        bad = ~np.isfinite(root).all(axis=1)
        root[bad] = centroid[bad]
        turned = verts.astype(np.float64) - root[:, None, :]
        turned = np.stack([-turned[..., 0], -turned[..., 1], turned[..., 2]], axis=-1)
        ours_yaw[s] = ((turned + root[:, None, :]).astype(np.float32), faces)
    arms["control_ours_yaw180_facing"] = ours_yaw
    return arms


def frame_alignment_check(ours: dict, masks: "MaskStore", subject_to_tracklet: dict,
                          scaled: dict, shape: tuple[int, int], cams: tuple[str, ...],
                          stride: int = 5, lags: tuple[int, ...] = (-2, -1, 0, 1, 2),
                          tolerance: float = 0.01) -> dict:
    """Does our frame f line up with mask frame f? Verification, never selection.

    The oracle arm is MAMMA against MAMMA and shares an index by construction; OURS is a
    separate Blender export whose only alignment evidence is an action range. A one-frame
    offset would cost a few points of IoU and be charged to the pipeline.

    Two things this got wrong first, both worth keeping written down. The lag profile is
    *flat near its peak* -- at 30 fps a body moves a few pixels per frame -- so the
    verdict allows a relative `tolerance` rather than demanding lag 0 win outright, and
    the margin is on the report either way. And the first version scored each lag on
    whichever frames had a partner in range, which put lag -2 and lag 0 on DIFFERENT
    populations near the ends of the take; with a moving subject that alone flipped the
    winner and read as a timebase defect. Every lag is now scored on one frame set. On
    the corrected check lag 0 leads by 1.2 % over four cameras at 960x540 and by 2.6 %
    on A001 alone at 1920x1080.
    """
    # Same denominator across lags: only frames whose shifted partner exists for EVERY
    # lag are scored, or a lag near the ends is judged on a different population.
    span = max(abs(lag) for lag in lags)
    frames = list(range(span, FRAMES - span, stride))
    collected: dict[int, list[float]] = {lag: [] for lag in lags}
    for cam in cams:
        for s in (0, 1):
            mask = masks.get(cam, subject_to_tracklet[cam][s])
            verts, faces = ours[s]
            for f in frames:
                render = rasterise(verts[f], faces, scaled[cam], shape)
                for lag in lags:
                    collected[lag].append(score(render, mask[f + lag])[2])
    profile = {str(lag): float(np.median(values)) for lag, values in collected.items()}
    best = max(profile, key=profile.get)
    shortfall = (profile[best] - profile["0"]) / max(profile[best], 1e-9)
    return {"median_iou_by_lag_frames": profile, "best_lag": int(best),
            "lag0_relative_shortfall": shortfall, "tolerance": tolerance,
            "verdict": "aligned" if shortfall <= tolerance else "MISALIGNED",
            "note": "our exported frame f scored against mask frame f+lag, every "
                    f"{stride}th frame, every camera in the run and both subjects; lag 0 "
                    f"must lead or trail the best lag by at most {tolerance:.0%}"}


def arm_frame(entry: tuple[np.ndarray, np.ndarray], f: int) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = entry
    return verts[f if len(verts) > 1 else 0], faces


# ------------------------------------------------------------------------------ scoring

def moving_block_bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """Median difference a-b with a moving-block CI, both arms on IDENTICAL draws.

    150 frames whose per-frame agreement has lag-1 autocorrelation ~0.99 are nothing
    like 150 independent samples; ordinary resampling would quote an interval several
    times too tight (CLAUDE.md).
    """
    n = len(a)
    if n < BLOCK:
        return {"median_difference": float(np.median(a - b)), "ci95": None, "blocks": 0}
    starts = n - BLOCK + 1
    per = int(np.ceil(n / BLOCK))
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        idx = (rng.integers(0, starts, size=per)[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n]
        draws[i] = np.median(a[idx]) - np.median(b[idx])
    return {"median_difference": float(np.median(a) - np.median(b)),
            "ci95_of_the_median_difference": [float(np.percentile(draws, 2.5)),
                                              float(np.percentile(draws, 97.5))],
            "block_length": BLOCK, "resamples": RESAMPLES,
            "lag1_autocorrelation": float(np.corrcoef(a[:-1], a[1:])[0, 1])}


def summary(rows: np.ndarray) -> dict:
    return {"precision": {"median": float(np.median(rows[:, 0])),
                          "p05": float(np.percentile(rows[:, 0], 5))},
            "recall": {"median": float(np.median(rows[:, 1])),
                       "p05": float(np.percentile(rows[:, 1], 5))},
            "iou": {"median": float(np.median(rows[:, 2])),
                    "p05": float(np.percentile(rows[:, 2], 5))}}


# --------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=4,
                    help="native/scale is the render resolution (4 -> 960x540)")
    ap.add_argument("--ids-only", action="store_true")
    ap.add_argument("--cameras", default=",".join(CAMERAS))
    ap.add_argument("--skip-point-to-surface", action="store_true")
    ap.add_argument("--out", default=str(REPORT))
    args = ap.parse_args()
    cams = tuple(c for c in args.cameras.split(",") if c)
    width, height = NATIVE[0] // args.scale, NATIVE[1] // args.scale
    shape = (width, height)

    rig = {c.name: c for c in load_camera_rig(RIG_PATH)}
    scaled = {name: cam.scaled(width, height) for name, cam in rig.items()}
    pred_joints = {b: np.load(MA3D / f"verts_joints_body_id-{b:02d}.npz",
                              allow_pickle=True)["pred_joints"].astype(np.float64)
                   for b in (0, 1)}

    print("=== id resolution 1/2: SAM2 tracklet <-> MAMMA body_id (projected joints, never IoU) ===")
    tracklet_evidence = resolve_tracklets(rig, pred_joints)

    ours_positions = np.stack([
        np.load(DELIVERY / f"subject-{s:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
        for s in (0, 1)])
    subject_to_body = mamma_index_for(ours_positions)
    print("=== id resolution 2/2: MAMMA body_id <-> our subject (3D pelvis agreement) ===")
    for s, b in sorted(subject_to_body.items()):
        print(f"our subject {s:02d} <-> body_id-{b:02d}")
    subject_to_tracklet = {
        cam: {s: int(tracklet_evidence[cam]["mapping_body_id_to_tracklet"]
                     [f"body_id-{subject_to_body[s]:02d}"]) for s in (0, 1)}
        for cam in CAMERAS}

    identity = {
        "mask_tracklet_to_body_id": {cam: tracklet_evidence[cam]["mapping_body_id_to_tracklet"]
                                     for cam in CAMERAS},
        "our_subject_to_body_id": {f"subject_{s:02d}": f"body_id-{b:02d}"
                                   for s, b in subject_to_body.items()},
        "our_subject_to_mask_tracklet": {
            cam: {f"subject_{s:02d}": f"{t:02d}" for s, t in subject_to_tracklet[cam].items()}
            for cam in CAMERAS},
        "method": {
            "tracklet_to_body_id":
                "containment of MAMMA's own projected pred_joints "
                f"({', '.join(PROBE_JOINTS)}) in the NATIVE-resolution SAM2 masks, counted "
                "per frame; a joint inside both masks counts as ambiguous and resolves nothing",
            "body_id_to_our_subject":
                "tools/head/subject_map.py -- 3D pelvis agreement, asserted at a >=5x margin",
            "never":
                "no identity is resolved by silhouette overlap. IoU is the quantity this "
                "instrument reports; pairing by it would make every figure circular.",
        },
        "evidence": tracklet_evidence,
    }
    if args.ids_only:
        print(json.dumps({k: v for k, v in identity.items() if k != "evidence"}, indent=2))
        return

    masks = MaskStore(args.scale, cams)
    mesh = delivered_mesh()
    smplx_faces = np.load(SMPLX, allow_pickle=True)["f"].astype(np.int32)
    pred_vertices = {b: np.load(MA3D / f"verts_joints_body_id-{b:02d}.npz",
                                allow_pickle=True)["pred_vertices"] for b in (0, 1)}
    arms = build_arms(mesh, pred_vertices, pred_joints, smplx_faces, subject_to_body)

    alignment = frame_alignment_check(arms["ours_delivered"], masks, subject_to_tracklet,
                                      scaled, shape, cams)
    print(f"frame alignment: {alignment['verdict']} -- {alignment['median_iou_by_lag_frames']}")
    if alignment["verdict"] != "aligned":
        raise SystemExit("our export does not line up with the masks; fix the timebase "
                         "before reading any figure below")

    dilations = {r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                 for r in DILATION_RADII}
    names = list(BASE_ARMS) + [f"control_dilated_{r}px" for r in DILATION_RADII] \
        + ["control_billboard"]

    stats = {n: np.full((len(cams), 2, FRAMES, 3), np.nan) for n in names}
    self_iou = np.full((len(cams), 2, FRAMES), np.nan)
    population = np.zeros((len(cams), 2, FRAMES), dtype=bool)
    mask_area = np.zeros((len(cams), 2, FRAMES), dtype=int)

    started = time.time()
    for c, cam in enumerate(cams):
        for s in (0, 1):
            mask = masks.get(cam, subject_to_tracklet[cam][s])
            area = mask.reshape(FRAMES, -1).sum(axis=1)
            mask_area[c, s] = area
            population[c, s] = area >= MIN_MASK_PX
            for f in np.nonzero(population[c, s])[0]:
                render = {}
                for name in BASE_ARMS:
                    verts, faces = arm_frame(arms[name][s], int(f))
                    render[name] = rasterise(verts, faces, scaled[cam], shape)
                ours = render["ours_delivered"].astype(np.uint8)
                for r, kernel in dilations.items():
                    render[f"control_dilated_{r}px"] = cv2.dilate(ours, kernel).astype(bool)
                box = np.zeros_like(render["ours_delivered"])
                ys, xs = np.nonzero(render["ours_delivered"])
                if len(ys):
                    box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
                render["control_billboard"] = box
                self_iou[c, s, f] = score(render["control_oracle_yaw180_facing"],
                                          render["ORACLE_mamma_mesh"])[2]
                for name in names:
                    stats[name][c, s, f] = score(render[name], mask[f])
            print(f"scored {cam} subject {s:02d}: {int(population[c, s].sum())} frames "
                  f"({time.time() - started:.0f}s)")

    # --- front/back: where does turning the ORACLE 180 degrees change its own outline?
    asymmetric = np.zeros((len(cams), 2, FRAMES), dtype=bool)
    front_back: dict[str, Any] = {
        "criterion": "the ORACLE mesh turned 180 degrees about its own pelvis vertical, "
                     "scored against its own un-turned silhouette. A LOW self-IoU means "
                     "front and back are distinguishable in that camera on that frame. "
                     "Measured on MAMMA's mesh, so the frame list is independent of our "
                     "own unfixed facing defect and of every figure scored here.",
        "finding": "on this take front and back are distinguishable essentially "
                   "everywhere: the median yaw-180 self-IoU is 0.28-0.54 and no frame in "
                   "any camera reaches 0.70. An absolute threshold therefore selects the "
                   "whole population, so the arm figures are split at the per-(camera, "
                   "subject) MEDIAN instead -- 75 frames on each side, same denominator.",
        "split_used_for_the_arm_figures":
            "self-IoU below the per-(camera, subject) median = 'more distinguishable half'",
        "per_camera": {}}
    for c, cam in enumerate(cams):
        front_back["per_camera"][cam] = {}
        for s in (0, 1):
            ok = population[c, s]
            median = float(np.nanmedian(self_iou[c, s][ok]))
            asymmetric[c, s] = (self_iou[c, s] < median) & ok
            front_back["per_camera"][cam][f"subject_{s:02d}"] = {
                "yaw180_self_iou_median": median,
                "yaw180_self_iou_p05": float(np.nanpercentile(self_iou[c, s][ok], 5)),
                "yaw180_self_iou_p95": float(np.nanpercentile(self_iou[c, s][ok], 95)),
                "frames_below_absolute_threshold": {
                    str(t): int(np.count_nonzero((self_iou[c, s] <= t) & ok))
                    for t in ASYMMETRY_THRESHOLDS},
                "more_distinguishable_half_frames":
                    [int(f) for f in np.nonzero(asymmetric[c, s])[0]],
                "count": int(asymmetric[c, s].sum()),
            }
    strict = sorted({int(f) for c in range(len(cams)) for s in (0, 1)
                     for f in np.nonzero((self_iou[c, s] <= 0.30) & population[c, s])[0]})
    front_back["frames_below_0.30_in_at_least_one_camera"] = strict
    front_back["count_below_0.30_in_at_least_one_camera"] = len(strict)

    anatomical = {}
    for s in (0, 1):
        j = pred_joints[subject_to_body[s]]
        up = j[:, NECK] - j[:, PELVIS]
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        right = j[:, RHIP] - j[:, LHIP]
        right -= (right * up).sum(1, keepdims=True) * up
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        forward = np.cross(up, right)
        worst = np.zeros(FRAMES)
        for li, ri in LIMB_PAIRS:
            dl = ((j[:, li] - j[:, PELVIS]) * forward).sum(1)
            dr = ((j[:, ri] - j[:, PELVIS]) * forward).sum(1)
            worst = np.maximum(worst, np.abs(dl - dr))
        anatomical[f"subject_{s:02d}"] = {
            "definition": "max over the wrist / elbow / knee / ankle L-R pairs of the "
                          "difference in signed projection onto the body forward axis, "
                          "relative to the pelvis; camera-independent",
            "median_mm": float(np.median(worst) * 1000.0),
            "p95_mm": float(np.percentile(worst, 95) * 1000.0),
            "frames_over_150mm": int(np.count_nonzero(worst > 0.150)),
        }
    front_back["anatomical_fore_aft_asymmetry"] = anatomical

    # --- report
    results: dict[str, Any] = {}
    for name in names:
        results[name] = {}
        for c, cam in enumerate(cams):
            results[name][cam] = {}
            for s in (0, 1):
                ok = population[c, s]
                entry = summary(stats[name][c, s][ok])
                entry["frames_scored"] = int(ok.sum())
                sel = asymmetric[c, s]
                rest = ok & ~sel
                if sel.any():
                    entry["front_back_more_distinguishable_half"] = dict(
                        n=int(sel.sum()), **summary(stats[name][c, s][sel]))
                if rest.any():
                    entry["front_back_less_distinguishable_half"] = dict(
                        n=int(rest.sum()), **summary(stats[name][c, s][rest]))
                results[name][cam][f"subject_{s:02d}"] = entry

    # How much can a silhouette see a 180-degree facing error AT ALL? The oracle's IoU
    # minus the same mesh turned around, both against the same masks. This is the number
    # step D1 needs from I6: if it is small, a silhouette cannot gate facing and D1 rests
    # on the forward-dot check alone.
    facing = {"definition": "median IoU of ORACLE_mamma_mesh minus median IoU of "
                            "control_oracle_yaw180_facing, same frames, same masks",
              "per_camera": {}}
    drops_all, drops_half = [], []
    for c, cam in enumerate(cams):
        facing["per_camera"][cam] = {}
        for s in (0, 1):
            ok, sel = population[c, s], asymmetric[c, s]
            whole = float(np.median(stats["ORACLE_mamma_mesh"][c, s][ok, 2])
                          - np.median(stats["control_oracle_yaw180_facing"][c, s][ok, 2]))
            half = float(np.median(stats["ORACLE_mamma_mesh"][c, s][sel, 2])
                         - np.median(stats["control_oracle_yaw180_facing"][c, s][sel, 2]))
            drops_all.append(whole)
            drops_half.append(half)
            facing["per_camera"][cam][f"subject_{s:02d}"] = {
                "all_frames": whole, "more_distinguishable_half": half}
    facing["oracle_minus_yaw180_iou_on_distinguishable_frames"] = {
        "median": float(np.median(drops_half)),
        "min": float(np.min(drops_half)), "max": float(np.max(drops_half))}
    facing["oracle_minus_yaw180_iou_all_frames"] = {
        "median": float(np.median(drops_all)),
        "min": float(np.min(drops_all)), "max": float(np.max(drops_all))}
    # D1's question asked of the pixels: is the DELIVERED mesh facing the right way?
    # With a moving-block bootstrap on each cell: this is the load-bearing margin in the
    # whole report and a median on 150 frames of lag-1 0.97 data is not a robust pass.
    facing_rng = np.random.default_rng(SEED + 1)
    turned_better, per_cell, decisive = [], {}, 0
    for c, cam in enumerate(cams):
        for s in (0, 1):
            ok = population[c, s]
            boot = moving_block_bootstrap(stats["control_ours_yaw180_facing"][c, s][ok, 2],
                                          stats["ours_delivered"][c, s][ok, 2], facing_rng)
            turned_better.append(boot["median_difference"])
            per_cell[f"{cam}_subject_{s:02d}"] = boot
            if boot["ci95_of_the_median_difference"][1] < 0.0:
                decisive += 1
    cells = len(turned_better)
    facing["delivered_mesh_turned_180_minus_delivered_mesh_iou"] = {
        "per_cell": per_cell,
        "median": float(np.median(turned_better)),
        "max": float(np.max(turned_better)),
        "cells_where_the_upper_ci_bound_is_below_zero": f"{decisive} of {cells}",
        "verdict": (f"the delivered facing beats its own 180-degree turn in all {cells} "
                    "camera x subject cells, every one of them with the upper bound of a "
                    "moving-block 95 % interval below zero"
                    if decisive == cells and max(turned_better) < 0 else
                    f"only {decisive} of {cells} cells are decisive at 95 % -- read the "
                    "per-cell intervals before quoting this"),
        "reading": "if the delivery shipped a 180-degree facing error, turning it round "
                   "would IMPROVE agreement with the footage. One of the two orientations "
                   "has to lose, and the instrument has the sensitivity to tell them apart "
                   "(see oracle_minus_yaw180_iou_*). This does NOT clear a left/right "
                   "MIRROR, which a silhouette cannot see on a fore-aft symmetric pose, "
                   "and it does not speak to any internal 180-degree yaw that a later "
                   "stage compensates -- it speaks only to the delivered surface.",
    }
    facing["reading"] = (
        "a 180-degree facing error costs the ORACLE this much IoU on the very masks it "
        "otherwise fits best, so the silhouette IS a facing instrument on this take -- "
        "but it is a scalar and a mirrored (rather than turned) human can hold its "
        "outline, which is why D1's forward-dot check is the primary gate and this is "
        "the corroboration")

    rng = np.random.default_rng(SEED)
    margins = {}
    for c, cam in enumerate(cams):
        margins[cam] = {}
        for s in (0, 1):
            ok = population[c, s]
            margins[cam][f"subject_{s:02d}"] = moving_block_bootstrap(
                stats["ours_delivered"][c, s][ok, 2],
                stats["ORACLE_mamma_mesh"][c, s][ok, 2], rng)

    agreement = ({"skipped": True} if args.skip_point_to_surface
                 else point_to_surface(mesh, pred_vertices, smplx_faces, subject_to_body))

    report = {
        "step": "I6",
        "title": "surface instrument: silhouette of the delivered mesh vs MAMMA's SAM2 masks",
        "fixture": FIXTURE,
        "frame_window_of_take": list(FRAME_WINDOW),
        "frames": FRAMES,
        "cameras": list(cams),
        "identity": identity,
        "render": {
            "native_resolution": list(NATIVE),
            "render_resolution": [width, height],
            "downscale_factor": args.scale,
            "rasteriser": "cv2.fillConvexPoly ONE CALL PER TRIANGLE, no z-buffer (a "
                          "silhouette is the union of the projected triangles), 4-bit "
                          "subpixel shift, faces with any vertex at depth <= 0 dropped. "
                          "Batching every triangle into a single cv2.fillPoly call is "
                          "WRONG and was the first version of this instrument: fillPoly "
                          "applies an even-odd rule ACROSS the contours it is handed, so "
                          "the self-overlapping projection of a closed body cancels "
                          "itself. It disagreed with the correct fill on 190 of ~7000 px "
                          "on a real frame here -- small enough to look plausible.",
            "mask_downsampling": "cv2.INTER_AREA then > 127: a render pixel is foreground "
                                 "when more than half of its native footprint was",
            "camera_model": "CalibratedCamera.project (pinhole, no distortion -- MAMMA's "
                            "ma_cap frames are already undistorted) scaled to the render size",
            "delivered_mesh": "subject-XX.glb posed in Blender 4.2 with scene fps 30 set "
                              "BEFORE the import, evaluated depsgraph, world matrix applied; "
                              "13380 vertices / 26756 triangles per subject",
            "oracle_mesh": "pred_vertices (150, 10475, 3) with the SMPLX_NEUTRAL faces "
                           "(20908, 3)",
            "population": f"(camera, frame, subject) whose resolved mask covers at least "
                          f"{MIN_MASK_PX} px at render resolution; identical for every arm",
            "mask_area_px_median": {cam: {f"subject_{s:02d}": int(np.median(mask_area[c, s]))
                                          for s in (0, 1)} for c, cam in enumerate(cams)},
            "frame_alignment": alignment,
            "resolution_check": resolution_check(results, width),
        },
        "arms": results,
        "ours_minus_oracle_iou_moving_block_bootstrap": margins,
        "facing_sensitivity": facing,
        "front_back": front_back,
        "point_to_surface_agreement": agreement,
        "blind_to":
            "A silhouette is blind to DEPTH -- the outline is a projection, so nearer or "
            "farther along the viewing ray fills the same pixels and only the four cameras "
            "disagreeing constrains it. It is blind to left/right MIRRORING of a fore-aft "
            "symmetric pose. And it is blind to EVERYTHING INSIDE THE OUTLINE, which is most "
            "of a body, including every self-occluded limb: an arm swung through the torso "
            "costs nothing. It cannot separate a shape error from a pose error. The masks "
            "segment a CLOTHED person with hair and both meshes are nude bodies, so no arm "
            "can reach recall 1 -- measuring that ceiling is what the oracle arm is for, and "
            "the gap is not a defect. It is NOT blind, on the frames where the body is "
            "fore-aft asymmetric in a camera, to a 180-degree facing error: that is the one "
            "thing step D1 needs from it, and control_oracle_yaw180_facing is the "
            "self-consistent backwards human that D1's gate must reject.",
        "standing_rules": {
            "mamma_ships": False,
            "constants_selected_here":
                "none. This instrument reports. The dilation radii, the asymmetry threshold, "
                "the mask floor and the render resolution are instrument settings and enter "
                "no delivered artifact.",
            "same_denominator":
                "every arm scored on the same (camera, frame, subject) population",
            "gt_variables_used": "none -- gt_joints / gt_vertices are byte-copies of pred_*",
        },
        "input_sha256": {
            "subject-00.glb": sha256(DELIVERY / "subject-00.glb"),
            "subject-01.glb": sha256(DELIVERY / "subject-01.glb"),
            "camera-rig.json": sha256(RIG_PATH),
            "SMPLX_NEUTRAL.npz": sha256(SMPLX),
            "verts_joints_body_id-00.npz": sha256(MA3D / "verts_joints_body_id-00.npz"),
            "verts_joints_body_id-01.npz": sha256(MA3D / "verts_joints_body_id-01.npz"),
            "smplx_params_body_id-00.npz": sha256(MA3D / "smplx_params_body_id-00.npz"),
            "smplx_params_body_id-01.npz": sha256(MA3D / "smplx_params_body_id-01.npz"),
            "ma_masks_A001_mask_0000_01.png": sha256(MASKS / "A001/masks/mask_0000_01.png"),
        },
        "plan_corrections": [
            "tools/swap-harness/camera_overlay.py does NOT set scene.render.fps before "
            "importing the GLB. The delivered motion is 30 fps and Blender's factory scene "
            "is 24, so the action arrives spanning frames 0..119.2: every overlay it has "
            "rendered for frame F actually shows take frame 60 + 1.25*F, and any F > 119 "
            "shows a frozen last pose. One line fixes it (scene.render.fps = 30 before the "
            "import). Step D1's flip check runs through that script.",
            "The masks live at ma_masks/ma_cap/<take>/<cam>/masks/mask_NNNN_XX.png, not "
            "ma_masks/<take>/<cam>/. masks.npy beside them is a TORCH pickle and cannot be "
            "read from .venv at all; the 300 PNGs per camera carry the same data and load "
            "in the venv, so this instrument uses those.",
            "The mask tracklet ids are the PNG suffixes 01 and 02, not 0 and 1; the "
            "anchor_report.json in the same directory indexes them from 0.",
            d1_tension(facing, front_back, cams),
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nWROTE {args.out}")
    summarise(report, cams)


def d1_tension(facing: dict, front_back: dict, cams: tuple[str, ...]) -> str:
    """The one place this instrument disagrees with the plan, stated as tension.

    Every number in the sentence is computed, so it cannot go stale behind a rerun.
    """
    turned = facing["delivered_mesh_turned_180_minus_delivered_mesh_iou"]
    per_cell = turned["per_cell"]
    sensitivity = facing["oracle_minus_yaw180_iou_on_distinguishable_frames"]
    weakest = sorted(per_cell, key=lambda k: per_cell[k]["median_difference"])[-2:]
    self_iou = {f"{cam}_subject_{s:02d}":
                front_back["per_camera"][cam][f"subject_{s:02d}"]["yaw180_self_iou_median"]
                for cam in cams for s in (0, 1)}
    least_visible = sorted(self_iou, key=self_iou.get)[-2:]
    agrees = set(weakest) == set(least_visible)
    return (
        "TENSION WITH D1's GATE CARD, not a verdict. LADDER_EXECUTION_PLAN's D1 row assumes "
        "the delivery ships a 180-degree facing yaw out of _frame_alignment. This instrument "
        "has demonstrated sensitivity to exactly that error -- turning MAMMA's own mesh round "
        f"costs it {sensitivity['min']:.2f}-{sensitivity['max']:.2f} IoU on the "
        "front/back-distinguishable half -- and finds no such error in the delivered GLB: "
        f"turning OUR mesh round lowers its median IoU in all {len(per_cell)} camera x "
        "subject cells, decisively (upper bound of a moving-block 95 % interval below zero) "
        f"in {turned['cells_where_the_upper_ci_bound_is_below_zero']}. The weakest cells are "
        + ", ".join(f"{k} ({per_cell[k]['median_difference']:+.2f})" for k in weakest)
        + (", and those are the instrument agreeing with itself rather than wobbling: they "
           "are also the two cells with the HIGHEST yaw-180 self-IoU ("
           + ", ".join(f"{k} {self_iou[k]:.2f}" for k in least_visible)
           + "), i.e. the two views in which front and back look most alike. "
           if agrees else
           ", which are NOT the cells where front and back look most alike ("
           + ", ".join(f"{k} {self_iou[k]:.2f}" for k in least_visible)
           + ") -- worth a second look before quoting the margin. ")
        + "Three readings survive and this instrument cannot choose between them: the yaw is "
        "internal and already compensated before export; the 2026-08-30 flip-check diagnosis "
        "located it wrongly; or the defect is a left/right MIRROR, which a silhouette is "
        "structurally blind to on a fore-aft symmetric pose. D1 should re-confirm where the "
        "defect lives before changing code.")


HALF_RES = WORK / "silhouette-1920x1080-A001.json"


def resolution_check(results: dict, width: int) -> dict:
    """Is 960x540 fine enough, or is the figure a quantisation artefact?

    The subject is ~850 px tall natively and ~215 px at the render resolution, so the
    outline is a few thousand pixels and the boundary is a real fraction of them. Rerun
    one camera at twice the linear resolution and compare; a figure that moves is a
    figure about the rasteriser. Regenerate the companion with

        .venv/bin/python tools/compare/silhouette.py --scale 2 --cameras A001 \\
            --skip-point-to-surface --out artifacts/compare/i6/silhouette-1920x1080-A001.json
    """
    if not HALF_RES.exists():
        return {"status": "not run", "how": "see the docstring of resolution_check()"}
    other = json.loads(HALF_RES.read_text())
    if other["render"]["render_resolution"][0] == width:
        return {"status": "same resolution as the main run; no comparison"}
    out: dict[str, Any] = {
        "status": "run",
        "compared_at": other["render"]["render_resolution"],
        "cameras": other["cameras"],
        "reading": "a difference here is the rasteriser and the mask downsampling, not "
                   "the pipeline; the arms move together or the resolution is too coarse",
        "iou_median": {}}
    for arm in ("ours_delivered", "ORACLE_mamma_mesh"):
        out["iou_median"][arm] = {}
        for cam in other["cameras"]:
            for s in (0, 1):
                key = f"{cam}_subject_{s:02d}"
                coarse = results[arm][cam][f"subject_{s:02d}"]["iou"]["median"]
                fine = other["arms"][arm][cam][f"subject_{s:02d}"]["iou"]["median"]
                out["iou_median"][arm][key] = {
                    "at_main_resolution": coarse, "at_finer_resolution": fine,
                    "difference": fine - coarse}
    return out


def point_to_surface(mesh: dict, pred_vertices: dict, smplx_faces: np.ndarray,
                     subject_to_body: dict[int, int], stride: int = 15,
                     sample: int = 2000) -> dict:
    """Symmetric point-to-SURFACE distance, ours <-> MAMMA's mesh. AGREEMENT ONLY.

    Both sides are fits, of different body models, with different vertex counts and
    different surface conventions. This number is model-mediated in a way the silhouette
    figures are not: it says nothing about the footage. It is here because D6 will want a
    dense figure, labelled so it is never read as accuracy.
    """
    import trimesh  # noqa: PLC0415 -- only this function needs it
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {
        "label": "AGREEMENT, model-mediated -- NOT a measurement against the footage; "
                 "both sides are fits and neither is truth",
        "method": f"trimesh.proximity.closest_point (true point-to-triangle, not "
                  f"nearest-vertex), {sample} sampled vertices in each direction, "
                  f"every {stride}th frame",
        "subjects": {}}
    frames = list(range(0, FRAMES, stride))
    for s in (0, 1):
        ours_v = mesh[f"verts_{s:02d}"].astype(np.float64)
        ours_f = mesh[f"faces_{s:02d}"]
        theirs = pred_vertices[subject_to_body[s]].astype(np.float64)
        forward, backward = [], []
        for f in frames:
            a = trimesh.Trimesh(ours_v[f], ours_f, process=False)
            b = trimesh.Trimesh(theirs[f], smplx_faces, process=False)
            pa = ours_v[f][rng.choice(len(ours_v[f]), sample, replace=False)]
            pb = theirs[f][rng.choice(len(theirs[f]), sample, replace=False)]
            forward.append(np.abs(trimesh.proximity.closest_point(b, pa)[1]))
            backward.append(np.abs(trimesh.proximity.closest_point(a, pb)[1]))
        fw, bw = np.concatenate(forward), np.concatenate(backward)
        both = np.concatenate([fw, bw])
        out["subjects"][f"subject_{s:02d}"] = {
            "ours_to_mamma_median_mm": float(np.median(fw) * 1000.0),
            "mamma_to_ours_median_mm": float(np.median(bw) * 1000.0),
            "symmetric_median_mm": float(np.median(both) * 1000.0),
            "symmetric_p95_mm": float(np.percentile(both, 95) * 1000.0),
            "frames": frames,
        }
    return out


def summarise(report: dict, cams: tuple[str, ...]) -> None:
    print("\n=== median IoU / precision / recall ===")
    header = "  ".join(f"{cam} s{s}".ljust(17) for cam in cams for s in (0, 1))
    print(f"{'arm':34s} {header}")
    for name, per_cam in report["arms"].items():
        cells = []
        for cam in cams:
            for s in (0, 1):
                e = per_cam[cam][f"subject_{s:02d}"]
                cells.append(f"{e['iou']['median']:.3f}/{e['precision']['median']:.3f}/"
                             f"{e['recall']['median']:.3f}")
        print(f"{name:34s} " + "  ".join(c.ljust(17) for c in cells))


if __name__ == "__main__":
    main()
