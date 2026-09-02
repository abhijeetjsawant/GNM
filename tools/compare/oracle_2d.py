#!/usr/bin/env python3
"""I2 -- the perfect-2D oracle. Our whole pipeline on MAMMA's own skeleton.

Every millimetre this lane has measured against MAMMA mixes two terms: what our
detector got wrong in 2D, and what our reconstruction does to 2D once it has it.
This separates them. Project MAMMA's `pred_joints` into the four calibrated
cameras, feed the projections through `reconstruct_multiview` in the ordinary
observation contract, and score the result against the very joints that were
projected. Whatever error survives is the *floor our pipeline imposes* -- the
sequence solve, the fill, the Savitzky-Golay window and the association gates --
with the detector removed by construction.

This is the ORACLE ARM for rungs 2-7 of `docs/SUBSTITUTION_LADDER.md`. A gate no
oracle can pass is miscalibrated, and until now those rungs had none.

    .venv/bin/python tools/compare/oracle_2d.py

Arms
----
* **exact**       noiseless projections of `pred_joints`.
* **noise**       the same projections displaced by MAMMA's own measured 2D
                  residual distribution (its fitted 512 landmarks reprojected
                  against its own `ma_2d`), >= 5 seeds, mean and spread reported.
* **controls**    shuffled subject pairing; camera A001 time-shifted +1 and -1;
                  a frozen single-frame skeleton projected on every frame. Each
                  must fail, and each is in the report with its figure.

Standing rules honoured here
----------------------------
* **Wrap the pipeline; never re-implement it.** Every arm calls the real
  `reconstruct_multiview`. A hand replication of its association loop drifted
  9-19 mm from the retained tracks; there is no second copy of it in this file.
* MAMMA is an instrument. Nothing here selects a shipped constant, and the report
  lands under `artifacts/compare/`, never under `artifacts/commercial-multiview-*`.
* Subject correspondence is derived from pelvis agreement (`subject_map.py`),
  never assumed from the index: MAMMA's `body_id-00` is our subject 1 here.
* `gt_joints` / `gt_vertices` are byte-copies of `pred_*` and are never opened.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))

from autoanim_gnm import commercial_multiview as cm  # noqa: E402
from subject_map import mamma_index_for  # noqa: E402

MAMMA_OUT = ROOT / "artifacts/mamma/mamma-4cam-five-second-v2/output"
FIXTURE = "pushing_and_lifting_from_ground"
MA3D = MAMMA_OUT / "ma_3d" / FIXTURE
MA2D = MAMMA_OUT / "ma_2d" / FIXTURE
RIG = ROOT / "artifacts/commercial-multiview-soma77/camera-rig.json"
VERTS_512 = ROOT / ".cache/mamma/data/body_models/downsampled_verts/verts_512.pkl"
CACHE = ROOT / "artifacts/compare/oracle-2d-cache"
REPORT = ROOT / "artifacts/compare/oracle-2d.json"

CAMERAS = ("A001", "B001", "C001", "D001")
# The delivered detector runs at 1280 wide, and every pixel-denominated gate
# inside the pipeline is quoted at that width. Running the oracle at native
# 3840 would silently loosen all of them, so the floor would not be the floor
# the delivery path actually stands on.
WORKING_WIDTH, WORKING_HEIGHT = 1280, 720
NATIVE_WIDTH = 3840
SCHEMA_VERSION = "autoanim.body-observations/1.1"
DETECTOR = "mamma_pred_joints_projected"
SAMPLE_RATE_HZ = 30

# our 19-joint name -> SMPL-X joint index in MAMMA's `pred_joints`.
# Copied from tools/compare/mamma_scoreboard.py so both instruments score the
# same 15 joints against the same reference and may share an axis. `nose` is
# SMPL-X joint 15 -- the Head joint, a point inside the skull, not a surface
# nose (CLAUDE.md).
PAIRS = {
    "root": 0, "neck": 12, "nose": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
    "left_hip": 1, "right_hip": 2,
    "left_knee": 4, "right_knee": 5,
    "left_ankle": 7, "right_ankle": 8,
}
SCORED = tuple(PAIRS)
PELVIS = PAIRS["root"]

# Moving-block bootstrap: per-frame agreement in this lane has lag-1
# autocorrelation 0.99, so ordinary resampling is invalid. Half a second of
# frames per block, candidate and oracle on identical draws.
BLOCK_FRAMES = 15
BOOTSTRAP_DRAWS = 1000


# --------------------------------------------------------------------------- inputs

def sha256_of(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_truth() -> np.ndarray:
    """MAMMA's fitted joints, [body_id, frame, 127, 3], world Z-up metres.

    `pred_joints`, never `gt_joints` -- the latter is a byte-copy of the former
    and scoring against it would dress an instrument up as ground truth.
    """
    return np.stack([
        np.load(MA3D / f"verts_joints_body_id-{body:02d}.npz",
                allow_pickle=True)["pred_joints"].astype(np.float64)
        for body in (0, 1)
    ])


def verts_512_regressor() -> np.ndarray:
    """MAMMA's 512-landmark regressor, cached as plain float64.

    The file on disk is a pickled `torch.Tensor` and torch is not in `.venv`, so
    the conversion runs once through the system interpreter -- the same split the
    ladder documents for the swap-harness instruments.
    """
    cached = CACHE / "verts512-regressor.npy"
    if cached.exists():
        return np.load(cached)
    CACHE.mkdir(parents=True, exist_ok=True)
    script = (
        "import pickle, numpy, sys\n"
        "t = pickle.load(open(sys.argv[1], 'rb'))\n"
        "numpy.save(sys.argv[2], numpy.asarray(t).astype(numpy.float64))\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(script)
        helper = handle.name
    subprocess.run(["python3", helper, str(VERTS_512), str(cached)], check=True)
    Path(helper).unlink()
    return np.load(cached)


# ------------------------------------------------------------------- projection check

def projection_convention_check(cameras, truth: np.ndarray) -> dict:
    """Is our projection convention the one MAMMA's own 2D was written in?

    A SANITY CHECK, NOT A SCORE. Two independent readings, both in native pixels:

    * *dense* -- regress MAMMA's 512 landmarks from its own `pred_vertices`,
      project them through our rig, and subtract its `ma_2d`. This residual is a
      lower bound on MammaNet's error (the fit consumed those landmarks) and is
      useless as an accuracy figure; as a convention check it is decisive,
      because a swapped axis, a mirrored principal point or a wrong quaternion
      order puts it in the hundreds or thousands of pixels rather than single
      digits. It is also the pool the noise arm samples from.
    * *per joint* -- for each of the 15 scored joints, the nearest of those 512
      surface landmarks in 3D, then that landmark's own `ma_2d` position against
      our projection of the joint. A skeletal joint sits inside the body, so the
      2D offset here is expected to be roughly the 3D offset scaled by focal
      length over depth; both are reported so the reader can check that it is.
    """
    regressor = verts_512_regressor()
    landmarks = np.stack([np.load(MA2D / f"{name}.npz", allow_pickle=True)["landmarks"]
                          for name in CAMERAS])
    visibility = np.stack([np.load(MA2D / f"{name}.npz", allow_pickle=True)["visibilities"]
                           for name in CAMERAS])
    native = cm.load_camera_rig(RIG)

    dense: list[np.ndarray] = []
    per_joint: dict[str, dict] = {}
    for body in (0, 1):
        vertices = np.load(MA3D / f"verts_joints_body_id-{body:02d}.npz",
                           allow_pickle=True)["pred_vertices"].astype(np.float64)
        surface = np.einsum("kv,fvc->fkc", regressor, vertices)
        for index in range(len(CAMERAS)):
            for frame in range(0, vertices.shape[0], 2):
                uv, depth = native[index].project(surface[frame])
                keep = (depth > 0.0) & (visibility[index, frame, body] >= 0.5)
                dense.append(np.linalg.norm(uv - landmarks[index, frame, body, :, :2],
                                            axis=1)[keep])
        # one frame, one camera is enough for the per-joint reading
        frame, index = 0, 0
        for name, joint in PAIRS.items():
            point = truth[body, frame, joint]
            nearest = int(np.argmin(np.linalg.norm(surface[frame] - point, axis=1)))
            offset_mm = float(np.linalg.norm(surface[frame, nearest] - point) * 1000.0)
            uv, depth = native[index].project(point)
            offset_px = float(np.linalg.norm(uv - landmarks[index, frame, body, nearest, :2]))
            focal = float(native[index].intrinsics[0, 0])
            ray = point - native[index].camera_center_world_m
            predicted = offset_mm / 1000.0 * focal / float(np.linalg.norm(ray))
            per_joint.setdefault(name, {})[f"body_id-{body:02d}"] = {
                "nearest_surface_landmark": nearest,
                "joint_to_landmark_mm": round(offset_mm, 2),
                "our_projection_vs_ma2d_px_native": round(offset_px, 2),
                "px_predicted_by_the_3d_offset": round(float(predicted), 2),
            }
    pool = np.concatenate(dense)
    quantiles = [10, 25, 50, 75, 90, 95, 99]
    return {
        "convention": "world is Z-up metres and is MAMMA's own world frame; "
                      "cm.CalibratedCamera.project (pinhole, no distortion, no image "
                      "flip); intrinsics scaled by CalibratedCamera.scaled for the "
                      f"{WORKING_WIDTH}x{WORKING_HEIGHT} working size",
        "verified_by": "MAMMA's 512 fitted surface landmarks reprojected through our rig "
                       "against its own ma_2d, plus a per-joint nearest-landmark reading",
        "is_a_score": False,
        "dense_residual_px_native": {
            "quantiles": quantiles,
            "values": [round(float(v), 2) for v in np.percentile(pool, quantiles)],
            "n": int(pool.size),
            "note": "visibility >= 0.5, every second frame, all four cameras, both bodies; "
                    "a lower bound on MammaNet's error because the fit consumed these "
                    "landmarks -- shape only, never an accuracy figure",
        },
        "per_joint_native_px": per_joint,
    }, pool


# ----------------------------------------------------------------------- observations

def observations_for(cameras, subjects: np.ndarray, *, noise=None) -> list[list[dict]]:
    """Project truth into the ordinary observation contract.

    `subjects` is [subject, frame, 127, 3]; only the 15 joints in PAIRS are
    emitted, so the four head landmarks our 19-joint contract carries (eyes and
    ears) are absent exactly as they are absent from SOMA-77 on real footage.

    `noise`, when given, is called with a count and returns that many radial
    displacements in working pixels; the direction is uniform. Confidence stays
    at the noiseless 0.95 in both arms: a fabricated per-observation confidence
    would hand the solver a sigma the real pipeline never gets, and the arms
    would then differ in two things instead of one.
    """
    frames = subjects.shape[1]
    records: list[list[dict]] = []
    for index, camera in enumerate(cameras):
        rows: list[dict] = []
        for frame in range(frames):
            people = []
            for subject in range(subjects.shape[0]):
                joints = {}
                for name, joint in PAIRS.items():
                    uv, depth = camera.project(subjects[subject, frame, joint])
                    if depth <= 0.0:
                        continue
                    if noise is not None:
                        radius = noise(1)[0]
                        angle = noise.rng.uniform(0.0, 2.0 * np.pi)
                        uv = uv + radius * np.asarray([np.cos(angle), np.sin(angle)])
                    joints[name] = {"x": float(uv[0]), "y": float(uv[1]), "confidence": 0.95}
                if joints:
                    people.append({"index": len(people), "joints": joints})
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "detector": DETECTOR,
                "frame_index": frame,
                "width": camera.width,
                "height": camera.height,
                "image_path": f"oracle://{CAMERAS[index]}/{frame:06d}",
                "people": people,
            })
        records.append(rows)
    return records


class ResidualNoise:
    """MAMMA's own measured 2D residual, resampled by inverse CDF.

    The pool is radial distance in native pixels; the pipeline runs at 1280, so
    it is divided by the width ratio before it is drawn from. Sampling the
    empirical quantiles rather than a fitted lognormal keeps the tail MAMMA
    actually has instead of the tail a two-parameter family can express.
    """

    def __init__(self, pool_native_px: np.ndarray, seed: int) -> None:
        grid = np.linspace(0.0, 100.0, 2001)
        self.quantiles = np.percentile(pool_native_px, grid) * (WORKING_WIDTH / NATIVE_WIDTH)
        self.probabilities = grid / 100.0
        self.rng = np.random.default_rng(seed)

    def __call__(self, count: int) -> np.ndarray:
        return np.interp(self.rng.random(count), self.probabilities, self.quantiles)


def time_shifted(records: list[list[dict]], camera_index: int, shift: int) -> list[list[dict]]:
    """One camera's frames moved by `shift`, edges held, frame numbers restamped.

    The pipeline refuses streams whose frame numbers differ, so the shift has to
    be in the *content*: the observation that was frame f-shift now carries
    frame f. That is precisely the failure a real rig produces when one camera's
    sync is off by a frame.
    """
    shifted = [list(rows) for rows in records]
    source = records[camera_index]
    frames = len(source)
    rows = []
    for frame in range(frames):
        origin = min(max(frame - shift, 0), frames - 1)
        row = dict(source[origin])
        row["frame_index"] = frame
        row["image_path"] = f"{row['image_path']}#shift{shift:+d}"
        rows.append(row)
    shifted[camera_index] = rows
    return shifted


# ---------------------------------------------------------------------------- scoring

AMBIGUOUS_MAP_ARMS: list[str] = []


def resolve_mapping(positions: np.ndarray, fallback: dict[int, int] | None = None,
                    arm: str = "") -> dict[int, int]:
    """Pelvis-agreement subject map, with a named fallback for degenerate arms.

    `mamma_index_for` refuses to guess below a 5x margin, which is right: on a
    control arm whose output is a constant the pelvis distances can close up, and
    a silently guessed pairing would turn a control's failure into a pairing bug.
    When it refuses, the exact arm's derived map is used and the report says so.
    """
    try:
        return mamma_index_for(positions)
    except SystemExit:
        if fallback is None:
            raise
        AMBIGUOUS_MAP_ARMS.append(arm)
        print(f"    subject map ambiguous on arm {arm!r}; using the exact arm's map")
        return dict(fallback)


def run_arm(cameras, records: list[list[dict]]) -> tuple[np.ndarray, np.ndarray, dict, float]:
    """One pass of the REAL pipeline. Returns (smoothed, raw, diagnostics, seconds).

    `raw` is the pre-fill triangulation with its NaNs intact -- the third and fourth
    returns of `reconstruct_multiview` share a shape and are easy to swap, and the
    fourth is the one that has not been through the sequence solve, the fill or the
    Savitzky-Golay window. Scoring both is what splits the floor into geometry and
    temporal.
    """
    started = time.time()
    _tracks, diagnostics, positions, raw = cm.reconstruct_multiview(
        cameras, records, subject_count=2, sample_rate_hz=SAMPLE_RATE_HZ
    )
    return positions, raw, diagnostics.as_dict(), time.time() - started


def error_matrices(positions: np.ndarray, truth: np.ndarray,
                   mapping: dict[int, int]) -> dict[int, dict[str, np.ndarray]]:
    """Per subject, [frame, joint] millimetre error, absolute and root-relative.

    Root-relative subtracts each arm's OWN root -- our `root` joint from ours,
    SMPL-X pelvis from MAMMA's -- so the figure is the pose with the global
    placement taken out, and never a partial alignment of one to the other.
    """
    out: dict[int, dict[str, np.ndarray]] = {}
    frames = min(positions.shape[1], truth.shape[1])
    for subject, body in mapping.items():
        ours = np.stack([positions[subject, :frames, cm.JOINT_INDEX[n]] for n in SCORED], axis=1)
        theirs = np.stack([truth[body, :frames, PAIRS[n]] for n in SCORED], axis=1)
        absolute = np.linalg.norm(ours - theirs, axis=2) * 1000.0
        our_root = positions[subject, :frames, cm.JOINT_INDEX["root"]][:, None, :]
        their_root = truth[body, :frames, PELVIS][:, None, :]
        relative = np.linalg.norm((ours - our_root) - (theirs - their_root), axis=2) * 1000.0
        out[subject] = {"absolute": absolute, "relative": relative}
    return out


def summarise(matrix: np.ndarray) -> dict:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return {"median_mm": None, "p95_mm": None, "max_mm": None, "coverage": 0.0}
    return {
        "median_mm": round(float(np.median(finite)), 3),
        "p95_mm": round(float(np.percentile(finite, 95)), 3),
        "max_mm": round(float(finite.max()), 3),
        "coverage": round(float(finite.size / matrix.size), 5),
    }


def block_bootstrap_median(matrix: np.ndarray, seed: int = 11) -> dict:
    """Moving-block bootstrap CI on the median. Blocks, because lag-1 is 0.99."""
    rng = np.random.default_rng(seed)
    frames = matrix.shape[0]
    blocks = int(np.ceil(frames / BLOCK_FRAMES))
    starts_max = max(frames - BLOCK_FRAMES, 0)
    medians = []
    for _ in range(BOOTSTRAP_DRAWS):
        starts = rng.integers(0, starts_max + 1, blocks)
        draw = np.concatenate([matrix[s:s + BLOCK_FRAMES] for s in starts]).ravel()
        draw = draw[np.isfinite(draw)]
        if draw.size:
            medians.append(float(np.median(draw)))
    if not medians:
        return {}
    return {"block_frames": BLOCK_FRAMES, "draws": len(medians),
            "median_ci95_mm": [round(float(np.percentile(medians, 2.5)), 3),
                               round(float(np.percentile(medians, 97.5)), 3)]}


def score_arm(positions: np.ndarray, truth: np.ndarray, mapping: dict[int, int],
              *, bootstrap: bool = False) -> dict:
    matrices = error_matrices(positions, truth, mapping)
    subjects: dict[str, dict] = {}
    pooled = {"absolute": [], "relative": []}
    for subject, body in sorted(mapping.items()):
        entry: dict = {"mamma_body_id": f"body_id-{body:02d}"}
        for basis in ("absolute", "relative"):
            matrix = matrices[subject][basis]
            entry[basis] = summarise(matrix)
            if bootstrap:
                entry[basis].update(block_bootstrap_median(matrix))
            pooled[basis].append(matrix)
        entry["per_joint_median_mm_absolute"] = {
            name: round(float(np.nanmedian(matrices[subject]["absolute"][:, index])), 3)
            for index, name in enumerate(SCORED)
        }
        subjects[f"our_subject_{subject:02d}"] = entry
    overall = {basis: summarise(np.concatenate(pooled[basis], axis=0))
               for basis in ("absolute", "relative")}
    return {"subjects": subjects, "overall": overall}


# ------------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5,
                        help="noise-arm seeds; the plan's floor is 5")
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args()
    # `subject_map.py` resolves MAMMA's output relative to the repo root. Done here
    # rather than at import so importing this module (a test, the extractor) does not
    # move the caller's working directory out from under it.
    os.chdir(ROOT)

    rig = cm.load_camera_rig(RIG)
    cameras = tuple(camera.scaled(WORKING_WIDTH, WORKING_HEIGHT) for camera in rig)
    truth = load_truth()
    print(f"truth: {truth.shape[0]} MAMMA bodies x {truth.shape[1]} frames, "
          f"{len(PAIRS)} of 127 joints scored")

    convention, residual_pool = projection_convention_check(cameras, truth)
    print("projection check, dense residual px @3840: "
          f"{convention['dense_residual_px_native']['values']}")

    arms: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}

    # --- exact ------------------------------------------------------------------
    exact_records = observations_for(cameras, truth)
    positions, raw, diagnostics["exact"], seconds = run_arm(cameras, exact_records)
    mapping = mamma_index_for(positions)
    print(f"subject map from pelvis agreement: "
          f"{', '.join(f'our {k} -> body_id-{v:02d}' for k, v in sorted(mapping.items()))}"
          f"   ({seconds:.1f}s)")
    arms["exact"] = score_arm(positions, truth, mapping, bootstrap=True)
    arms["exact"]["what_it_is"] = (
        "noiseless projections of MAMMA's pred_joints through our rig, through the whole "
        "pipeline -- association, triangulation, sequence solve, fill, Savitzky-Golay -- "
        "scored against the joints that were projected. THE ORACLE FLOOR."
    )
    arms["exact"]["before_the_temporal_stage"] = score_arm(raw, truth, mapping)
    # "0.0 mm" at three decimals invites the suspicion that something is being scored
    # against itself, so say how zero it is. `raw` is reconstruct_multiview's FOURTH
    # return -- the pre-fill triangulation -- and `truth` is MAMMA's array; they meet
    # only through a projection and a DLT.
    raw_errors = np.concatenate([m["absolute"].ravel()
                                 for m in error_matrices(raw, truth, mapping).values()])
    # Unrounded on purpose: rounding this to three places prints 0.0, which is the
    # same string the millimetre field prints and so says nothing more than it did.
    arms["exact"]["before_the_temporal_stage"]["max_micrometres"] = float(
        np.nanmax(raw_errors) * 1000.0)
    arms["exact"]["before_the_temporal_stage"]["what_it_is"] = (
        "the SAME run's raw triangulation -- reconstruct_multiview's fourth return, before "
        "the sequence solve, the fill and the Savitzky-Golay window. It splits the floor: "
        "whatever this is, geometry costs; whatever the difference is, the temporal stage "
        "costs. Coverage is its own, not the smoothed arm's, because NaNs are intact here. "
        "It comes back at machine precision -- see max_micrometres -- so on this fixture "
        "the ENTIRE floor is the temporal stage, and association and triangulation "
        "contribute nothing to it. Note what that does NOT say: the temporal stage is what "
        "recovers joints when a camera loses one, and there are no such frames here."
    )
    print(f"  exact: absolute median {arms['exact']['overall']['absolute']['median_mm']} mm, "
          f"p95 {arms['exact']['overall']['absolute']['p95_mm']} mm  "
          f"(raw triangulation, same run: "
          f"{arms['exact']['before_the_temporal_stage']['overall']['absolute']['median_mm']} mm "
          f"at coverage "
          f"{arms['exact']['before_the_temporal_stage']['overall']['absolute']['coverage']})")

    # --- noise ------------------------------------------------------------------
    per_seed = []
    for seed in range(args.seeds):
        noise = ResidualNoise(residual_pool, seed)
        records = observations_for(cameras, truth, noise=noise)
        noisy, _raw, diagnostics[f"noise_seed_{seed}"], seconds = run_arm(cameras, records)
        scored = score_arm(noisy, truth, resolve_mapping(noisy, mapping, f"noise_seed_{seed}"),
                           bootstrap=(seed == 0))
        scored["seed"] = seed
        per_seed.append(scored)
        print(f"  noise seed {seed}: absolute median "
              f"{scored['overall']['absolute']['median_mm']} mm, "
              f"p95 {scored['overall']['absolute']['p95_mm']} mm  ({seconds:.1f}s)")

    def across(basis: str, statistic: str) -> dict:
        values = [s["overall"][basis][statistic] for s in per_seed]
        return {"mean": round(float(np.mean(values)), 3),
                "sd": round(float(np.std(values, ddof=1)), 3) if len(values) > 1 else 0.0,
                "min": round(float(np.min(values)), 3),
                "max": round(float(np.max(values)), 3),
                "per_seed": values}

    arms["noise"] = {
        "what_it_is": (
            "the same projections displaced by MAMMA's own measured 2D residual "
            "(the dense pool above, scaled from 3840 to the 1280 working width), "
            "isotropic direction, i.i.d. per camera-frame-joint"
        ),
        "seeds": args.seeds,
        "note": (
            "The root-relative median is LARGER than the absolute one on this arm. That is "
            "arithmetic, not a defect: subtracting a root that carries its own error adds "
            "that error to every joint, and with i.i.d. 2D noise the root error is "
            "uncorrelated with each joint's. Absolute and root-relative are different "
            "questions and neither is the conservative one in general."
        ),
        "across_seeds": {basis: {statistic: across(basis, statistic)
                                 for statistic in ("median_mm", "p95_mm")}
                         for basis in ("absolute", "relative")},
        "per_seed": per_seed,
    }

    # --- controls ---------------------------------------------------------------
    controls: dict[str, dict] = {}

    crossed = {ours: 1 - body for ours, body in mapping.items()}
    controls["shuffled_subject_pairing"] = score_arm(positions, truth, crossed)
    controls["shuffled_subject_pairing"]["what_it_is"] = (
        "the exact arm's own output scored against the OTHER performer. Nothing in "
        "either output declares the order; this is the figure that says pairing by "
        "index would have been wrong."
    )
    controls["shuffled_subject_pairing"]["mapping"] = {
        f"our_{k}": f"body_id-{v:02d}" for k, v in sorted(crossed.items())}

    for shift in (+1, -1):
        records = time_shifted(exact_records, 0, shift)
        shifted, _raw, diagnostics[f"time_shift_{shift:+d}"], seconds = run_arm(cameras, records)
        control = score_arm(shifted, truth, resolve_mapping(shifted, mapping, f"time_shift_{shift:+d}"))
        control["what_it_is"] = (f"exact 2D with camera {CAMERAS[0]} advanced by "
                                 f"{shift:+d} frame; the other three untouched")
        controls[f"time_shift_{shift:+d}"] = control
        print(f"  time shift {shift:+d}: absolute median "
              f"{control['overall']['absolute']['median_mm']} mm  ({seconds:.1f}s)")

    frozen_truth = np.repeat(truth[:, :1], truth.shape[1], axis=1)
    records = observations_for(cameras, frozen_truth)
    frozen, _raw, diagnostics["frozen_skeleton"], seconds = run_arm(cameras, records)
    control = score_arm(frozen, truth, resolve_mapping(frozen, mapping, "frozen_skeleton"))
    control["what_it_is"] = (
        "frame 0's skeleton projected on every frame -- a constant. It reconstructs "
        "perfectly and agrees with nothing, which is the point: this is the figure a "
        "degenerate solution scores, and no band below it means anything."
    )
    controls["frozen_skeleton"] = control
    print(f"  frozen skeleton: absolute median "
          f"{control['overall']['absolute']['median_mm']} mm  ({seconds:.1f}s)")

    floor_abs = arms["exact"]["overall"]["absolute"]["median_mm"]
    floor_rel = arms["exact"]["overall"]["relative"]["median_mm"]
    for control in controls.values():
        control["ratio_to_oracle_floor"] = {
            "absolute": round(control["overall"]["absolute"]["median_mm"] / floor_abs, 2),
            "relative": round(control["overall"]["relative"]["median_mm"] / floor_rel, 2),
        }

    # --- report -----------------------------------------------------------------
    report = {
        "instrument": "I2 perfect-2D oracle",
        "generated_by": "tools/compare/oracle_2d.py",
        "regenerate": ".venv/bin/python tools/compare/oracle_2d.py",
        "reference": ("MAMMA's pred_joints, 15 body joints via PAIRS, 150 frames "
                      "[60,210) of pushing_and_lifting_from_ground -- the SAME reference "
                      "as the pose and delivered rungs, so it may share their axis"),
        "note": "agreement with an instrument, not accuracy; MAMMA never ships",
        "working_resolution_px": [WORKING_WIDTH, WORKING_HEIGHT],
        "joints_scored": list(SCORED),
        "joints_absent_from_the_map": sorted(set(cm.JOINT_NAMES) - set(SCORED)),
        "subject_correspondence": {f"our_{k}": f"body_id-{v:02d}"
                                   for k, v in sorted(mapping.items())},
        "subject_correspondence_source": ("tools/head/subject_map.py, from 3D pelvis "
                                          "agreement, asserted at a 5x margin"),
        "projection": convention,
        "arms": arms,
        "controls": controls,
        "controls_note": (
            "Every control's `ratio_to_oracle_floor` divides like against like -- absolute "
            "against the absolute floor, root-relative against the root-relative floor. "
            "The time shift is the weakest of the four and it is the interesting one: it "
            "fails, but by the smallest margin, and its root-relative ratio is somewhat "
            "below its absolute one, so part of what a one-frame sync error does is "
            "translate the body and a root-aligned gate sees less of it. Read the two "
            "ratios, not one of them; and note that this control costs about what "
            "MAMMA-grade 2D noise costs, so a rig one frame out of sync is not a "
            "second-order problem."
        ),
        "diagnostics": diagnostics,
        "diagnostics_note": (
            "`valid_joint_fraction` is 15/19 and `interpolated_joint_fraction` is 4/19 on "
            "EVERY arm: PAIRS maps 15 of our 19 joints, so the eyes and ears are absent by "
            "construction, exactly as they are absent from SOMA-77 on real footage. Neither "
            "figure moves between arms and neither says anything about reconstruction "
            "quality here; coverage on the 15 scored joints is 1.0 throughout. What does "
            "move: reprojection error (0 px exactly on both constant-2D arms -- the frozen "
            "skeleton reconstructs perfectly and agrees with nothing, which is why "
            "reprojection cannot be an accuracy gate), the association objective, and "
            "`contact_frames`, where the frozen arm reads [[150, 0], [150, 0]] -- one foot "
            "planted on every frame and the other never, a tell no accuracy metric gave."
        ),
        "subject_map_ambiguous_arms": list(AMBIGUOUS_MAP_ARMS),
        "inputs": {
            "camera_rig": {"path": str(RIG.relative_to(ROOT)), "sha256": sha256_of(RIG)},
            "pred_joints": {
                f"body_id-{b:02d}": sha256_of(MA3D / f"verts_joints_body_id-{b:02d}.npz")
                for b in (0, 1)},
            "ma_2d": {name: sha256_of(MA2D / f"{name}.npz") for name in CAMERAS},
            "verts_512": sha256_of(VERTS_512),
            "script": sha256_of(Path(__file__)),
        },
        "blind_to": (
            "Everything upstream of 2D. There are no images here, so this measures no "
            "detector error, no calibration error, no lens distortion, no sync error and "
            "no soft-tissue artefact -- and no joint-definition error at all, because the "
            "same 15 SMPL-X joints go in and come out, while joint-convention mismatch is "
            "the term that dominates every real-footage figure in this lane. It is blind "
            "to depth in the same way reprojection is: the cameras are the rig's four and "
            "a residual along a shared viewing ray is cheap. The noise arm is doubly "
            "conservative -- MAMMA's residual is regularised toward its own landmarks and "
            "so is a lower bound, and it is injected i.i.d., while a real detector's error "
            "is correlated across joints, across frames and across cameras, which is the "
            "kind an association gate and a temporal prior cannot filter. The oracle floor "
            "it reports is therefore a LOWER bound on what our pipeline costs, never an "
            "accuracy claim, and nothing here licenses a band on real footage."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    floor = arms["exact"]["overall"]["absolute"]["median_mm"]
    failures = [name for name, control in controls.items()
                if control["overall"]["absolute"]["median_mm"] <= floor * 3.0]
    if failures:
        print(f"CONTROL DID NOT FAIL: {failures} -- the instrument is not discriminating")
        return 1
    print(f"all {len(controls)} controls fail against the {floor} mm floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
