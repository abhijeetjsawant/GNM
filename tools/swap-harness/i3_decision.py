#!/usr/bin/env python3
"""I3 -- the decision rule, written down BEFORE any of the three reports was computed,
plus the pieces all three share so that they share a denominator.

WHY A SHARED FILE. `docs/LADDER_EXECUTION_PLAN.md` §2 (I3 row) says: *define the
decision rule for the next detector experiment before writing JSON.* A rule that lives
in three copies is three rules. This is the one copy; every report embeds it verbatim
under `decision_rule` and fills in the variables it can compute.

It also holds the one thing that makes the three reports comparable at all: the
detector's 2D is assigned to OUR subject slots by the REAL pipeline (a recording
associator wrapped around `associate_frame_graph`, inside a real
`reconstruct_multiview` call -- CLAUDE.md: *wrap the pipeline to instrument it; never
re-implement it*), and our subject is paired to MAMMA's `body_id` by pelvis agreement
through `tools/head/subject_map.py`, never by index.

RUN WITH THE SYSTEM PYTHON: `verts_512.pkl` unpickles torch tensors and only the
system interpreter has torch here. All three I3 instruments live under
`tools/swap-harness/` for that reason.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))

from autoanim_gnm import commercial_multiview as cm  # noqa: E402

CAMERAS = ("A001", "B001", "C001", "D001")
FIXTURE = "pushing_and_lifting_from_ground"
MAMMA_OUT = ROOT / "artifacts/mamma/mamma-4cam-five-second-v2/output"
MA3D = MAMMA_OUT / "ma_3d" / FIXTURE
MA2D = MAMMA_OUT / "ma_2d" / FIXTURE
RIG = ROOT / "artifacts/soma77-full/camera-rig.json"
WORK = ROOT / "artifacts/commercial-multiview-soma77/work"
RUN_REPORT = ROOT / "artifacts/commercial-multiview-soma77/run-report.json"
VERTS_512 = ROOT / ".cache/mamma/data/body_models/downsampled_verts/verts_512.pkl"

# The delivered detector runs at 1280 wide and every pixel-denominated gate inside
# the pipeline is quoted there. Native capture is 3840. Both widths appear in these
# reports and every pixel figure says which one it is in.
WORKING_WIDTH, WORKING_HEIGHT = 1280, 720
NATIVE_WIDTH = 3840
PX_1280_TO_3840 = NATIVE_WIDTH / WORKING_WIDTH

SAMPLE_RATE_HZ = 30
FRAME_WINDOW = (60, 210)

# our 19-joint contract name -> SMPL-X joint index in MAMMA's `pred_joints`.
# Copied from tools/compare/mamma_scoreboard.py and tools/compare/oracle_2d.py so all
# three instruments score the same joints against the same reference.
PAIRS = {
    "root": 0, "neck": 12, "nose": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
    "left_hip": 1, "right_hip": 2,
    "left_knee": 4, "right_knee": 5,
    "left_ankle": 7, "right_ankle": 8,
}

# THE SEMANTIC EXCLUSIONS. Every joint our 19-joint contract carries that must NOT be
# scored against MAMMA, and why. The reports count these rather than dropping them
# quietly, and report 3 keeps an arm WITH `nose` in it as the control that must be
# worse.
SEMANTIC_EXCLUSIONS = {
    "nose": "our `nose` is SOMA-77 index 6, the *Head skeletal joint* -- a point inside "
            "the skull, behind the eyes (CLAUDE.md). MAMMA's counterpart in PAIRS is "
            "SMPL-X joint 15, also a skull-interior head joint but of a DIFFERENT "
            "skeleton, so the pair carries a convention offset of centimetres that is "
            "not detector error. Excluded from the scored population; kept as a control.",
    "left_eye": "`pred_joints` DOES carry SMPL-X eye joints (23/24) and they move with "
                "the head, 82-91 mm from the head joint -- so the reason is not that there "
                "is nothing there. It is that an eyeball-CENTRE joint and SOMA-77's eye "
                "landmark are different points by a convention, the scoreboard's PAIRS "
                "never mapped them, and the locked-head fit holds the eyes at zero "
                "relative to the skull so they carry no independent information. Scoring "
                "them would price a convention offset as detector error.",
    "right_eye": "as left_eye: the joint exists and moves, but it is an eyeball centre "
                 "against a surface eye landmark, unmapped in PAIRS, and locked to the "
                 "skull by MAMMA's locked-head configuration.",
    "left_ear": "SOMA-77 has no ears; `left_ear`/`right_ear` are schema-only and are "
                "populated on zero frames (CLAUDE.md). Nothing to score.",
    "right_ear": "SOMA-77 has no ears either side; the contract slot exists and is "
                 "populated on zero frames, so there is nothing on our side to score.",
}
# What is actually scored against MAMMA: PAIRS minus the semantic exclusions.
SCORED_VS_MAMMA = tuple(name for name in PAIRS if name not in SEMANTIC_EXCLUSIONS)

# Moving-block bootstrap. CLAUDE.md: per-frame agreement in this lane has lag-1
# autocorrelation ~0.99, so ordinary resampling is invalid -- but *measure it, do not
# assume it*, which is why every report quotes its own series' lag-1.
BLOCK_FRAMES = 15          # half a second at 30 fps
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20260902


# ============================================================ THE DECISION RULE
# Fixed before the first figure was computed. The thresholds are set on the COST of
# each destination, not on the size of any measured effect -- that is what makes them
# preregisterable. Nothing here selects a shipped constant: destination (b) authorises
# an *experiment*, and the offsets it would use must be refitted against a MAMMA-free
# reference before anything ships (LADDER_EXECUTION_PLAN §4).

DECISION_RULE = {
    "written_before_any_figure_was_computed": True,
    "question": "what is the next 2D-detector experiment, and does the evidence pick it?",
    "destinations": {
        "a_pseudo_label_campaign": (
            "retrain or fine-tune the 2D detector on labels from our own footage. "
            "Expensive, and gated on lane H: labels triangulated through MAMMA's "
            "calibration would put MAMMA-derived data into weights "
            "(LADDER_EXECUTION_PLAN §4)."
        ),
        "b_per_camera_offset_fix": (
            "four 2D translation constants, one per camera, applied to the detector's "
            "output before triangulation. A coordinate-convention repair that costs "
            "nothing to apply -- but it is a SHIPPED CONSTANT, so it carries a "
            "provenance burden and must clear a floor to be worth it."
        ),
        "c_neither": (
            "the detector is not where the next millimetre is; the lane's effort "
            "belongs in delivery (lane D) instead."
        ),
    },
    "variables": {
        "R1_self_agreement_px1280": "report 1: median ONE-SIDED cross-view epipolar "
                                    "distance of our detector, reference-free, px at 1280",
        "R1_shuffled_px1280": "report 1 control: the same with the wrong cross-view partner",
        "R1_liveness_px_per_frame": "report 1: median per-frame 2D displacement; a frozen "
                                    "skeleton scores exactly 0",
        "R2_base_mm": "report 2: 3D median error of our triangulated joints against "
                      "MAMMA's projected pred_joints, no offset applied",
        "R2_heldout_mm": "report 2: the same after four per-camera offsets fitted on the "
                         "OTHER half of the frames",
        "R2_oracle_mm": "report 2: the same after removing the full per-(subject, frame, "
                        "camera) common mode fitted on the scored frames themselves -- "
                        "the CEILING of any coordinate-level fix",
        "G_heldout": "1 - R2_heldout_mm / R2_base_mm",
        "G_ceiling": "1 - R2_oracle_mm / R2_base_mm",
        "P_heldout": "P(held-out arm's median < baseline's median) on identical "
                     "moving-block bootstrap draws",
        "D_halves": "max over cameras of |offset_firsthalf - offset_secondhalf| divided "
                    "by the mean of the two magnitudes",
        "G_shuffled_cameras": "the same gain as G_heldout with the four offsets permuted "
                              "across cameras",
    },
    "thresholds": {
        "b_per_camera_offset_fix": {
            "all_of": [
                "G_heldout >= 0.10",
                "P_heldout >= 0.95",
                "D_halves <= 0.50",
                "G_shuffled_cameras < 0.5 * G_heldout",
            ],
            "why_these_numbers": (
                "0.10 because a shipped four-number constant that buys less than a tenth "
                "of the error is not worth the provenance burden it adds (I8); 0.95 "
                "because that is this lane's standing bar for a margin under the block "
                "bootstrap; 0.50 because an offset that changes by more than half its own "
                "size between the two halves of a five-second take is not static and a "
                "static constant cannot express it; the shuffled-camera clause because a "
                "gain a wrong-camera assignment reproduces is not a per-camera effect."
            ),
        },
        "a_pseudo_label_campaign": {
            "all_of": ["1 - G_ceiling >= 0.50"],
            "why_these_numbers": (
                "0.50 because pseudo-labelling is the expensive, lane-H-gated option and "
                "is only worth opening if MORE THAN HALF the detector's 3D cost survives "
                "the BEST POSSIBLE coordinate-level fix -- i.e. is per-joint error that "
                "no convention repair can reach."
            ),
        },
        "c_neither": {"all_of": ["neither (a) nor (b) triggered"]},
    },
    "not_exclusive": "a and b can both trigger; they are different experiments.",
    "what_this_rule_does_not_authorise": (
        "shipping the offsets measured in report 2. They are fitted against MAMMA "
        "projections and MAMMA never enters the delivery path (CLAUDE.md). Destination "
        "(b) authorises building the fix and re-selecting its four numbers from a "
        "MAMMA-free reference (lane H, or a documented crop convention)."
    ),
}

THRESHOLD_G_HELDOUT = 0.10
THRESHOLD_P_HELDOUT = 0.95
THRESHOLD_D_HALVES = 0.50
THRESHOLD_SHUFFLE_RATIO = 0.5
THRESHOLD_CEILING_SURVIVES = 0.50


def evaluate_decision_rule(*, g_heldout, p_heldout, d_halves, g_shuffled_cameras,
                           g_ceiling) -> dict:
    """Apply the rule above. Every clause is reported with its value, not just the verdict."""
    def _f(value):
        return None if value is None else float(value)

    g_heldout, p_heldout = _f(g_heldout), _f(p_heldout)
    d_halves, g_shuffled_cameras, g_ceiling = _f(d_halves), _f(g_shuffled_cameras), _f(g_ceiling)
    b_clauses = {
        "G_heldout >= 0.10": (g_heldout is not None and g_heldout >= THRESHOLD_G_HELDOUT),
        "P_heldout >= 0.95": (p_heldout is not None and p_heldout >= THRESHOLD_P_HELDOUT),
        "D_halves <= 0.50": (d_halves is not None and d_halves <= THRESHOLD_D_HALVES),
        "G_shuffled_cameras < 0.5 * G_heldout": (
            g_shuffled_cameras is not None and g_heldout is not None
            and g_shuffled_cameras < THRESHOLD_SHUFFLE_RATIO * g_heldout),
    }
    survives = None if g_ceiling is None else 1.0 - g_ceiling
    a_clauses = {"1 - G_ceiling >= 0.50": (survives is not None
                                           and survives >= THRESHOLD_CEILING_SURVIVES)}
    b = all(b_clauses.values())
    a = all(a_clauses.values())
    return {
        "values": {"G_heldout": g_heldout, "P_heldout": p_heldout, "D_halves": d_halves,
                   "G_shuffled_cameras": g_shuffled_cameras, "G_ceiling": g_ceiling,
                   "share_surviving_the_ceiling": survives},
        "b_per_camera_offset_fix": {"clauses": b_clauses, "triggered": bool(b)},
        "a_pseudo_label_campaign": {"clauses": a_clauses, "triggered": bool(a)},
        "verdict": ("a+b" if (a and b) else "b" if b else "a" if a else "c_neither"),
    }


# ==================================================================== small helpers

def sha256_of(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_shas(paths) -> dict:
    out = {}
    for path in paths:
        path = Path(path)
        try:
            rel = str(path.relative_to(ROOT))
        except ValueError:
            rel = str(path)
        out[rel] = sha256_of(path) if path.exists() else None
    return out


def lag1_autocorrelation(series: np.ndarray) -> float | None:
    """Lag-1 autocorrelation of a per-frame series, NaNs dropped pairwise.

    CLAUDE.md records 0.99 for the head lane's agreement series. That is a measurement
    of a different series, so every report here measures its own and quotes it rather
    than inheriting the number.
    """
    values = np.asarray(series, dtype=np.float64)
    a, b = values[:-1], values[1:]
    keep = np.isfinite(a) & np.isfinite(b)
    if int(keep.sum()) < 3:
        return None
    a, b = a[keep], b[keep]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def block_draws(frames: int, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS,
                block: int = BLOCK_FRAMES) -> list[np.ndarray]:
    """Moving-block frame index draws, generated ONCE so every arm is scored on the
    identical draws. Pairing the arms on the same draws is what attributes the
    fragility to the right term (CLAUDE.md)."""
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(frames / block))
    high = max(frames - block, 0)
    out = []
    for _ in range(draws):
        starts = rng.integers(0, high + 1, blocks)
        out.append(np.concatenate([np.arange(s, s + block) for s in starts])[:frames])
    return out


def paired_block_bootstrap(per_frame: dict[str, np.ndarray], baseline: str,
                           candidates, *, seed: int = BOOTSTRAP_SEED) -> dict:
    """Median of every arm on IDENTICAL moving-block draws, and P(candidate < baseline).

    `per_frame[arm]` is [frame, ...] -- everything after the frame axis is pooled.
    """
    frames = per_frame[baseline].shape[0]
    draws = block_draws(frames, seed=seed)
    medians: dict[str, list[float]] = {name: [] for name in per_frame}
    for index in draws:
        for name, matrix in per_frame.items():
            sample = matrix[index].ravel()
            sample = sample[np.isfinite(sample)]
            medians[name].append(float(np.median(sample)) if sample.size else np.nan)
    out = {"block_frames": BLOCK_FRAMES, "draws": len(draws), "arms": {}}
    base = np.asarray(medians[baseline])
    for name, values in medians.items():
        values = np.asarray(values)
        entry = {
            "median_of_draws": _round(np.nanmedian(values)),
            "ci95": [_round(np.nanpercentile(values, 2.5)), _round(np.nanpercentile(values, 97.5))],
        }
        if name in candidates:
            keep = np.isfinite(values) & np.isfinite(base)
            entry["P_beats_baseline"] = _round(float(np.mean(values[keep] < base[keep])), 4)
        out["arms"][name] = entry
    return out


def _round(value, places: int = 3):
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else round(value, places)


# ============================================== the detector's 2D, on our subject slots

def load_records() -> list[list[dict]]:
    return [cm.load_observation_jsonl(WORK / f"{name}-soma77-observations.jsonl")
            for name in CAMERAS]


def working_cameras():
    return tuple(camera.scaled(WORKING_WIDTH, WORKING_HEIGHT)
                 for camera in cm.load_camera_rig(RIG))


def assigned_detector_2d():
    """[subject, frame, camera, 19, 3] of our detector's own 2D, on OUR subject slots.

    Obtained by WRAPPING the real pipeline: a recording associator around
    `associate_frame_graph` inside a real `reconstruct_multiview` call. The rows the
    associator hands back are the very rows it was given, so this is the pipeline's own
    assignment and not a second copy of its association loop -- a hand replication of
    that loop drifted 9-19 mm from the retained tracks (CLAUDE.md).

    Returns (assigned, positions, raw, diagnostics). `positions` reproduces the retained
    `subject-XX.body-track.npz` triangulated positions to ~1e-5 mm, which is the check
    that the wrap is the pipeline and not a re-implementation.
    """
    cameras = working_cameras()
    records = load_records()
    captured: list[np.ndarray] = []

    def recording_associator(*args, **kwargs):
        result = cm.associate_frame_graph(*args, **kwargs)
        captured.append(np.array(result[0], copy=True))
        return result

    _tracks, diagnostics, positions, raw = cm.reconstruct_multiview(
        cameras, records, subject_count=2, sample_rate_hz=SAMPLE_RATE_HZ,
        associator=recording_associator,
    )
    assigned = np.stack(captured, axis=1)          # [subject, frame, camera, 19, 3]
    return assigned, positions, raw, diagnostics


def mamma_pred_joints() -> np.ndarray:
    """[body_id, frame, 127, 3]. `pred_joints`, never `gt_joints` -- the latter is a
    byte-copy of the former (CLAUDE.md)."""
    return np.stack([
        np.load(MA3D / f"verts_joints_body_id-{body:02d}.npz", allow_pickle=True)["pred_joints"]
        .astype(np.float64)
        for body in (0, 1)
    ])


def subject_mapping(positions: np.ndarray) -> dict[int, int]:
    """our subject index -> MAMMA body_id, from pelvis agreement. Never by index."""
    import os
    from subject_map import mamma_index_for  # noqa: E402
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return mamma_index_for(positions)
    finally:
        os.chdir(cwd)


def project_joints(cameras, points: np.ndarray) -> np.ndarray:
    """[..., 3] world -> [..., 2] pixels per camera; NaN where the point is behind."""
    flat = points.reshape(-1, 3)
    out = np.full((len(cameras), flat.shape[0], 2), np.nan)
    for index, camera in enumerate(cameras):
        uv, depth = camera.project(flat)
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        depth = np.asarray(depth, dtype=np.float64).reshape(-1)
        keep = depth > 0.0
        out[index, keep] = uv[keep]
    return out.reshape((len(cameras),) + points.shape[:-1] + (2,))


def population_header(extra: dict | None = None) -> dict:
    base = {
        "fixture": FIXTURE,
        "retained_run": "artifacts/mamma/mamma-4cam-five-second-v2",
        "cameras": list(CAMERAS),
        "frames": f"{FRAME_WINDOW[0]}..{FRAME_WINDOW[1]} of the source, 150 frames, 30 fps",
        "subjects": 2,
        "our_detector": "nvidia_gemx_soma77 at 1280x720, 17 of the 19-joint contract "
                        "populated (ears are schema-only and never emitted)",
        "subject_pairing": "our subject slots come from the real pipeline's own "
                           "association (recording associator around "
                           "associate_frame_graph); our subject -> MAMMA body_id comes "
                           "from pelvis agreement via tools/head/subject_map.py, never "
                           "from the index (CLAUDE.md: body_id-00 is our subject 1).",
        "statistics": f"moving-block bootstrap, block {BLOCK_FRAMES} frames "
                      f"(0.5 s), {BOOTSTRAP_DRAWS} draws, every arm on identical draws; "
                      "each report quotes the measured lag-1 autocorrelation of its own "
                      "series rather than assuming 0.99.",
    }
    if extra:
        base.update(extra)
    return base


if __name__ == "__main__":
    import json
    print(json.dumps(DECISION_RULE, indent=2))
