"""D4b: the drawn-set rule, computed and FROZEN before any fit (the card, `docs/LADDER_EXECUTION_PLAN.md` §2).

The rule, verbatim in substance: take each of D4's ten named channels to either end of its configured
limit on the truth body, pose held, at every donor frame, and read the largest displacement of any of the
17 mapped joints (a) RAW and (b) POSE-ORTHOGONAL -- the least-squares residual after projecting that
displacement onto the pose Jacobian's column space at that frame (the 26 `*_flexible` channels excluded).
A channel is DRAWN iff (b)'s median over frames exceeds 2 mm on BOTH donors.

Frozen here, before any fresh evaluation (Astra finding 3):
  * the evaluation identity: the ZERO identity (every `scale_*` 0), on the clamped donor pose;
  * the endpoint aggregation: the max over the 17 mapped joints of the per-joint displacement norm, then
    the max over the two ends of the configured limit, then the median over frames;
  * the rank convention: singular values above 1e-6 of the largest.

The pose Jacobian is central finite differences of the 17 mapped joints against every non-flexible POSE
parameter of the 178-parameter character (110), at step `FD_STEP` in each parameter's own units, float64.

Also frozen here, because the gate scores only segments a drawn channel moves: which of the card's twelve
rest segments each channel MOVES, read at MHR's zero pose with the zero identity, each channel at either end
of its limit; a channel moves a segment iff the rest length changes by more than `SEGMENT_MOVE_MM`.

The rank and condition number of the drawn set are a LOCAL diagnostic (Astra finding 3), never a threshold:
individually surviving columns need not be jointly identifiable.

This module's top level is momentum-free (constants and the projection arithmetic) so the gate's tests import
it on `.venv`; the measurement itself runs on `/tmp/momenv/bin/python`:

    /tmp/momenv/bin/python tools/fitter/d4b_identifiability.py --out JSON [--frozen-copy JSON]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# The card's population: six NEW seeds per donor, two donors, twelve fixtures.
SEEDS = (20261001, 20261002, 20261003, 20261004, 20261005, 20261006)
DONOR_INDICES = (0, 1)
DONOR_FILES = {0: "artifacts/compare/d4-body/fitted/motion_0.npz",
               1: "artifacts/compare/d4-body/fitted/motion_1.npz"}
ARMS = ("oracle", "exact_identity", "mean_body", "spine_displaced", "warm", "converged")
FRAMES = 150

# The 17 mapped MHR joints in `mhr_delivery.MAP` order (asserted equal at run time in momentum).
MAPPED_JOINTS = ("root", "c_neck", "c_head", "l_uparm", "r_uparm", "l_lowarm", "r_lowarm",
                 "l_wrist", "r_wrist", "l_upleg", "r_upleg", "l_lowleg", "r_lowleg",
                 "l_foot", "r_foot", "l_eye", "r_eye")

# The card's twelve rest segments between mapped joints.
SEGMENTS = {
    "trunk": ("root", "c_neck"),
    "neck_head": ("c_neck", "c_head"),
    "shoulder_width": ("l_uparm", "r_uparm"),
    "l_upper_arm": ("l_uparm", "l_lowarm"),
    "r_upper_arm": ("r_uparm", "r_lowarm"),
    "l_forearm": ("l_lowarm", "l_wrist"),
    "r_forearm": ("r_lowarm", "r_wrist"),
    "hip_width": ("l_upleg", "r_upleg"),
    "l_thigh": ("l_upleg", "l_lowleg"),
    "r_thigh": ("r_upleg", "r_lowleg"),
    "l_shin": ("l_lowleg", "l_foot"),
    "r_shin": ("r_lowleg", "r_foot"),
}

# D4's ten named channels (`d4_o1_exactness.CHANNELS`; asserted equal at run time).
CHANNELS = ("scale_spine_length", "scale_neck_length", "scale_shoulder_width", "scale_uparms",
            "scale_lowarms", "scale_hip_width", "scale_hip_height", "scale_uplegs",
            "scale_lowlegs", "scale_foot_length")

THRESHOLD_MM = 2.0            # the card: above every per-joint pose floor D4 measured (max 1.64 mm)
RANK_RTOL = 1e-6              # singular values above 1e-6 of the largest (the card's rank convention;
                              # applied to the pose Jacobian's column space and to the drawn set's columns)
SENSITIVITY_RTOLS = (1e-4, 1e-3)   # REPORTED only: the drawn set under coarser pose-Jacobian cuts
FD_STEP = 1e-3                # central differences, each parameter in its own units
SEGMENT_MOVE_MM = 0.1         # a channel "moves" a segment iff its rest length changes by more
SPINE_DISPLACEMENT = 0.149    # must-fail (ii): the rounded minimum of D4's measured shrink (0.148777)


def per_joint_max_mm(displacement_cm: np.ndarray) -> float:
    """The frozen endpoint aggregation: max over the 17 joints of the per-joint norm (cm in, mm out)."""
    return float(np.linalg.norm(np.asarray(displacement_cm).reshape(-1, 3), axis=1).max() * 10.0)


def column_basis(jacobian: np.ndarray, rtol: float = RANK_RTOL) -> tuple[np.ndarray, np.ndarray]:
    """An orthonormal basis of `jacobian`'s column space under the frozen rank convention.

    The left singular vectors whose singular values exceed `rtol` of the largest. Measured before the
    freeze (2026-09-24): the finite-difference pose Jacobian's spectrum has a clean gap -- 36 values at or
    above ~2e-5 of the largest, then values of 1e-9..1e-11 that scale with the step squared (truncation
    error of genuinely null directions), then 1e-14 round-off -- so 1e-6 sits inside the gap. Plain
    `numpy.linalg.lstsq(rcond=None)` cuts at machine epsilon and projects onto those truncation-error
    directions too, which is a numerical artefact, not a pose.
    """
    u, singular, _ = np.linalg.svd(np.asarray(jacobian, np.float64), full_matrices=False)
    keep = singular > rtol * singular[0] if singular.size and singular[0] > 0 else np.zeros(0, bool)
    return u[:, keep], singular[keep]


def pose_orthogonal(displacement: np.ndarray, jacobian: np.ndarray, rtol: float = RANK_RTOL) -> np.ndarray:
    """The least-squares residual of `displacement` after projection onto `jacobian`'s column space.

    Works on a flattened 3J vector (or a 3J x k matrix of columns). The column space is the one the
    frozen rank convention keeps (`column_basis`).
    """
    basis, _ = column_basis(jacobian, rtol)
    displacement = np.asarray(displacement, np.float64)
    return displacement - basis @ (basis.T @ displacement)


def absorbing_step_norm(displacement: np.ndarray, jacobian: np.ndarray, rtol: float = RANK_RTOL) -> float:
    """REPORTED, never ruled on: the norm of the minimum-norm pose step (parameter units) whose
    first-order effect is the projected displacement. A column-space membership test cannot see how
    far the pose would have to move; this is that distance."""
    u, singular, _ = np.linalg.svd(np.asarray(jacobian, np.float64), full_matrices=False)
    keep = singular > rtol * singular[0]
    coefficients = (u[:, keep].T @ np.asarray(displacement, np.float64)) / singular[keep]
    return float(np.linalg.norm(coefficients))


def rank_and_condition(columns: np.ndarray, rtol: float = RANK_RTOL) -> dict:
    """Numerical rank (singular values above `rtol` of the largest) and the condition number."""
    singular = np.linalg.svd(np.asarray(columns, np.float64), compute_uv=False)
    rank = int((singular > rtol * singular[0]).sum()) if singular.size and singular[0] > 0 else 0
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else float("inf")
    return {"singular_values": [float(s) for s in singular], "rank": rank,
            "columns": int(np.asarray(columns).shape[1]), "condition_number": condition}


def rule(orthogonal_median_mm: dict[int, float], threshold_mm: float = THRESHOLD_MM) -> bool:
    """DRAWN iff the pose-orthogonal median exceeds the threshold on BOTH donors."""
    return all(orthogonal_median_mm[d] > threshold_mm for d in DONOR_INDICES)


def displaced_spine(value: float) -> float:
    """Must-fail (ii): `value` displaced 0.149 toward zero, THROUGH zero if the draw is smaller.

    An undrawn spine (truth exactly 0) has no direction toward zero; the displacement is then -0.149
    (a declared direction), because (ii) is unconditional and may never be skipped.
    """
    direction = -np.sign(value) if value != 0.0 else -1.0
    return float(value + direction * SPINE_DISPLACEMENT)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(lod: int = 2) -> dict:
    """Run on momenv. Returns the drawn-set record (the frozen JSON's content)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib.metadata as metadata

    import pymomentum.geometry as g

    import d4_o1_exactness as d4
    from mhr_delivery import MAP, load_character

    if tuple(MAP.values()) != MAPPED_JOINTS:
        raise SystemExit("MAPPED_JOINTS is not mhr_delivery.MAP's joint order")
    if tuple(d4.CHANNELS) != CHANNELS:
        raise SystemExit("CHANNELS is not D4's ten named channels")
    limits = d4.configured_limits()
    full = load_character(lod)
    simple = load_character(lod, drop_flexible=True)
    full_names = list(full.parameter_transform.names)
    names = list(simple.parameter_transform.names)
    pose_mask = np.asarray(simple.parameter_transform.pose_parameters, bool)
    pose_index = np.flatnonzero(pose_mask)
    if any(names[i].endswith("_flexible") or names[i].startswith("scale_") for i in pose_index):
        raise SystemExit("the pose set is not the non-flexible pose parameters")
    skeleton = list(simple.skeleton.joint_names)
    rows = [skeleton.index(j) for j in MAPPED_JOINTS]
    column = {c: names.index(c) for c in CHANNELS}

    def fk(params: np.ndarray) -> np.ndarray:
        """[..., 178] -> [..., 17*3] mapped joint positions, cm, float64."""
        state = np.asarray(g.model_parameters_to_skeleton_state(simple, np.asarray(params, np.float64)))
        return state[..., rows, :3].reshape(*params.shape[:-1], len(rows) * 3)

    # Segment -> channel map, at MHR's zero pose with the zero identity.
    def lengths_mm(params: np.ndarray) -> dict[str, float]:
        positions = fk(params).reshape(len(rows), 3)
        at = {j: positions[i] for i, j in enumerate(MAPPED_JOINTS)}
        return {s: float(np.linalg.norm(at[a] - at[b]) * 10.0) for s, (a, b) in SEGMENTS.items()}

    zero = np.zeros(len(names), np.float64)
    base_lengths = lengths_mm(zero)
    segment_change = {}
    for c in CHANNELS:
        low, high = limits[c]
        change = {s: 0.0 for s in SEGMENTS}
        for end in (low, high):
            p = zero.copy()
            p[column[c]] = end
            moved = lengths_mm(p)
            for s in SEGMENTS:
                change[s] = max(change[s], abs(moved[s] - base_lengths[s]))
        segment_change[c] = {s: round(v, 6) for s, v in change.items()}
    moves = {s: [c for c in CHANNELS if segment_change[c][s] > SEGMENT_MOVE_MM] for s in SEGMENTS}

    per_donor = {}
    for donor in DONOR_INDICES:
        d4.DONOR = ROOT / DONOR_FILES[donor]
        # The clamped donor pose with the ZERO identity: every drawn limit collapsed to (0, 0), so
        # D4's own `truth_motion` (import, never copy) applies the fixture's clamp and draws nothing.
        zero_limits = dict(limits)
        for c in CHANNELS:
            zero_limits[c] = (0.0, 0.0)
        motion_full, drawn, repair = d4.truth_motion(full, np.random.default_rng(0), zero_limits)
        if any(v != 0.0 for v in drawn.values()):
            raise SystemExit("the evaluation identity is not zero")
        motion = np.stack([motion_full[:, full_names.index(n)] for n in names], axis=1).astype(np.float64)
        frames = motion.shape[0]
        raw = {c: np.zeros(frames) for c in CHANNELS}
        orth = {c: np.zeros(frames) for c in CHANNELS}
        raw_end = {c: {"low": np.zeros(frames), "high": np.zeros(frames)} for c in CHANNELS}
        orth_end = {c: {"low": np.zeros(frames), "high": np.zeros(frames)} for c in CHANNELS}
        derivative_columns = {c: [] for c in CHANNELS}
        derivative_columns_raw = {c: [] for c in CHANNELS}
        jacobian_rank = []
        step = {c: np.zeros(frames) for c in CHANNELS}
        orth_sensitivity = {rt: {c: np.zeros(frames) for c in CHANNELS} for rt in SENSITIVITY_RTOLS}
        orth_lstsq_default = {c: np.zeros(frames) for c in CHANNELS}
        for f in range(frames):
            p = motion[f]
            base = fk(p[None])[0]
            steps = np.zeros((2 * len(pose_index), len(names)))
            for k, i in enumerate(pose_index):
                steps[2 * k] = p
                steps[2 * k, i] += FD_STEP
                steps[2 * k + 1] = p
                steps[2 * k + 1, i] -= FD_STEP
            positions = fk(steps)
            jacobian = ((positions[0::2] - positions[1::2]) / (2 * FD_STEP)).T   # (51, 110)
            jacobian_rank.append(int(column_basis(jacobian)[0].shape[1]))
            for c in CHANNELS:
                low, high = limits[c]
                for label, end in (("low", low), ("high", high)):
                    q = p.copy()
                    q[column[c]] = end
                    d = fk(q[None])[0] - base
                    raw_end[c][label][f] = per_joint_max_mm(d)
                    orth_end[c][label][f] = per_joint_max_mm(pose_orthogonal(d, jacobian))
                    step[c][f] = max(step[c][f], absorbing_step_norm(d, jacobian))
                    for rt in SENSITIVITY_RTOLS:
                        orth_sensitivity[rt][c][f] = max(orth_sensitivity[rt][c][f],
                                                         per_joint_max_mm(pose_orthogonal(d, jacobian, rt)))
                    coefficients, *_ = np.linalg.lstsq(jacobian, d, rcond=None)
                    orth_lstsq_default[c][f] = max(orth_lstsq_default[c][f],
                                                   per_joint_max_mm(d - jacobian @ coefficients))
                raw[c][f] = max(raw_end[c]["low"][f], raw_end[c]["high"][f])
                orth[c][f] = max(orth_end[c]["low"][f], orth_end[c]["high"][f])
                # local derivative at the evaluation identity, for the rank diagnostic
                plus, minus = p.copy(), p.copy()
                plus[column[c]] += FD_STEP
                minus[column[c]] -= FD_STEP
                derivative = (fk(plus[None])[0] - fk(minus[None])[0]) / (2 * FD_STEP)
                derivative_columns_raw[c].append(derivative)
                derivative_columns[c].append(pose_orthogonal(derivative, jacobian))
        per_donor[donor] = {
            "donor_file_sha256": _sha256(ROOT / DONOR_FILES[donor]),
            "fixture_repair_clamped_parameter_frames": repair["clamped_parameter_frames"],
            "frames": frames,
            "pose_jacobian_columns": int(len(pose_index)),
            "pose_jacobian_shape": [len(rows) * 3, int(len(pose_index))],
            "pose_jacobian_rank_min_median_max_over_frames_at_the_rank_convention": [int(np.min(jacobian_rank)), float(np.median(jacobian_rank)), int(np.max(jacobian_rank))],
            "raw_median_mm": {c: round(float(np.median(raw[c])), 6) for c in CHANNELS},
            "pose_orthogonal_median_mm": {c: round(float(np.median(orth[c])), 6) for c in CHANNELS},
            "raw_median_mm_by_end": {c: {e: round(float(np.median(raw_end[c][e])), 6)
                                         for e in ("low", "high")} for c in CHANNELS},
            "pose_orthogonal_median_mm_by_end": {c: {e: round(float(np.median(orth_end[c][e])), 6)
                                                     for e in ("low", "high")} for c in CHANNELS},
            "pose_orthogonal_p05_p95_mm": {c: [round(float(np.percentile(orth[c], 5)), 6),
                                               round(float(np.percentile(orth[c], 95)), 6)]
                                           for c in CHANNELS},
            "REPORTED_absorbing_pose_step_norm_median_over_frames": {
                c: round(float(np.median(step[c])), 4) for c in CHANNELS},
            "REPORTED_pose_orthogonal_median_mm_at_coarser_cuts": {
                f"{rt:g}": {c: round(float(np.median(orth_sensitivity[rt][c])), 6) for c in CHANNELS}
                for rt in SENSITIVITY_RTOLS},
            "REPORTED_pose_orthogonal_median_mm_numpy_lstsq_default_ARTEFACT": {
                c: round(float(np.median(orth_lstsq_default[c])), 6) for c in CHANNELS},
            "_columns": {c: np.concatenate(derivative_columns[c]) for c in CHANNELS},
            "_columns_raw": {c: np.concatenate(derivative_columns_raw[c]) for c in CHANNELS},
        }

    drawn, excluded = [], {}
    for c in CHANNELS:
        orth_by_donor = {d: per_donor[d]["pose_orthogonal_median_mm"][c] for d in DONOR_INDICES}
        raw_by_donor = {d: per_donor[d]["raw_median_mm"][c] for d in DONOR_INDICES}
        low, high = limits[c]
        if rule(orth_by_donor):
            drawn.append(c)
        elif low == high:
            excluded[c] = "UNDRAWABLE: the configured limit is the point [%g, %g]" % (low, high)
        elif any(raw_by_donor[d] <= THRESHOLD_MM for d in DONOR_INDICES):
            excluded[c] = ("UNSEEN: the raw displacement of every mapped joint is at or below the threshold "
                           "(nothing the 17 landmarks see moves with it)")
        else:
            # Not a pre-registered category (UNSEEN and UNDRAWABLE are): the landmarks see the channel
            # raw, and the frozen first-order column-space test finds it inside the pose's span.
            excluded[c] = ("BELOW THE RULE: the raw displacement exceeds the threshold, the pose-orthogonal "
                           "residual at the frozen rank convention does not")

    diagnostics = {}
    for donor in DONOR_INDICES:
        for kind, key in (("pose_orthogonal", "_columns"), ("raw", "_columns_raw")):
            matrix = np.stack([per_donor[donor][key][c] for c in drawn], axis=1) if drawn else np.zeros((1, 0))
            record = rank_and_condition(matrix) if drawn else {"rank": 0, "columns": 0}
            unit = matrix / np.linalg.norm(matrix, axis=0, keepdims=True) if drawn else matrix
            cosine = {}
            for a in range(len(drawn)):
                for b in range(a + 1, len(drawn)):
                    value = float(unit[:, a] @ unit[:, b])
                    if abs(value) > 0.3:
                        cosine[f"{drawn[a]}~{drawn[b]}"] = round(value, 4)
            if "scale_spine_length" in drawn and "scale_neck_length" in drawn:
                a, b = drawn.index("scale_spine_length"), drawn.index("scale_neck_length")
                record["spine_neck_cosine"] = round(float(unit[:, a] @ unit[:, b]), 6)
            record["column_cosines_above_0.3"] = cosine
            diagnostics[f"donor_{donor}_{kind}"] = record
        for key in ("_columns", "_columns_raw"):
            per_donor[donor].pop(key)

    model = ROOT / ".cache/mhr/assets/compact_v6_1.model"
    return {
        "step": "D4b", "stage": 2,
        "frozen_before_any_fit": True,
        "rule": ("a channel is DRAWN iff the median over frames of the POSE-ORTHOGONAL max-over-17-joints "
                 "displacement, the channel at either end of its configured limit on the truth body (pose "
                 "held, the zero identity), exceeds 2 mm on BOTH donors"),
        "frozen": {
            "evaluation_identity": "zero (every scale_* channel 0) on the clamped donor pose, flexible channels 0",
            "endpoint_aggregation": "max over the 17 mapped joints of the per-joint displacement norm; then the "
                                    "max over the two ends of the configured limit; then the median over frames",
            "rank_convention": f"singular values above {RANK_RTOL:g} of the largest",
            "threshold_mm": THRESHOLD_MM,
            "pose_jacobian": f"central finite differences, step {FD_STEP:g} in each parameter's own units, "
                             "float64, against every non-flexible pose parameter of the 178-parameter character",
            "projection": ("residual d - U U^T d, U the left singular vectors of J with singular values above "
                           f"{RANK_RTOL:g} of the largest (the card's rank convention applied to the pose Jacobian)"),
            "projection_decided_before_the_freeze": (
                "a first run used numpy.linalg.lstsq(rcond=None), which cuts at machine epsilon and projects onto "
                "finite-difference truncation directions (singular values 1e-9..1e-11 of the largest, scaling with "
                "the step squared). Found and replaced before this file was frozen and before any fit; it read "
                "the same drawn set. Its readings are kept below as REPORTED_..._ARTEFACT."),
            "segment_move_mm": SEGMENT_MOVE_MM,
            "spine_displacement_units": SPINE_DISPLACEMENT,
        },
        "channels_considered": list(CHANNELS),
        "configured_limits": {c: list(limits[c]) for c in CHANNELS},
        "drawn_set": drawn,
        "REPORTED_drawn_set_at_coarser_pose_jacobian_cuts_never_selecting": {
            f"{rt:g}": [c for c in CHANNELS if rule({d: per_donor[d]["REPORTED_pose_orthogonal_median_mm_at_coarser_cuts"][f"{rt:g}"][c]
                                                     for d in DONOR_INDICES})]
            for rt in SENSITIVITY_RTOLS},
        "excluded": excluded,
        "prediction_from_the_card": "eight drawn; scale_foot_length UNSEEN, scale_hip_height UNDRAWABLE",
        "card_disposition_if_the_spine_falls_below_the_rule": (
            "the card: a finding about the INPUT, not a pass for the fitter; reported as such and handed to the "
            "integration step's Spine1 feed; exclusion never skips must-fail (ii), which is unconditional; a trunk "
            "that cannot be scored and rejected is not a demonstrated trunk failure, and such a run cannot close D4"),
        "per_donor": {str(d): per_donor[d] for d in DONOR_INDICES},
        "rank_and_condition_of_the_drawn_set": diagnostics,
        "rank_note": "a LOCAL diagnostic at the zero identity, the drawn channels' derivative columns stacked over "
                     "all frames; never a threshold. Individually surviving columns need not be jointly identifiable.",
        "segments": {s: list(pair) for s, pair in SEGMENTS.items()},
        "segment_rest_length_change_mm_by_channel": segment_change,
        "segment_moved_by": moves,
        "segments_scored": [s for s in SEGMENTS if any(c in drawn for c in moves[s])],
        "segments_reported_not_scored": [s for s in SEGMENTS if not any(c in drawn for c in moves[s])],
        "zero_identity_rest_lengths_mm": {s: round(v, 4) for s, v in base_lengths.items()},
        "provenance": {
            "pymomentum": metadata.version("pymomentum-cpu"),
            "model_sha256": _sha256(model),
            "fbx_sha256": _sha256(ROOT / f".cache/mhr/assets/lod{lod}.fbx"),
            "this_file_sha256": _sha256(Path(__file__)),
        },
        "lod": lod,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen-copy", type=Path)
    parser.add_argument("--lod", type=int, default=2)
    arguments = parser.parse_args()
    record = measure(arguments.lod)
    text = json.dumps(record, indent=1)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(text, encoding="utf-8")
    if arguments.frozen_copy:
        arguments.frozen_copy.write_text(text, encoding="utf-8")
    print("drawn:", record["drawn_set"])
    print("excluded:", json.dumps(record["excluded"], indent=1))
    for d in DONOR_INDICES:
        row = record["per_donor"][str(d)]
        print(f"donor {d}: " + ", ".join(f"{c.replace('scale_', '')} raw {row['raw_median_mm'][c]:.2f} "
                                         f"orth {row['pose_orthogonal_median_mm'][c]:.2f}" for c in CHANNELS))
    print("rank/condition:", json.dumps({k: {x: v[x] for x in ("rank", "columns", "condition_number")
                                             if x in v} for k, v in record["rank_and_condition_of_the_drawn_set"].items()}))
    print("segments scored:", record["segments_scored"], "reported:", record["segments_reported_not_scored"])
    print("moved by:", json.dumps(record["segment_moved_by"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
