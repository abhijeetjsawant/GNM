"""D4c precondition 0: the drawn set by a LIMIT-AWARE rule, computed and FROZEN before any fit.

The card is the "D4c the calibration's start" row of `docs/LADDER_EXECUTION_PLAN.md` §2. D4b's rule was a
first-order column-space test that could not see how far the pose would have to move; it dropped the spine
through absorbing pose steps of 17-65 units against configured limits of at most 1.5 (CLAUDE.md, D4b). This
rule asks the question with the limits in it:

  For each of D4's ten named channels at each end of its configured limit, per donor frame: the least-squares
  pose correction delta WITHIN THE CONFIGURED LIMITS (`scipy.optimize.lsq_linear`, bounds = limits - donor
  pose, flexible channels excluded) minimising ||J_pose . delta - d_c|| over the 17 mapped joints. The channel
  is DRAWN iff the residual's median over frames exceeds 2 mm on BOTH donors.

Frozen here, before any fit (the card, and Astra's card-review finding 2):
  * the evaluation identity: ZERO (every `scale_*` 0) on the clamped donor pose (D4's `truth_motion`, imported);
  * the displacement d_c = FK(p with the channel at the limit end) - FK(p), the 17 mapped joints, cm, float64;
  * the Jacobian: central finite differences of the same 17 joints, step `FD_STEP` (D4b's) in each parameter's
    own units, against every non-flexible POSE parameter of the 178-parameter character (110 columns);
  * the sign convention: delta is the pose step that REPRODUCES d_c (J delta ~ d_c), i.e. the pose a
    wrong-identity fit would move to; its bounds are [lo - p, hi - p] so p + delta stays within the limits;
  * a pose parameter with no configured limit is UNBOUNDED; a FIXED one (configured limit a point, lo == hi) has
    zero width, and `lsq_linear` refuses lb == ub, so its COLUMN IS REMOVED (delta = 0 by construction);
  * the solver: `lsq_linear(method="bvls", tol=1e-10, max_iter=10000)`; any frame whose solve does not report
    success (status <= 0) is a SOLVER FAILURE and the step STOPS -- never a silent drop;
  * the aggregation, D4b's in full: the max over the 17 joints of the per-joint residual norm, then the max
    over the two limit ends, then the median over frames; NO rounding before comparison;
  * the 0.1 mm segment-movement cut (a channel moves a segment iff its rest length changes by more, at either
    end, at MHR's zero pose with the zero identity).

REPORTED beside it, never ruling: the UNBOUNDED projection (D4b's pose-orthogonal residual at D4b's rank
convention, `d4b_identifiability.pose_orthogonal`), the bounded solve's step norm, and how many columns sat on a
bound.

Also computed here, because the calibration's starting identity needs them (the card: "computed afresh"): each
channel's SIGNED rest-length change per unit, k_c = (l_s(c = upper limit) - l0_s) / upper, the lower-end slope
for the linearity check, and l0_s, the zero-identity rest length. These are the values the fitter's
`landmark_start` recomputes at run time from its own character; the gate cross-checks them.

The top level is momentum-free (constants and arithmetic) so the gate's tests import it on `.venv`; the
measurement runs on `/tmp/momenv/bin/python`:

    /tmp/momenv/bin/python tools/fitter/d4c_identifiability.py --out JSON [--frozen-copy JSON]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d4b_identifiability import (CHANNELS, DONOR_FILES, DONOR_INDICES, FD_STEP,  # noqa: E402
                                 MAPPED_JOINTS, SEGMENTS, THRESHOLD_MM, per_joint_max_mm)

ROOT = Path(__file__).resolve().parents[2]

SEGMENT_MOVE_MM = 0.1
LSQ_METHOD = "bvls"
LSQ_TOL = 1e-10
LSQ_MAX_ITER = 10000
BOUND_TOUCH = 1e-9          # REPORTED only: a column counts as "on a bound" within this of it


def rule(residual_median_mm: dict[int, float], threshold_mm: float = THRESHOLD_MM) -> bool:
    """DRAWN iff the bounded residual's median exceeds the threshold on BOTH donors (no rounding)."""
    return all(residual_median_mm[d] > threshold_mm for d in DONOR_INDICES)


def pose_bounds(pose_names: list[str], pose_values: np.ndarray, limits: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(keep, lb, ub) for the pose columns under the frozen conventions.

    keep[i] is False for a FIXED parameter (configured limit a point): lsq_linear refuses lb == ub, so its
    column is removed and its delta is zero by construction. A parameter with no configured limit is unbounded.
    """
    keep = np.ones(len(pose_names), bool)
    lb = np.full(len(pose_names), -np.inf)
    ub = np.full(len(pose_names), np.inf)
    for i, name in enumerate(pose_names):
        if name in limits:
            low, high = limits[name]
            if low == high:
                keep[i] = False
                continue
            lb[i] = low - float(pose_values[i])
            ub[i] = high - float(pose_values[i])
    return keep, lb[keep], ub[keep]


def bounded_residual(jacobian: np.ndarray, displacement: np.ndarray, keep: np.ndarray,
                     lb: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, dict]:
    """The frozen solve: min ||J delta - d|| within [lb, ub] over the kept columns. Returns (residual, info)."""
    from scipy.optimize import lsq_linear

    matrix = np.asarray(jacobian, np.float64)[:, keep]
    target = np.asarray(displacement, np.float64)
    result = lsq_linear(matrix, target, bounds=(lb, ub), method=LSQ_METHOD, tol=LSQ_TOL,
                        max_iter=LSQ_MAX_ITER)
    residual = target - matrix @ result.x
    on_bound = int(np.sum((np.abs(result.x - lb) <= BOUND_TOUCH) | (np.abs(result.x - ub) <= BOUND_TOUCH)))
    return residual, {"status": int(result.status), "success": bool(result.success),
                      "step_norm": float(np.linalg.norm(result.x)), "columns_on_a_bound": on_bound}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(lod: int = 2) -> dict:
    """Run on momenv. Returns the drawn-set record (the frozen JSON's content)."""
    import importlib.metadata as metadata

    import pymomentum.geometry as g
    import scipy

    import d4_o1_exactness as d4
    import d4b_identifiability as d4b
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
    pose_names = [names[i] for i in pose_index]
    if any(n.endswith("_flexible") or n.startswith("scale_") for n in pose_names):
        raise SystemExit("the pose set is not the non-flexible pose parameters")
    skeleton = list(simple.skeleton.joint_names)
    rows = [skeleton.index(j) for j in MAPPED_JOINTS]
    column = {c: names.index(c) for c in CHANNELS}

    def fk(params: np.ndarray) -> np.ndarray:
        state = np.asarray(g.model_parameters_to_skeleton_state(simple, np.asarray(params, np.float64)))
        return state[..., rows, :3].reshape(*params.shape[:-1], len(rows) * 3)

    def lengths_mm(params: np.ndarray) -> dict[str, float]:
        positions = fk(params).reshape(len(rows), 3)
        at = {j: positions[i] for i, j in enumerate(MAPPED_JOINTS)}
        return {s: float(np.linalg.norm(at[a] - at[b]) * 10.0) for s, (a, b) in SEGMENTS.items()}

    # ---- the rest-length table at MHR's zero pose, zero identity: moved-by, signed slopes, linearity
    zero = np.zeros(len(names), np.float64)
    base_lengths = lengths_mm(zero)
    change_abs, slope_upper, slope_lower = {}, {}, {}
    for c in CHANNELS:
        low, high = limits[c]
        change_abs[c] = {s: 0.0 for s in SEGMENTS}
        slope_upper[c], slope_lower[c] = {}, {}
        for label, end in (("low", low), ("high", high)):
            p = zero.copy()
            p[column[c]] = end
            moved = lengths_mm(p)
            for s in SEGMENTS:
                delta = moved[s] - base_lengths[s]
                change_abs[c][s] = max(change_abs[c][s], abs(delta))
                if end != 0.0:
                    (slope_upper if label == "high" else slope_lower)[c][s] = delta / end
    moves = {s: [c for c in CHANNELS if change_abs[c][s] > SEGMENT_MOVE_MM] for s in SEGMENTS}

    # ---- the rule, per donor
    per_donor, solver_failures = {}, []
    for donor in DONOR_INDICES:
        d4.DONOR = ROOT / DONOR_FILES[donor]
        zero_limits = dict(limits)
        for c in CHANNELS:
            zero_limits[c] = (0.0, 0.0)
        motion_full, drawn, repair = d4.truth_motion(full, np.random.default_rng(0), zero_limits)
        if any(v != 0.0 for v in drawn.values()):
            raise SystemExit("the evaluation identity is not zero")
        motion = np.stack([motion_full[:, full_names.index(n)] for n in names], axis=1).astype(np.float64)
        frames = motion.shape[0]
        bounded = {c: {"low": np.zeros(frames), "high": np.zeros(frames)} for c in CHANNELS}
        unbounded = {c: {"low": np.zeros(frames), "high": np.zeros(frames)} for c in CHANNELS}
        raw = {c: {"low": np.zeros(frames), "high": np.zeros(frames)} for c in CHANNELS}
        step = {c: np.zeros(frames) for c in CHANNELS}
        touching = {c: np.zeros(frames, int) for c in CHANNELS}
        kept_columns = None
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
            keep, lb, ub = pose_bounds(pose_names, p[pose_index], limits)
            kept_columns = int(keep.sum())
            for c in CHANNELS:
                low, high = limits[c]
                for label, end in (("low", low), ("high", high)):
                    q = p.copy()
                    q[column[c]] = end
                    d = fk(q[None])[0] - base
                    raw[c][label][f] = per_joint_max_mm(d)
                    residual, info = bounded_residual(jacobian, d, keep, lb, ub)
                    if not info["success"] or info["status"] <= 0:
                        solver_failures.append({"donor": donor, "frame": f, "channel": c, "end": label,
                                                "status": info["status"]})
                    bounded[c][label][f] = per_joint_max_mm(residual)
                    unbounded[c][label][f] = per_joint_max_mm(d4b.pose_orthogonal(d, jacobian))
                    step[c][f] = max(step[c][f], info["step_norm"])
                    touching[c][f] = max(touching[c][f], info["columns_on_a_bound"])
        agg = {c: np.maximum(bounded[c]["low"], bounded[c]["high"]) for c in CHANNELS}
        agg_unbounded = {c: np.maximum(unbounded[c]["low"], unbounded[c]["high"]) for c in CHANNELS}
        agg_raw = {c: np.maximum(raw[c]["low"], raw[c]["high"]) for c in CHANNELS}
        per_donor[donor] = {
            "donor_file_sha256": _sha256(ROOT / DONOR_FILES[donor]),
            "fixture_repair_clamped_parameter_frames": repair["clamped_parameter_frames"],
            "frames": frames,
            "pose_columns": int(len(pose_index)),
            "pose_columns_kept_after_removing_fixed": kept_columns,
            "bounded_residual_median_mm": {c: float(np.median(agg[c])) for c in CHANNELS},
            "bounded_residual_median_mm_by_end": {c: {e: float(np.median(bounded[c][e])) for e in ("low", "high")}
                                                  for c in CHANNELS},
            "bounded_residual_p05_p95_mm": {c: [float(np.percentile(agg[c], 5)), float(np.percentile(agg[c], 95))]
                                            for c in CHANNELS},
            "raw_median_mm": {c: float(np.median(agg_raw[c])) for c in CHANNELS},
            "REPORTED_unbounded_projection_median_mm_D4b_rank_convention": {
                c: float(np.median(agg_unbounded[c])) for c in CHANNELS},
            "REPORTED_bounded_step_norm_median_over_frames": {c: float(np.median(step[c])) for c in CHANNELS},
            "REPORTED_columns_on_a_bound_median_over_frames": {c: float(np.median(touching[c])) for c in CHANNELS},
        }

    drawn, excluded = [], {}
    stop_solver = bool(solver_failures)
    for c in CHANNELS:
        by_donor = {d: per_donor[d]["bounded_residual_median_mm"][c] for d in DONOR_INDICES}
        raw_by_donor = {d: per_donor[d]["raw_median_mm"][c] for d in DONOR_INDICES}
        low, high = limits[c]
        if rule(by_donor):
            drawn.append(c)
        elif low == high:
            excluded[c] = "UNDRAWABLE: the configured limit is the point [%g, %g]" % (low, high)
        elif any(raw_by_donor[d] <= THRESHOLD_MM for d in DONOR_INDICES):
            excluded[c] = ("UNSEEN: the raw displacement of every mapped joint is at or below the threshold "
                           "(nothing the 17 landmarks see moves with it)")
        else:
            excluded[c] = ("BELOW THE RULE: the raw displacement exceeds the threshold; the pose, within its "
                           "configured limits, reproduces it to within the threshold")

    # the start's segment map: each drawn channel -> the segments it moves (bilateral: both sides)
    start_segments = {c: [s for s in SEGMENTS if c in moves[s]] for c in drawn}
    one_kind = {c: len({s.replace("l_", "").replace("r_", "") for s in segs}) == 1 for c, segs in start_segments.items()}
    model = ROOT / ".cache/mhr/assets/compact_v6_1.model"
    spine_drawn = "scale_spine_length" in drawn
    return {
        "step": "D4c", "stage": 1,
        "frozen_before_any_fit": True,
        "rule": ("a channel is DRAWN iff the median over frames of the max-over-17-joints residual of the least-squares "
                 "pose correction WITHIN THE CONFIGURED LIMITS (the channel at either end of its configured limit on "
                 "the zero identity, the clamped donor pose, flexible channels excluded) exceeds 2 mm on BOTH donors"),
        "frozen": {
            "evaluation_identity": "zero (every scale_* channel 0) on the clamped donor pose, flexible channels 0",
            "displacement": "d_c = FK(p, channel at the limit end) - FK(p): the 17 mapped joints, cm, float64",
            "jacobian": f"central finite differences, step {FD_STEP:g} in each parameter's own units, float64, the "
                        "17 mapped joints against every non-flexible pose parameter of the 178-parameter character",
            "sign_convention": "delta reproduces d_c (J delta ~ d_c); bounds [lo - p, hi - p] keep p + delta within "
                               "the configured limits",
            "pose_column_membership": "the 110 non-flexible pose parameters (pose_parameters of the simplified "
                                      "character); every scale_* and every *_flexible parameter excluded",
            "unconfigured_bounds": "unbounded (-inf, inf): root_tx/ty/tz/rx/ry/rz carry no configured limit",
            "fixed_parameters": "a configured limit that is a point (lo == hi) has zero width; lsq_linear refuses "
                                "lb == ub, so the column is REMOVED and its delta is 0 by construction",
            "solver": f"scipy.optimize.lsq_linear(method={LSQ_METHOD!r}, tol={LSQ_TOL:g}, max_iter={LSQ_MAX_ITER})",
            "solver_failure": "any (frame, channel, end) whose solve reports status <= 0 STOPS the step; never dropped",
            "endpoint_aggregation": "max over the 17 mapped joints of the per-joint residual norm; then the max over "
                                    "the two ends of the configured limit; then the median over frames",
            "rounding": "none before comparison",
            "threshold_mm": THRESHOLD_MM,
            "segment_move_mm": SEGMENT_MOVE_MM,
        },
        "channels_considered": list(CHANNELS),
        "configured_limits": {c: list(limits[c]) for c in CHANNELS},
        "drawn_set": drawn,
        "excluded": excluded,
        "prediction_from_the_card": "eight drawn, including scale_spine_length and scale_shoulder_width",
        "precondition_0": {
            "scale_spine_length_drawn": spine_drawn,
            "solver_failures": solver_failures,
            "STOP": (not spine_drawn) or stop_solver,
            "card": "If scale_spine_length is not drawn, the step STOPS here: no fit runs and no acceptance fixture "
                    "is generated. A STOP says the trunk failed THIS screening rule, not that it is unidentifiable.",
        },
        "per_donor": {str(d): per_donor[d] for d in DONOR_INDICES},
        "REPORTED_drawn_set_under_the_unbounded_projection_never_selecting": [
            c for c in CHANNELS if rule({d: per_donor[d]["REPORTED_unbounded_projection_median_mm_D4b_rank_convention"][c]
                                         for d in DONOR_INDICES})],
        "segments": {s: list(pair) for s, pair in SEGMENTS.items()},
        "segment_rest_length_change_mm_by_channel_max_abs": change_abs,
        "segment_moved_by": moves,
        "start_segments_by_drawn_channel": start_segments,
        "start_segments_one_kind_per_channel": one_kind,
        "signed_rest_length_change_mm_per_unit_at_the_upper_limit": {
            c: {s: slope_upper[c][s] for s in start_segments[c]} for c in drawn},
        "REPORTED_linearity_slope_at_the_lower_limit_mm_per_unit": {
            c: {s: slope_lower[c][s] for s in start_segments[c]} for c in drawn},
        "REPORTED_linearity_lower_over_upper": {
            c: {s: (slope_lower[c][s] / slope_upper[c][s] if slope_upper[c][s] else None) for s in start_segments[c]}
            for c in drawn},
        "segments_scored": [s for s in SEGMENTS if any(c in drawn for c in moves[s])],
        "segments_reported_not_scored": [s for s in SEGMENTS if not any(c in drawn for c in moves[s])],
        "zero_identity_rest_lengths_mm": base_lengths,
        "provenance": {
            "pymomentum": metadata.version("pymomentum-cpu"),
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "model_sha256": _sha256(model),
            "fbx_sha256": _sha256(ROOT / f".cache/mhr/assets/lod{lod}.fbx"),
            "fitter_sha256": _sha256(ROOT / "tools/fitter/mhr_delivery.py"),
            "d4_fixture_sha256": _sha256(ROOT / "tools/fitter/d4_o1_exactness.py"),
            "d4b_identifiability_sha256": _sha256(ROOT / "tools/fitter/d4b_identifiability.py"),
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
        arguments.frozen_copy.parent.mkdir(parents=True, exist_ok=True)
        arguments.frozen_copy.write_text(text, encoding="utf-8")
    print("drawn:", record["drawn_set"])
    print("excluded:", json.dumps(record["excluded"], indent=1))
    for d in DONOR_INDICES:
        row = record["per_donor"][str(d)]
        print(f"donor {d}: " + ", ".join(
            f"{c.replace('scale_', '')} raw {row['raw_median_mm'][c]:.2f} bounded {row['bounded_residual_median_mm'][c]:.2f} "
            f"unbounded {row['REPORTED_unbounded_projection_median_mm_D4b_rank_convention'][c]:.2f}" for c in CHANNELS))
    print("precondition 0:", json.dumps(record["precondition_0"]))
    print("start segments:", json.dumps(record["start_segments_by_drawn_channel"]))
    print("k_c:", json.dumps(record["signed_rest_length_change_mm_per_unit_at_the_upper_limit"]))
    print("linearity:", json.dumps(record["REPORTED_linearity_lower_over_upper"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
