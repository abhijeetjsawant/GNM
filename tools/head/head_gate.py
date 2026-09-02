#!/usr/bin/env python3
"""The head gate of `HEAD_ORIENTATION_MEASURED.md` §6, run against a candidate and
against **both** controls it must reject.

`BODY_LANE_PLAN.md` §1: no gate a constant can pass, and every band ships with a
demonstration that a degenerate solution fails it. §0 of the head document showed one
control is not enough here -- a locked head fails a spread test while a *noisy* head
sails through it, and the shipped constant scores at parity with MAMMA on frame-to-frame
jitter. So two controls run beside every candidate, and both are real artefacts:

  C1  the LOCKED head -- the delivered `subject-*.body-track.npz`, whose Head/Neck/eye
      local rotations are the identity quaternion on every frame.
  C2  the NOISY head -- the same landmarks and the same rigid template as the candidate,
      but fitted per frame to independently triangulated 3D. That is the candidate minus
      the two things being tested: the multi-view 2D objective and the temporal prior.

**Everything is mean-removed.** A rigid head fit determines orientation only up to a
constant rotation (`solve_head.py`), and our thorax frame is landmark-derived while
MAMMA's comes from its own kinematic chain, so the two differ by a constant by
construction. Removing each take's own mean relative pose makes the comparison a
**tracking** comparison, which is what the goal asks for.

Blind to:
  * a constant heading error, removed by construction above. Nothing here says the head
    points the right way -- only that it turns when MAMMA's turns.
  * accuracy. MAMMA is an instrument, not truth, so this is a **parity** gate. If MAMMA's
    head is smoothly wrong, a candidate matching it scores well and is equally wrong.
    `BODY_LANE_PLAN.md` §1 forbids reading any of this as an accuracy claim.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoanim_gnm.commercial_multiview import JOINT_INDEX, _thorax_frames  # noqa: E402
from mamma_head_bar import chain_world, geodesic_deg, rodrigues  # noqa: E402
from solve_head import HEAD, NAMES, gather, initialise, log_so3  # noqa: E402
from subject_map import mamma_index_for  # noqa: E402
from triangulate_soma import triangulate  # noqa: E402

MA3D = Path("artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/pushing_and_lifting_from_ground")
OUT = Path("artifacts/head-lane")
ROOT_DIR = Path(__file__).resolve().parents[2]
# The delivered tracks the gate reads its torso frame from. A module constant rather
# than a literal so an alternative build (e.g. a different detector width) can be
# scored through THIS gate instead of a copy of it.
TRACKS = Path("artifacts/commercial-multiview-soma77")
SOMA_FRAME = {"Hips": 0, "Neck2": 5, "LeftArm": 12, "RightArm": 40}
M_PELVIS, M_NECK, M_L_SH, M_R_SH, M_HEAD = 0, 12, 16, 17, 15
PELVIS, SPINE1, SPINE2, SPINE3, NECK, HEAD_J = 0, 3, 6, 9, 12, 15

BANDS = {"P1_median_deg": 8.0, "P1_p95_deg": 20.0, "P2_spread_min_deg": 8.0,
         "P3_opposing_frames": 0, "P4_travel_multiple": 3.0}
# Pre-registered in HEAD_ORIENTATION_MEASURED.md §6b, before measuring: a frame is
# review_required when fewer than this many cameras see at least MINIMUM_HEAD_LANDMARKS
# of the five head landmarks. Chosen from geometry -- two rays fix a 3D point but leave
# the rotation of a ~120 mm object at 5 m badly conditioned.
MINIMUM_HEAD_CAMERAS = 3
MINIMUM_HEAD_LANDMARKS = 3


def observed_frames(subject: int) -> np.ndarray:
    """Frames whose head is observed well enough to be reported without a flag."""
    observations, _ = gather(subject)
    seen = (np.isfinite(observations[..., :2]).all(axis=3)
            & (observations[..., 2] >= 0.25))
    return (seen.sum(axis=2) >= MINIMUM_HEAD_LANDMARKS).sum(axis=1) >= MINIMUM_HEAD_CAMERAS


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def frame_from(up: np.ndarray, across: np.ndarray) -> np.ndarray:
    """Orthonormal frame with column 2 along `up` and column 0 along `across`."""
    z = unit(up)
    x = across - z * np.einsum("ni,ni->n", across, z)[:, None]
    x = unit(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=2)


def mean_removed(relative: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Remove the take's mean relative pose -- the gauge fix P1 rests on.

    `mask` selects the population the mean is estimated FROM, and it must be the same
    population the result is scored on. Estimating the mean over frames that are not
    scored biases it; the original version of this function had no mask, and the caller
    injected the identity on unscored frames before calling it, which dragged the mean
    toward identity and **inflated every gated arm by roughly 2x**. That defect produced
    §6c's original conclusion. Corrected, the flag helps subject 0 rather than hurting
    every arm. The ungated arms were unaffected -- their masks are all-true -- so the
    headline verdict never depended on it.
    """
    population = relative if mask is None else relative[mask]
    u, _, vt = np.linalg.svd(population.mean(axis=0))
    mean = u @ vt
    if np.linalg.det(mean) < 0:
        u[:, -1] *= -1
        mean = u @ vt
    return np.einsum("nij,kj->nik", relative, mean)


def travel(rotations: np.ndarray, valid: np.ndarray) -> np.ndarray:
    index = np.flatnonzero(valid)
    pairs = index[1:][np.diff(index) == 1]
    if pairs.size == 0:
        return np.array([np.nan])
    return geodesic_deg(rotations[pairs], rotations[pairs - 1])


def score(deviation: np.ndarray, reference: np.ndarray, valid: np.ndarray,
          mamma_travel_p95: float) -> dict:
    agreement = geodesic_deg(deviation[valid], reference[valid])
    spread = geodesic_deg(deviation[valid], np.broadcast_to(np.eye(3), deviation[valid].shape))
    own_travel = travel(deviation, valid)
    p1_median, p1_p95 = float(np.median(agreement)), float(np.percentile(agreement, 95))
    p2 = float(np.median(spread))
    p4 = float(np.nanpercentile(own_travel, 95))
    passes = {
        "P1": p1_median <= BANDS["P1_median_deg"] and p1_p95 <= BANDS["P1_p95_deg"],
        "P2": p2 >= BANDS["P2_spread_min_deg"],
        "P4": p4 <= BANDS["P4_travel_multiple"] * mamma_travel_p95,
    }
    return {
        "frames": int(valid.sum()),
        "P1_agreement_with_mamma_deg": {"median": p1_median, "p95": p1_p95},
        "P2_spread_about_take_mean_deg_median": p2,
        "P4_travel_p95_deg": p4,
        "P4_ceiling_deg": BANDS["P4_travel_multiple"] * mamma_travel_p95,
        "passes": passes,
        "verdict": "PASS" if all(passes.values()) else "FAIL",
        "failed": [k for k, v in passes.items() if not v],
    }


def main() -> None:
    # Which solve to score. Default is the prototype's; pass a path to score the one the
    # PIPELINE ships. They are different estimators with different selection paths, and on
    # this fixture they chose different temporal weights -- so "passes the gate" and "is
    # delivered" must be shown of the SAME rotations or they are two claims about two
    # heads. See tools/head/gate_the_shipped_head.py.
    # Default to the SHIPPED solve, not the prototype's. The default used to be
    # `head-solve.npz`, the prototype's file, and scoring it by accident reports a
    # verdict for a head nobody receives -- on this fixture the two disagree enough to
    # swap which performer fails. That is the "two claims about two heads" defect this
    # tool's companion was written to prevent, so the default now points at delivery
    # and the banner says which estimator was scored.
    shipped = OUT / "head-solve-shipped.npz"
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
    else:
        source = shipped if shipped.is_file() else OUT / "head-solve.npz"
    print(f"scoring {source}"
          + ("  [the head the PIPELINE delivers]" if source == shipped
             else "  [WARNING: NOT the delivered head -- prototype estimator]"))
    solved = np.load(source)
    # The pipeline applies an anatomical gauge, so its head's local frame is canonical
    # rather than the raw template frame. P3 must ask for the skull's up axis accordingly.
    gauge_applied = "gauge_applied" in solved.files
    soma = triangulate([SOMA_FRAME[n] for n in SOMA_FRAME] + [HEAD[n] for n in NAMES])[0]
    fslot = {n: i for i, n in enumerate(SOMA_FRAME)}
    hslot = {n: len(SOMA_FRAME) + i for i, n in enumerate(NAMES)}
    mapping = mamma_index_for(np.load(OUT / "three-detector-cache.npz")["apple_vision"])
    report: dict = {"bands": BANDS, "subject_correspondence":
                    {f"our_{k}": f"mamma_body_id-{v:02d}" for k, v in mapping.items()}}

    for subject in range(2):
        # --- MAMMA's reference, both frames from its own output -------------------
        params = np.load(MA3D / f"smplx_params_body_id-{mapping[subject]:02d}.npz", allow_pickle=True)
        joints = np.load(MA3D / f"verts_joints_body_id-{mapping[subject]:02d}.npz",
                         allow_pickle=True)["pred_joints"].astype(np.float64)
        pose = params["smplx_pose"].astype(np.float64)
        m_head = chain_world(pose, (PELVIS, SPINE1, SPINE2, SPINE3, NECK, HEAD_J))
        m_thorax = frame_from(joints[:, M_NECK] - joints[:, M_PELVIS],
                              joints[:, M_L_SH] - joints[:, M_R_SH])
        m_relative = np.einsum("nji,njk->nik", m_thorax, m_head)
        m_dev = mean_removed(m_relative)
        m_travel_p95 = float(np.nanpercentile(travel(m_dev, np.ones(len(m_dev), bool)), 95))

        # --- our thorax frame ----------------------------------------------------
        # From the pipeline's own SMOOTHED torso positions, not raw triangulation.
        # The first version of this gate built it from raw triangulated Neck2/Hips/
        # shoulders and scored the result against MAMMA's *fitted* thorax. That is a
        # same-denominator violation and it dominated the verdict: the candidate head's
        # own world travel is p95 1.2 / 4.7 deg, while the relative-pose travel the gate
        # reported was 13 / 37 -- almost all of it the reference frame's noise, not the
        # head's. Scored against a comparably-conditioned thorax, P4 measures the head.
        pos = soma[subject]
        smoothed = np.load(
            TRACKS / f"subject-{subject:02d}.body-track.npz"
        )["triangulated_world_positions_z_up_m"]
        up = smoothed[:, JOINT_INDEX["neck"]] - smoothed[:, JOINT_INDEX["root"]]
        # ONE definition of the torso frame, imported from the pipeline rather than
        # rebuilt here. The gate and the solver having their own copies is how they drift,
        # and this lane has already been caught scoring one head while shipping another.
        torso_ok = np.isfinite(smoothed).all(axis=(1, 2))
        thorax = np.full((len(smoothed), 3, 3), np.nan)
        thorax[torso_ok] = _thorax_frames(smoothed)[torso_ok]

        # --- candidate: the multi-view, multi-frame rigid head fit ---------------
        candidate_world = solved[f"subject_{subject:02d}_head_world"]
        template = solved[f"subject_{subject:02d}_template_m"]

        # --- C2: same landmarks, same template, per-frame from triangulated 3D ---
        _, noisy_rot, _ = initialise(subject)
        noisy_world = rodrigues(noisy_rot)
        noisy_ok = np.isfinite(pos[:, [hslot[n] for n in NAMES]]).all(axis=(1, 2))

        arms = {}
        for label, head_world, ok in (
            # ORACLE: MAMMA's OWN head rotations, scored through our thorax frame and
            # the identical path. A perfect head cannot do better than this, so its P1
            # is the floor the two thorax DEFINITIONS impose -- ours landmark-derived,
            # MAMMA's from its fitted chain. Mean-removal kills the constant offset
            # between them; the pose-dependent part survives and lands in P1. Without
            # this arm the gate demonstrates that degenerate solutions fail and never
            # that a passing candidate exists. A gate no oracle can pass is
            # miscalibrated, which is the dual of "no gate a constant can pass".
            # NOTE: an oracle arm passing is not "MAMMA passes, ship MAMMA" -- MAMMA is
            # an instrument and never ships (CLAUDE.md).
            ("ORACLE_mamma_head_our_thorax", m_head, torso_ok),
            ("candidate_multiview_fit", candidate_world, torso_ok),
            ("C2_noisy_per_frame_triangulated", noisy_world, torso_ok & noisy_ok),
        ):
            relative = np.einsum("nji,njk->nik", np.nan_to_num(thorax, nan=0.0), head_world)
            # Both sides' gauges are estimated on the SCORED population, never on frames
            # that are excluded -- see mean_removed's docstring for the defect this fixes.
            arms[label] = score(
                mean_removed(relative, ok), mean_removed(m_relative, ok), ok, m_travel_p95)

        # --- C1: the locked head -- the constant this lane used to deliver -------
        # A head welded to the torso: head world == thorax world, so the relative pose
        # is exactly the identity on every frame. This is the "no gate a constant can
        # pass" arm, and it must keep failing.
        #
        # It used to be read off the delivered artifact and called "delivered". That
        # was true only while the pipeline shipped the constant; now that the head
        # solve is on the delivery path, reading the artifact would silently turn this
        # control into a second copy of the candidate and the gate would lose its
        # demonstration. The control is therefore CONSTRUCTED, and its name says what
        # it is rather than where it came from.
        locked = np.broadcast_to(np.eye(3), (len(m_relative), 3, 3))
        arms["C1_locked_head_constant"] = score(
            locked, mean_removed(m_relative, torso_ok), torso_ok, m_travel_p95)
        arms["C1_locked_head_constant"]["note"] = (
            "head welded to the torso: head-relative-to-thorax is the identity on "
            "every frame. Constructed, not read from an artifact -- this is the "
            "degenerate solution the gate exists to reject."
        )

        # --- P3, on the candidate: the skull long axis against neck-up -----------
        if gauge_applied:
            from autoanim_gnm.head_orientation import CANONICAL_HEAD_AXES

            head_up_local = CANONICAL_HEAD_AXES[:, 1]
        else:
            head_up_local = unit(
                (template[NAMES.index("HeadEnd")] - template[NAMES.index("Head")])[None]
            )[0]
        skull = np.einsum("nij,j->ni", candidate_world, head_up_local)
        neck_up = up
        p3_ok = torso_ok & np.isfinite(skull).all(axis=1)
        dots = np.einsum("ni,ni->n", unit(skull[p3_ok]), unit(neck_up[p3_ok]))
        opposing = int((dots < 0).sum())
        arms["candidate_multiview_fit"]["P3_opposing_neck_up"] = {
            "frames": int(p3_ok.sum()), "opposing": opposing}
        arms["candidate_multiview_fit"]["passes"]["P3"] = opposing == BANDS["P3_opposing_frames"]
        arms["candidate_multiview_fit"]["failed"] = [
            k for k, v in arms["candidate_multiview_fit"]["passes"].items() if not v]
        arms["candidate_multiview_fit"]["verdict"] = (
            "PASS" if all(arms["candidate_multiview_fit"]["passes"].values()) else "FAIL")

        # --- §6b: the same four arms, re-scored on the un-flagged population -------
        # The verdict above does NOT move; this reports what a flagged output delivers.
        reported = observed_frames(subject)
        gated: dict[str, dict] = {}
        for label, head_world, ok in (
            ("ORACLE_mamma_head_our_thorax", m_head, torso_ok & reported),
            ("candidate_multiview_fit", candidate_world, torso_ok & reported),
            ("C2_noisy_per_frame_triangulated", noisy_world, torso_ok & noisy_ok & reported),
        ):
            relative = np.einsum("nji,njk->nik", np.nan_to_num(thorax, nan=0.0), head_world)
            gated[label] = score(
                mean_removed(relative, ok), mean_removed(m_relative, ok), ok, m_travel_p95)
        locked_ok = torso_ok & reported
        gated["C1_locked_head_constant"] = score(
            np.broadcast_to(np.eye(3), (len(m_dev), 3, 3)),
            mean_removed(m_relative, locked_ok), locked_ok, m_travel_p95)

        report[f"subject_{subject:02d}"] = {
            "review_flag": {
                "rule": f"fewer than {MINIMUM_HEAD_CAMERAS} cameras seeing >= "
                        f"{MINIMUM_HEAD_LANDMARKS} of 5 head landmarks",
                "frames_reported": int(reported.sum()),
                "frames_flagged": int((~reported).sum()),
                "flagged_fraction": float((~reported).mean()),
                "over_25pct_ceiling": bool((~reported).mean() > 0.25),
            },
            "gated_arms_verdict_not_binding": gated,
            "mamma_reference": {
                "spread_about_take_mean_deg_median": float(np.median(geodesic_deg(
                    m_dev, np.broadcast_to(np.eye(3), m_dev.shape)))),
                "travel_p95_deg": m_travel_p95,
            },
            "arms": arms,
        }

    # ABSOLUTE FACING, carried here as a standing figure so the head's verdict can never
    # again be read without it. Every band above MEAN-REMOVES each take, so this gate
    # scores TRACKING and is structurally blind to a constant offset -- a head pointing
    # backwards, smoothly, on every frame passes it (CLAUDE.md; and D1 measured exactly
    # that shipping for a month). The forward-dot from `tools/compare/facing_location.py`
    # is the reading that is not blind to it, so it is quoted beside the verdict rather
    # than living in another report. It is not a band here and nothing above depends on
    # it; it is the caveat, printed.
    facing_paths = (
        ROOT_DIR / "artifacts/compare/d1-fix/facing-after.json",
        ROOT_DIR / "artifacts/compare/facing-location.json",
    )
    facing = None
    for candidate in facing_paths:
        if candidate.exists():
            try:
                facing = (str(candidate), json.loads(candidate.read_text()))
                break
            except (OSError, ValueError):
                continue
    if facing is not None:
        source, values = facing
        report["absolute_facing_not_a_band"] = {
            "source": source,
            "what_it_is": ("the delivered Head's own +Z against the direction the footage "
                           "says the performer faces. +1 is with the performer, -1 exactly "
                           "opposed. Every band in this file is blind to it."),
            "per_subject": {
                subject: {
                    "forward_dot_vs_our_capture": values["forward_dot"][subject]
                    .get("delivered_Head", {}).get("vs_our_capture_forward", {}).get("median"),
                    "forward_dot_vs_mamma": values["forward_dot"][subject]
                    .get("delivered_Head", {}).get("vs_mamma_forward", {}).get("median"),
                    "ORACLE_mamma_own_head_vs_our_capture": values["forward_dot"][subject]
                    .get("ORACLE_mamma_head_forward_vs_our_capture_forward", {})
                    .get("vs_our_capture_forward", {}).get("median"),
                }
                for subject in ("subject_00", "subject_01")
                if subject in values.get("forward_dot", {})
            },
            "reading": ("the reference is the BODY's forward, from the pelvis and the neck, "
                        "so a head is not expected to reach +1 -- the ORACLE row is what "
                        "this figure can reach on this take. What it rules out is a "
                        "REVERSAL, which is what the mean-removed bands cannot see."),
        }
    else:
        report["absolute_facing_not_a_band"] = {
            "unavailable": ("run tools/compare/facing_location.py -- without it this "
                            "gate's verdict carries no statement about absolute facing"),
        }

    destination = OUT / ("head-gate-shipped.json" if gauge_applied else "head-gate.json")
    destination.write_text(json.dumps(report, indent=2))
    for subject in ("subject_00", "subject_01"):
        block = report[subject]
        print(f"\n=== {subject}  (MAMMA spread {block['mamma_reference']['spread_about_take_mean_deg_median']:.2f}deg, "
              f"travel p95 {block['mamma_reference']['travel_p95_deg']:.2f}deg) ===")
        print(f"{'arm':34s} {'P1 med':>7s} {'P1 p95':>7s} {'P2 spread':>10s} "
              f"{'P4 travel':>10s}  verdict")
        for label, row in block["arms"].items():
            print(f"{label:34s} {row['P1_agreement_with_mamma_deg']['median']:7.2f} "
                  f"{row['P1_agreement_with_mamma_deg']['p95']:7.2f} "
                  f"{row['P2_spread_about_take_mean_deg_median']:10.2f} "
                  f"{row['P4_travel_p95_deg']:10.2f}  {row['verdict']}"
                  + (f"  (failed {','.join(row['failed'])})" if row["failed"] else ""))
        flag = block["review_flag"]
        total = flag["frames_flagged"] + flag["frames_reported"]
        print(f"  §6b review flag: {flag['frames_flagged']}/{total} frames "
              f"({100 * flag['flagged_fraction']:.1f}%)"
              + ("  ** OVER THE 25% CEILING **" if flag["over_25pct_ceiling"]
                 else "  (under the 25% ceiling)")
              + "  -- verdict below is NOT binding, see §6c")
        for label, row in block["gated_arms_verdict_not_binding"].items():
            print(f"  gated {label:28s} {row['P1_agreement_with_mamma_deg']['median']:7.2f} "
                  f"{row['P1_agreement_with_mamma_deg']['p95']:7.2f} "
                  f"{row['P2_spread_about_take_mean_deg_median']:10.2f} "
                  f"{row['P4_travel_p95_deg']:10.2f}  {row['verdict']}")


if __name__ == "__main__":
    main()
