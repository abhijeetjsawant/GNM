# The substitution harness

The lasting asset of 2026-08-29. Everything here answers one question — *what does
one component contribute* — by holding the rest fixed and changing exactly one
thing. That method arrived from the user, and it separated in single measurements
what four purpose-built instruments could not.

Findings and full context: `docs/BATTLE1_COMPONENT_SWAP.md`.

| script | what it isolates |
|---|---|
| `swap_true.py` | MAMMA's 2D into our triangulation, against its own 512 landmarks via the exact `verts_512` regressor |
| `sam3d_ladder.py` | any detector's cross-view self-agreement — needs no reference and no joint-convention mapping |
| `mamma_residuals.py` | MAMMA's own 2D residual distribution, the target spec a candidate detector is scored against |
| `rung1_smoother.py` | the temporal smoother, isolated on real footage at real acting speed |
| `res_ablation.py` | detector input resolution, scored on spread against a shared reference |
| `crlb.py` | the single-frame Cramér-Rao bound for this rig — how much estimator headroom exists at all |
| `armi_calibrate.py`, `armi_frames.py` | whether a per-joint offset is body-fixed, and whether its frame can be built from positions alone |
| `handframe.py` | wrist-local and wrist-relative fingertip metrics, built identically for two systems |
| `gate_vs_weight.py` | which *mechanism* a confidence channel acts through — holds the information fixed and moves only `minimum_confidence`, so gating is separated from weighting. Ships the shuffled-drop control |
| `weight_ordering_differential.py` | whether `weight_before_loss` changes anything at all, with a spy on the flag the solver actually receives. Answer on this fixture: bit-identical |
| `gate_quality_curve.py` | how good a visibility head must be before gating pays — recall and false-positive rate swept separately, because they are wildly asymmetric |
| `retarget_cost.py` | what the positions→rotations stage costs, split into converter defect and un-fitted body proportions by a canonical round-trip control. Uses no MAMMA asset |
| `common_mode.py` | whether the coherent 2D bias is per-joint convention error or ONE translation per camera — splits static from time-varying, with a shuffled-offset control |
| `camera_count.py` | what more cameras would buy — CRLB on synthetic rings fitted to the real rig, with a synthetic-4 control, plus a redundancy extrapolation that fails its own check and says so |
| `veto_ceiling.py` | what a PERFECT veto of the detector's tail would buy, in mm, with a shuffled-veto control and the fraction of flagged observations the inlier gate actually used. Answer: ~0 |
| `association_swap.py` | our cross-view association against MAMMA's truth-grade subject labels, with the contact and epipolar-margin distributions the band requires, and a ±1-frame sync arm |

**Two standing rules these encode.** Score both arms on the **same denominator** — a
composition shift masquerading as an effect has happened twice here. And **no gate a
constant can pass**: every band ships with a demonstration that a degenerate
solution fails it.
