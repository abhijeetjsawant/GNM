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

**Two standing rules these encode.** Score both arms on the **same denominator** — a
composition shift masquerading as an effect has happened twice here. And **no gate a
constant can pass**: every band ships with a demonstration that a degenerate
solution fails it.
