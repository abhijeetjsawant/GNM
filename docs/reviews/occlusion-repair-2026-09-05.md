# D8 — the captured limbs under occlusion: the yellow where two bodies overlap

**2026-09-05, branch `ladder/D8`.** Report: `artifacts/compare/d8-occlusion/gate.json`.
Instruments: `tools/compare/captured_limb_stability.py` (new, committed first),
`d8_occlusion_synthetic.py`, `d8_occlusion_delivery.py`, `d8_occlusion_silhouette.py`,
`d8_occlusion_gate.py`, and `--reference raw` added to `delivered_vs_capture.py`.
Extractor stub: `tools/compare/extractors/d8_occlusion.py`.
Tests: `tests/test_occlusion_repair.py`.

**Merge rule's mechanical outcome: see §6.**

---

## 0. The pre-registration, restated verbatim

From `docs/LADDER_EXECUTION_PLAN.md` §2, the **D8 card**, written 2026-09-05 and committed
at `7edd23f` before this branch's first instrument ran. Carried into
`artifacts/compare/d8-occlusion/gate.json` under `preregistration` and quoted here in full.

> **instrument first (the D7b way):** `tools/compare/captured_limb_stability.py` reads the
> shipped npz and the per-camera observation files and reports, per performer / landmark /
> frame, segment length vs the performer's own median, camera support, per-view confidence
> and reprojection residual of the raw point, ray-pair angles, NaN runs; it must reproduce
> the figures above (legs ±5 %, performer 1 forearm 18 frames, shoulder line 68–554, B/D
> dropout, A–C 172°) before any src change
>
> **SYNTHETIC TRUTH (selector and bands):** I7's fixture — SOMASKEL77 posed clips projected
> into THIS rig with the REAL per-camera seen pattern replayed (`replayed_keep`, so the
> A–C-only window occurs by construction) and the detector's own heavy-tail noise;
> selector: 3D error of the landmarks in the two-view window vs truth, for the conditioning
> gate, the reachability reject, both, and today's code; oracle: clean fully-seen input →
> ZERO demotions, ZERO rejections, output bit-identical; must-fail: the whole-take hold
> (frozen arm) and a step test in place of reachability (the two-big-steps plateau)
>
> **REAL TAKE, bands the candidate cannot optimise:** (B1) part-wise silhouette, arms, on
> the window frames vs D7b, not worse on either performer with the CI clear and predicted
> to rise on performer 1; (B2) held-out camera reprojection (the I5 pattern) on the frames
> that HAVE a third view (9–22 of 41 in the window; the two-view frames have no held-out
> arm and are reported as such); (B3) raw array bit-identical between builds; legs and
> root: number of leg demotions/rejections REPORTED (predicted 0; the ankle has real NaN
> frames so a correct fire there is not a fail); (B4) delivered hand/elbow from the file vs
> the RAW finite points (`delivered_vs_capture --reference raw`), never vs the repaired
> points; (B5) MAMMA's joints through `subject_map` on the window: oracle arm, report only;
> head gate and D7b neck figure rerun and reported
>
> **merge rule, fixed before numbers:** synthetic selector holds for the shipped
> combination AND the oracle passes AND B1 on both performers AND B3; everything else
> reports

---

## 1. The defect, and why no existing gate could see it

In the push-and-fall window — frame ids 85–125, which is array indices 25–65, which is the
card's 41 frames — cameras **B001 and D001 lose one performer entirely**: B001 on 20 of the
41 frames and D001 on 32, with A001 and C001 losing neither performer on any window frame.
19 window frames have A001 and C001 alone. At the falling performer those two cameras sit
**171–172° apart**.

Two rays meeting at 172° determine a point *across* their common axis and barely *along*
it. The along-axis error is amplified by `1/|sin θ|` — 7.2× at 172° against 1.0× at 90° —
so the point slides freely down the A–C line while **both reprojections stay perfect**.
`inlier_threshold_px` and `maximum_epipolar_px` measure agreement between views, and the
two views *agree*, to 3–5 px, on the wrong point. This is a **conditioning** failure, not a
noise one, and no residual threshold can see one. That is the whole of the diagnosis.

What it does to the capture, measured from the shipped arrays before any change:
the shoulder line reads 68 mm at frame 108 and 554 at frame 100 against a 363 mm median;
performer 1's forearm is off its own take median by more than 15 % on 18 frames and her
upper arm on 12; performer 0's two forearms on 7 and 8. **The legs do not move**: 0 frames
off by more than 15 % on all eight leg segments, both performers, through the same window.
That contrast is what makes this an occlusion finding rather than a motion one.

---

## 2. The instrument that sees it, and the two things the card did not say

`tools/compare/captured_limb_stability.py` was written and committed **before any change to
`src/`** and reproduced **10 of 10** of the card's clauses on its first run. It reads the
shipped npz, the delivered build's own cached observations and the rig; it never
re-detects and never re-triangulates. Which detection belongs to which performer comes
from the pipeline's own association, obtained by wrapping the real `reconstruct_multiview`
with a recording associator (CLAUDE.md: a hand replication of that loop drifted 9–19 mm),
and its boolean reduction is asserted equal to `temporal.real_run_seen_mask`. The
coordinator's hip-midpoint proxy match is computed beside it and the two disagree on 43
cells.

Two things had to be established before the figures could be reproduced at all, and
neither is in the card.

**(a) The card quotes TWO arrays.** The ranges and the frame counts are on the SMOOTHED
array (`triangulated_world_positions_z_up_m`); the two point values — 68 mm at frame 108
and 554 at frame 100 — are on the RAW array. Those same two frames read 101.7 and 542.9
smoothed. Reading one array for all of the figures reproduces none of them. Every figure is
now emitted on both arrays with the array named in the key.

**(b) The card's "20–32 of 41 frames" is a PERSON-level count**, not a landmark-level one,
and it reproduces exactly on that reading: B001 loses performer 0 entirely on 20 window
frames and D001 on 32. The companion "confidence 0.05–0.13" is D001's own sub-floor
confidence band (p10–p90) on that performer's arm landmarks. Both readings are in the
report as a full matrix so neither has to be guessed again.

**What this instrument is blind to.** *Truth* — there is none on this take, and a limb
welded to its own median scores perfectly here, which is why every D8 band lives on
synthetic truth or on the photographs and never on this file. *Common-mode error* — if
every camera is wrong the same way the residuals stay small and the angles stay wide.
*Direction* — a length invariant cannot score direction (CLAUDE.md); a same-length rotation
of the shoulder line about the A–C axis is invisible here. *The delivered file* — this
scores the capture, not the rig or the mesh.

---

## 3. What ships, and where each rule sits

All three rules sit **after `world` is captured and before the sequence solve and the
fill**, so `raw_triangulated_world_positions_z_up_m` is bit-identical between the D7b and
D8 builds and is the one reference this step did not move. `world` is never written to: the
repair takes a copy, and `world.copy()` is still what `reconstruct_multiview` returns.

1. **The conditioning gate** (`_repair_occluded_slots`). A slot supported by *exactly two*
   views whose rays meet beyond `RAY_PAIR_CONDITIONING_CEILING_DEG` — or inside its
   complement — is **demoted**: the point is withheld and the rays are not, so the existing
   `solve_sequence_positions` recovers it the way it already recovers single rays, from ray
   + the performer's own parent distance + temporal continuity. Evidence, not a new prior,
   and no new capacity. Three or more supporting views are never demoted: the third view
   fixes the depth the pair cannot.
2. **The reachability reject** (`_reachable_landmark_sequence`), per landmark, measured
   from the last **accepted** point with the envelope widened by the frames elapsed —
   `_reachable_clavicle_sequence`'s idiom one dimension down. `rule="step"` is the
   must-fail control at the identical call site.
3. **The gap clause** (`_hold_long_gaps_on_parent`). A run longer than
   `MAXIMUM_INTERPOLATED_GAP_FRAMES` is not left for the fill to draw a straight line
   through; the landmark is carried on its parent and the share is reported as
   `held_joint_fraction` beside `interpolated_joint_fraction`. It sits **between** the solve
   and the fill and never inside `_fill_and_smooth_positions`, because the solve calls that
   function itself to build its starting point and changing it would move the oracle.

Every rule is a keyword argument on `reconstruct_multiview`, so the selector runs all four
arms through the one real function and never a second association loop. With all three off
the smoothed output is bit-identical to the pre-D8 build, which is a unit test.

---

## 4. The refuted prediction that matters most

**The card said the ray-angle ceiling would be selected on synthetic truth. It cannot be,
and the reason is a property of the fixture rather than of the rule.**

The fixture's bodies have exactly rigid bones and exactly smooth motion. Those are
precisely the sequence solve's own priors — limb length and temporal continuity. A recovery
whose priors are true by construction beats a noisy two-view triangulation at **every**
angle bin, including the well-conditioned 70–120° ones the amplified mask supplies, so the
score curve has no crossover and its argmin is always the lowest candidate swept. That
argmin is "demote every two-view slot" — *never trust two views* — which is a capacity
change rather than evidence, and exactly what the card forbids. Shipping it would be this
lane's standing error: **a band the candidate can optimise proves nothing about the
candidate.**

That the preference is an artefact is not a hypothesis; it is the codebase's own measurement.
`solve_sequence_positions` deliberately refuses to overwrite already-triangulated slots,
with the reason in its comment: measured on real data the solve moved them *"by a median of
11–14 mm and up to 700 mm — the temporal and limb terms outvoting good geometry."* Those
priors are true on the fixture and false on the footage.

So the ceiling is **derived in closed form** and registered as an ENGINEERING LIMIT, not as
synthetic-truth selection. Two rays meeting at θ leave the along-axis error amplified by
`1/|sin θ|`; the ceiling is the angle at which that reaches **2×**, and 2× is the declared
choice. The complement clause is the same expression from the other side, so the rule is
exactly `|sin θ| < 0.5`.

**Order of discovery, disclosed.** 150.0 was in `src` as a first guess before the closed
form was written down. `k = 2` was *recognised* as the factor that yields it, not chosen
ahead of it. The value did not move; a worse justification was replaced by a better one.

The other two constants **are** selected on the fixture, and each on the population its own
rule acts on:

* `REACHABILITY_SLACK_M` is **measured, not scored** — the p99 of the frame-to-frame jitter
  of the triangulation's own error on well-supported cells. The score sweep is flat inside
  0.06 mm across 0.02–0.40 m, because at the anatomical speed ceilings a wrist may move
  1.13 m in one frame and the slack is a rounding error on that envelope.
* `MAXIMUM_INTERPOLATED_GAP_FRAMES` is scored on the cells inside a long no-view run, one
  fixed population shared by every candidate.

`REACHABILITY_SPEED_CEILING_M_S` is **anatomy** with its sources cited beside the table, and
is deliberately a physical *impossibility* bound rather than a plausibility one: a ceiling
tight enough to refuse what a body merely rarely does would be a smoother wearing a
reject's clothes. The fixture's job on it is the pair the standing rules ask for — the
oracle fires zero rejections on clean input, and a frozen arm fails.

---

## 5. Every band, with its number and its verdict

*(filled from `artifacts/compare/d8-occlusion/gate.json`)*

---

## 6. The merge rule, applied mechanically

*(filled from the gate)*

---

## 7. What the coordinator should know

*(filled below)*
