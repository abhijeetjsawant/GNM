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

What the fixture *can* do, and does, is show the mechanism is real. Binned by the angle its
two surviving views make at the true point, the **two-view triangulation's own error** rises
with the conditioning: 26.9 mm at 60–90°, 21.5 at 90–120°, then 32.4 at 140–155°, **59.6 at
155–165°** and 55.7 at 165–180°. The correlation with the closed form's `1/|sin θ|` across
the populated bins is **+0.74** — consistent with it, and quoted as that rather than as a
fit. What the fixture cannot do is find the crossover, because the *recovery* also beats
triangulation at 60–120° (19.7 against 26.9), where the pair is well conditioned and there
is nothing wrong to fix.

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
* `MAXIMUM_INTERPOLATED_GAP_FRAMES` **hits the same artefact and is not selected either.**
  Scored on the cells inside a long no-view run — one fixed population shared by every
  candidate, on the amplified mask, because the replayed mask has *no* such run — the
  fixture prefers interpolating at every candidate length (57.0 mm at N = 2, 56.9 at
  N = 3 through 18, 54.6 at N = 24): another curve whose argmin is "never hold". The fixture's
  motion is smooth by construction, so a straight line through a gap is nearly exact there.
  The value is a closed-form bound instead: a straight chord across a gap of duration `T`
  departs from a constant-acceleration trajectory by at most `a·T²/8`, and setting that
  equal to the measured slack gives `T = sqrt(8·slack/a)` — **5.7 frames at a limb
  acceleration of 20 m/s²**, 3.6 at 50. The shipped 6 is the 20 m/s² end and **that choice
  is declared, not derived**. The clause's value is the diagnostic it emits, not an error
  reduction; on a smooth fixture it is a cost, because there it is one.

**Two of the three rules are nearly inert on the fixture, and that is stated rather than
buried.** The conditioning gate carries the whole gain: today 47.5 mm → conditioning 22.1 →
both 22.1, on the two-view window cells against exact truth. A moving-block bootstrap on
identical draws puts `both` against `today` at [20.9, 22.6] vs [43.6, 49.2] mm with
P(candidate beats control) = 1.000, and `both` against `conditioning` alone at **P = 0.555**
— indistinguishable. The reachability reject fires on 0–1 slots there and the gap clause
costs nothing measurable. Whether either matters on the footage is B3's reported figure,
not the fixture's.

`REACHABILITY_SPEED_CEILING_M_S` is **anatomy** with its sources cited beside the table, and
is deliberately a physical *impossibility* bound rather than a plausibility one: a ceiling
tight enough to refuse what a body merely rarely does would be a smoother wearing a
reject's clothes. The fixture's job on it is the pair the standing rules ask for — the
oracle fires zero rejections on clean input, and a frozen arm fails.

---

## 5. Every band, with its number and its verdict

| band | what it asked | measured | verdict |
|---|---|---|---|
| **instrument first** | reproduce the card's figures before any src change | **10 of 10** clauses, first run | **PASS** |
| **synthetic selector** | the shipped combination beats today's code on the two-view window cells | **47.5 → 22.1 mm**; block bootstrap on identical draws, [20.9, 22.6] against [43.6, 49.2], P = 1.000 | **PASS** |
| **oracle** | clean fully-seen input → zero demotions, zero rejections, bit-identical | 0, 0, `held_joint_fraction` 0.0, smoothed AND raw bit-identical | **PASS** |
| **must-fail: frozen arm** | a whole-take hold must score worse | **29.5 mm** against the shipped 22.1 | **FAILS as required** |
| **must-fail: step test** | a step test in place of reachability | 22.074 mm — **ties**, see §7.5 | **inconclusive at the pipeline level; discriminated in the unit tests** |
| **B1 photographs, arms, window** | not worse than D7b on either performer "with the CI clear" | p0 **+0.0025 [0.00003, 0.0294]**; p1 **−0.0065 [−0.0260, 0.0146]** | **PASS on both** on the reading D7b used; see §6 |
| **B1 prediction** | a rise predicted on performer 1 | the rise landed on **performer 0** with the interval clear of zero; performer 1's point estimate **falls** | **REFUTED**, §7.7 |
| **B2 held-out camera** | reprojection into a camera the reconstruction never saw, on frames with a third view | pooled window median of the four fold medians **8.94 → 8.43 px**; better on 3 folds of 4 (A 8.39→7.82, B 8.99→8.28, C 8.89→8.58) and **worse on D001, 10.15 → 14.26** | **REPORTED** |
| **B3 raw array** | bit-identical between the D7b and D8 builds | **identical on both performers**; observations identical; smoothed differs, as it must | **PASS** |
| **B3 legs (reported)** | leg demotions and rejections, predicted 0 | **NOT 0** — p0 `left_hip` 35, `right_hip` 35, `right_knee` 26; p1 `left_knee` 35, `right_knee` 26, `left_hip` 25, `right_hip` 24, `right_ankle` 5 | **prediction REFUTED**, §7.7 |
| **B4 placement vs RAW** | delivered hand/elbow against the raw finite points | `LeftHand` p95 **159 → 120** (p0) and **158 → 102** (p1); `LeftLowerArm` p95 261 → 188 (p1); `Neck` p95 **105 → 367** (p1) | **REPORTED**, and see §7.8 |
| **B5 D3 closure** | the delivered GLB agrees with FK of the track it carries, ≤ 1e-6 m | D8 **4.58e-7 / 5.27e-7 m**; D7b 5.13e-7 / 4.71e-7 | **PASS** |
| **B5 same denominator** | expected to report CHANGED | smoothed landmarks changed on both performers | **CHANGED as expected** |
| **hygiene** | today's code on the same inputs byte-identical to the shipped delivery | **all 8 delivered files identical**, run BEFORE any src change | **PASS** |

### The capture itself, before and after

The measure the defect was stated in, rerun on the repaired build through the same
instrument. Lower is better; the count is frames off that performer's own take median by
more than 15 %.

| | performer 0 | performer 1 |
|---|---|---|
| shoulder line | **22 → 4** | **27 → 18** |
| shoulder line, min/max mm | 194/380 → **267/378** | 102/543 → **155/395** |
| forearm L | **7 → 0** | **18 → 4** |
| upper arm L | 0 → 0 | **12 → 7** |
| every leg segment | 0 → 0 | 0 → 0 |
| thigh spread p5/p95 | −4.4 %/+3.5 % → −4.8 %/+3.1 % | −4.2 %/+3.4 % → −3.9 %/+3.7 % |

The 554 mm shoulder line is gone and the 68 mm collapse with it. **The legs are unmoved on
every segment**, which matters because the legs *were* demoted (§7.7): a demotion that
changed nothing measurable is what a correctly-scoped rule looks like.

### The hygiene arm, run first with `src` unchanged

```
subject-00.glb              40e2563445fabdd0…  identical
subject-00.body-track.json  eb5d635a4e5a776b…  identical
subject-00.body-track.npz   0b64277adc474dc1…  identical
subject-00.mapping.npz      6632f33571fae013…  identical
subject-01.glb              9f864d67cf1800c0…  identical
subject-01.body-track.json  f85dc266ccdd7bdf…  identical
subject-01.body-track.npz   7073475fb78c8473…  identical
subject-01.mapping.npz      d11c72c17b482ca2…  identical
```

Every difference in the D8 arm therefore belongs to the src change and to nothing else. The
D8 build's own **raw** landmarks are byte-identical to the shipped ones on both performers,
so every band shares one denominator. **Nothing was written under
`artifacts/commercial-multiview-soma77/`** by any instrument in this step.

---

## 6. The merge rule, applied mechanically

> **merge rule, fixed before numbers:** synthetic selector holds for the shipped
> combination AND the oracle passes AND B1 on both performers AND B3; everything else
> reports

| clause | holds |
|---|---|
| synthetic selector holds for the shipped combination | **yes** — 47.5 → 22.1 mm, P = 1.000 on identical draws |
| the oracle passes | **yes** — 0 demotions, 0 rejections, bit-identical |
| B1 on both performers | **yes** — +0.0025 [0.00003, 0.0294] and −0.0065 [−0.0260, 0.0146] |
| B3 | **yes** — the raw array is byte-identical on both performers |

**Outcome: MERGE.** No clause failed and no band was moved.

### The one phrase in the rule that admits two readings, and which was applied

The card's B1 says *"not worse on either performer with the CI clear"*, and that admits two
readings:

* **"not significantly worse"** — the interval's upper bound is at or above zero. This is
  the reading **applied here**, and it is `d7b_silhouette_partwise.py`'s own `not_worse`
  predicate, written for the identical clause text in the D7b card and passed by D7b on
  arms whose intervals also spanned zero (+0.0044 and −0.0039, both spanning zero, recorded
  as PASS in `docs/reviews/trunk-resolve-2026-09-05.md` §3). D8 uses the same predicate on
  the same instrument lineage.
* **"the interval must exclude negative values"** — a strictly stronger test. Under it
  performer 1, at [−0.026, +0.015], **fails**, and the merge rule's outcome flips to **DO
  NOT MERGE**.

The band is not moved and the agent does not choose between the readings. The applied
reading is stated, the precedent that establishes it is cited, and the outcome under the
stricter reading is stated too. If the coordinator intends the stricter reading, the same
sentence in the D7b card would have to be re-read the same way, and D7b would not have
merged on its arms clause either.

It is not an unqualified pass, and the qualifications are in §7: **three predictions are
refuted** (B1's direction, the legs' zero demotions, and the card's own selection method for
two of the three constants), and **one figure is a possible real cost** — performer 1's
delivered neck moves further from the raw captured point on 19 window frames, which B4 is
structurally unable to adjudicate because its reference is the point the step discarded
(§7.8). The merge rule does not depend on any of them, which is what fixing it before the
numbers was for.

**Two of the three shipped rules are nearly inert.** On the fixture the conditioning gate
carries the whole gain (`both` vs `conditioning` alone: P = 0.555, indistinguishable). On
the real take the reachability reject fires 5 and 17 times and the gap clause holds
0.00035 and 0.0 of slots. They ship because the card ships them and because each removes a
claim the pipeline could not support; neither is doing measurable work here, and that is
stated rather than implied.

---

## 7. What the coordinator should know

### 7.1 Two of the card's three constants could not be selected the way the card said

Both are in §4 and both are the same class of failure: **the fixture's motion satisfies the
incumbent's priors exactly**, so the score curve is monotone and its argmin is the
degenerate. For the ray angle the degenerate is "never trust two views"; for the gap it is
"never hold". Neither is shipped. The ceiling is a closed-form conditioning bound at a
declared 2× amplification; the gap is a closed-form chord bound at a declared 20 m/s². Both
are registered as ENGINEERING LIMITS in `tools/compare/provenance.py`, not as
synthetic-truth selection, and the audit reports CLEAN. **`REACHABILITY_SLACK_M` is the one
D8 constant genuinely selected on synthetic truth**, and it is measured rather than scored.

The general lesson is one this lane already has in another form: *a band the candidate can
optimise proves nothing about the candidate.* Its dual is what bit here — **a fixture that
satisfies the incumbent's priors cannot adjudicate against the incumbent.** A fixture with
non-rigid bones and non-smooth motion, or marker data from lane H, would settle both.

### 7.2 What changed from the card, and why

* **The fixture's performer placement.** `temporal.fk_trajectory` places the synthetic
  performers at the whole-take root medians, which puts the A–C pair at 159–162° — not the
  window's own geometry. This step places them at the **real window's own median root**,
  which gives 164° and 171° against the footage's 171–172°. It is a placement for a
  synthetic body; no real position, length or angle enters any score.
* **The amplified arm.** The replayed mask contains **no** two-view cell below 140° and
  **no** run in which all four cameras lose a landmark (0 gap cells). It therefore cannot
  locate a conditioning threshold and cannot select a gap ceiling. I7's committed
  `amplified_keep` deals the four most-occluded **measured** (subject, camera) series to
  the four cameras — every property of the outage is the footage's own, only the assignment
  changes, and I7 declares it as such — and it supplies cells at 70–178° and 526 gap cells.
  The four-arm selector table stays on the replayed mask.
* **The gap sweep's population, corrected mid-pass.** The first version scored the gap
  ceiling on every two-view window cell, which is almost entirely cells the rule never
  touches; the axis came out flat inside 0.6 mm for that reason. Recorded rather than
  quietly fixed.
* **A tie-break added after a flat sweep.** `AXIS_UNDETERMINED_MM` was written after the
  first pass came out flat, and is disclosed as a fallback rather than a first choice. In
  the end no axis was selected through it.
* **`--reference raw` on `delivered_vs_capture.py`** and `--landmarks-from` /
  `--skip-reproduction` on the stability instrument. Both defaults are unchanged; the
  default `delivered_vs_capture` mode was regression-checked figure for figure against the
  committed D7b report before anything else ran.

### 7.3 Which downstream instruments lost their denominator, and which did not

**This is by design and the card says so.** D8's repair sits between the raw triangulation
and the smoothed landmarks, so every instrument that compared two builds' *smoothed*
landmarks byte for byte now reports CHANGED:

| instrument | what it did | what it does now |
|---|---|---|
| `delivered_vs_capture.py` default mode | asserted smoothed landmarks identical across arms, exit 1 otherwise | reports `same_denominator: false` and exits 1 — **expected**. B4 uses `--reference raw` instead |
| the D3 closure gate's same-denominator clause | compared the two builds' triangulated landmarks | reports CHANGED — **expected**. Its real clause, that the delivered GLB agrees with FK of the track it carries, is unaffected and still a check |
| `d7b_silhouette_partwise.py` | raises if the builds do not share triangulated positions | would raise on a D8 arm. `d8_occlusion_silhouette.py` asserts the **RAW** array instead, and reports the smoothed one as expected-false |
| the part-wise silhouette's tilt terciles | cut by tilt from the smoothed array | D8 cuts by the **window**, and reports tilt from the RAW array |

**What did NOT move:** the raw array itself (B3), the per-camera observations, the
association, and therefore every association-derived figure. `real_run_seen_mask` and the
whole I7 outage description are unaffected.

**One that moves further than it looks.** `_solve_head_for_subject` and `_thorax_frames`
both read the **smoothed** `positions`, so on a D8 build the head *solve* itself changes,
not merely its scoring. `tools/head/head_gate.py` does **not** see this: it scores
`artifacts/head-lane/head-solve-shipped.npz`, a head-lane artifact that this build does not
write, so rerunning it reproduces its own figures unchanged and is **blind to D8's effect on
the delivered head**. That is stated here rather than left to be discovered; the head figure
that would see it is the delivered head world orientation read from the two GLBs' own bytes,
and it is not in the card.

### 7.4 Legacy callers

`reconstruct_multiview` now applies all three rules **by default**, so every caller gets
D8's smoothed output unless it passes the new keywords. That is `scripts/build_commercial_
multiview_comparison.py` (the delivery), `scripts/build_synthetic_truth_fixture.py`,
`scripts/compare_association_strategies.py`, `measure_detector_outputs.py`,
`measure_sigma_value.py`, and every instrument under `tools/` that reconstructs. **Every
committed report under `artifacts/` that quotes a smoothed figure was produced pre-D8 and
is not comparable to a fresh run** — the I7 temporal report, `oracle_2d`, the D3 and D7
gates among them. Their raw-array and association figures are unaffected. Nothing was
re-run to match, because re-running them is the coordinator's call and not a branch's.

With the three rules off the output is bit-identical to the pre-D8 build, which is
`tests/test_occlusion_repair.py::test_the_oracle_clean_input_is_bit_identical_and_fires_nothing`
and the selector's `today` arm.

### 7.5 Small things that would otherwise be discovered the hard way

* **"The falling performer" is two different people depending on the figure.** The card's
  camera classification (B001 20 frames, D001 32) is **performer 0**; the worst limb damage
  (forearm 18 frames, shoulder line 68–554, delivered left hand 276 mm p95) is **performer
  1**. Both bodies are in the same overlap and both are degraded; the numbers were taken
  from different ones and the card reads as though they were one.
* **The legs' "±5 %" is a rounding.** Measured, six of eight leg segments sit inside ±5 %
  and two do not: performer 0's left shin reads −5.1 % at p5 and performer 1's right thigh
  +6.2 % at p95. The card's exact sub-claim — 0 frames off by more than 15 % — holds on 8
  of 8. The seed registered is the measured spread, not the rounded one.
* **B2's held-out camera is the right instrument here for a reason worth one sentence.** On
  the B001 and D001 folds the held-out camera looks roughly *perpendicular* to the A–C axis,
  so it sees exactly the depth error the A–C pair is blind to. That is why this band is not
  the I5 pattern applied by habit.
* **The step test ties the reachability rule at the pipeline level** (22.074 mm both), because
  at the selected slack the reject fires on 0–1 slots and there is nothing for the two rules
  to disagree about. The crisp discrimination is in the unit tests, which construct both of
  the step test's defects directly.
* **`artifacts/` is gitignored**, so nothing under `artifacts/compare/d8-occlusion/` is in
  the commits. Every file is regenerated by the commands in the extractor's `REGEN`.

### 7.7 The three refuted predictions

**(a) B1's predicted direction.** The card predicted the arms would rise on **performer 1**.
They rise on **performer 0** — +0.0025 IoU with the interval clear of zero, [0.00003,
0.0294] — and on performer 1 the point estimate **falls**, −0.0065 [−0.0260, 0.0146],
spanning zero. The band as written ("not worse on either performer") passes on both; the
*prediction* is refuted and the band is not moved. Performer 0's precision and recall on the
arms **both** rise (0.785 → 0.826, 0.1146 → 0.1165), so it is not the dilated-blob pattern;
on performer 1 both fall (0.751 → 0.708, 0.1772 → 0.1693). On a 41-frame window a block of
15 gives about three blocks per draw, and that width was stated before the numbers.

**(b) The legs demote.** The card predicted 0 leg demotions and there are many: performer 0
`left_hip` 35, `right_hip` 35, `right_knee` 26; performer 1 `left_knee` 35, `right_knee` 26,
`left_hip` 25, `right_hip` 24, `right_ankle` 5. The cause is not a defect in the rule — the
hips' A–C ray pair sits at 154–156° median through the window, which is past the 150°
ceiling, so the rule is firing on exactly the geometry it describes. **B3's band is raw
byte-identity alone**; the card puts the leg counts in the reported column, so this is a
refuted prediction and never a failed band. What matters is whether the demotions harmed
the legs, and they did not: every leg segment is 0 → 0 frames off by more than 15 %, the
spreads are unchanged to a few tenths of a per cent, and the delivered leg placement moves
by ≤ 0.9 mm at the median with intervals spanning zero.

**(c) `MAXIMUM_INTERPOLATED_GAP_FRAMES` and `RAY_PAIR_CONDITIONING_CEILING_DEG` could not be
selected on synthetic truth**, §4. Both are the card's own selection method refuted by
measurement, not bands moved.

### 7.8 The one real regression, and why B4 cannot settle it

**Performer 1's delivered `Neck` moves further from the RAW captured neck**: p95 105 → 367
mm, max 291 → 452, and 9 → 19 frames over 100 mm. Every one of those frames is inside the
window (ids 89–113). The median *improves* (22.9 → 20.5 mm); it is purely a tail.

The mechanism is visible in the diagnostics: performer 1's `neck` is demoted on 20 frames
and rejected on 1, so on those frames the delivered neck follows a recovered point rather
than the triangulated one, and the trunk chain D7b aims at the captured neck carries it.

**First, what the number actually is.** D7b aims the trunk chain at the *smoothed* captured
neck, and the D3 closure clause proves the delivered GLB matches forward kinematics of the
track it carries to 5e-7 m. So the delivered `Neck` sits on the smoothed neck to within the
trunk-length floor, and **B4's neck figure is, to that precision, `‖smoothed neck − raw
neck‖` — the size of the correction the repair applied to the neck**, not a distance by
which the delivered neck misses a known answer. 452 mm of correction on a 172° pair is
entirely plausible: the shoulder line on those same frames moved by 386 mm (543 → 157 at
its worst). Read as "the delivered neck is 452 mm out" it would be alarming; read as what it
is, it is the repair working at full amplitude on the frames it was built for. The open
question is only whether it moved it the *right way*.

**B4 cannot say whether this is a regression.** Its reference is the raw triangulated point,
and on a demoted slot the raw point is *precisely the one the step judged unreliable* — a
two-view, ill-conditioned estimate at 172°. So on demoted slots B4 is structurally biased
**against** the repair: agreeing with the discarded point scores well and disagreeing scores
badly, whatever the truth is. That is a blindness of the instrument, not a result, and it is
the reason the card lists B4 as REPORTED. The photographs are the arm that is not
so biased, and there performer 1's torso on the window reads −0.0054 [−0.0296, 0.0166] —
spanning zero, so they do not settle it either.

**What would settle it** is not in the card: the delivered neck against a neck the A–C pair
did not determine — a held-out-camera neck, or marker data. The coordinator should treat
this as open. It is the single figure in this step that could be a real cost, and it is
stated rather than absorbed into a favourable average. Against it, on the same reference and
the same frames, the hands improve substantially: `LeftHand` p95 159 → 120 (performer 0) and
158 → 102 (performer 1), and `LeftLowerArm` 261 → 188 on performer 1.

### 7.6 Things this branch did not do

* **`tools/head/head_gate.py` was NOT rerun**, and the reason is §7.3: it scores a
  head-lane npz this build does not write, so rerunning it would reproduce its own figures
  and report a verdict about a head this step cannot have moved. The card asked for it to
  be "rerun and reported"; it is reported as structurally blind instead, with the
  command unchanged should the coordinator want the figures on the record anyway.
* **`tools/compare/d3_skeleton_gate.py` was NOT rerun as a whole.** Its closure clause —
  the one that matters, and a within-build check D8 cannot excuse — was recomputed inline
  in `d8_occlusion_gate.py` on BOTH builds from the same two sources the gate reads
  (`glb_joint_positions` and `forward_kinematics_positions` on `skeleton_for_track`).
  Running the whole gate rebuilds a third delivery into a committed instrument's own
  output directory, which this branch would not do.
* **`ladder.py` was not edited and `ladder.py` / `status.py` were not run**, per the brief.
  To register the extractor: `from extractors.d8_occlusion import x_occlusion_repair`,
  routing the `capture_` keys to rung 4, the `placement_` keys to rung 7 and the
  `silhouette_` keys to rung 1. `python3 tools/compare/extractors/d8_occlusion.py`
  self-checks that every `VISUALS` bar key resolves.
* **`docs/parity-board.html` was not touched** — the board is updated in the same pass as
  the plan, and neither is a branch's to edit.
* **Nothing was written under `artifacts/commercial-multiview-soma77/`.** The D8 delivery
  is at `artifacts/compare/d8-occlusion/delivery/`. If D8 merges it has to be rebuilt in
  place and checked byte-identical to this branch's build, exactly as D7 and D7b did.
* **The branch was not pushed and not merged.**
