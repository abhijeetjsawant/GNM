# The confidence channel is a gate, not a weight

**Snapshot: 2026-08-30.** This corrects a MEASURED entry in
`docs/BODY_LANE_PLAN.md` §3 and the corrected-results table in
`docs/BATTLE2_SYNTHETIC_TRUTH_FIXTURE.md` §0. It was found by the plan's own
tripwire — *"if a new measurement contradicts a MEASURED entry, stop and
re-verify the instrument before proceeding"* — and it is the **eighth** instance
of this lane's recurring defect: a correct measurement carrying a claim it does
not support.

**The decision that entry supported does not change.** Battle 4 still keeps
MammaNet's full μ/σ/visibility triple. Stated up front so a future session does
not re-litigate it on the strength of the correction.

---

## 1. The finding

The published comparison prices "a σ head" against "a visibility head" and
concludes the two **cannot be ranked**, because the ordering reverses between two
noise models that both fit our data.

The two arms were never two channels. They are **the same information at two
different positions relative to one threshold.** In
`scripts/measure_detector_outputs.py` the visibility arm sets bad observations to
`FLOOR * 0.5` = **0.125**, which is *below* `minimum_confidence = 0.25`, so those
observations are **dropped from `triangulate_point` entirely**. The σ arm clips to
**0.250001**, *above* the same threshold, so nothing is dropped and the value can
only arrive as a weight.

Hold the information fixed and move only the gate, and the "ranking" disappears:
the informed channel is worth **+3.09 mm** when it drops and **+0.90 mm** when it
merely re-weights — and the re-weighting arm lands on the published σ arm **to the
decimal**, because they are the same two-point weight field.

So the published pair does not measure σ against visibility. It measures
**dropping against down-weighting**, and dropping wins.

---

## 2. The mechanism, in code

Three facts in `src/autoanim_gnm/commercial_multiview.py`, each checked.

**(a) The sequence solve usually does not run at all.** Lines 1019–1029:

```python
direct = np.isfinite(positions).all(axis=2)
seen = (np.isfinite(observations[..., :2]).all(axis=3)
        & (observations[..., 2] >= minimum_confidence)).transpose(0, 2, 1)
support = seen.sum(axis=2)
candidate = (~direct) & (support >= 1)
if not candidate.any():
    return positions.copy(), np.zeros_like(direct)
```

`candidate` is only the slots that **failed** to triangulate. When there are
none, the function returns triangulation and `least_squares` is never called.

**(b) Even when it runs, its answer is discarded on every slot that
triangulated.** The function ends:

```python
output = positions.copy()
output[recovered] = solved[recovered]
return output, recovered
```

This is deliberate and documented in the source — the solve is a recovery pass
for missing evidence, not a re-estimator of present evidence, because the
temporal and limb terms were measured outvoting good geometry by a median of
11–14 mm and up to 700 mm. The consequence is not documented: **the reprojection
objective — and therefore every per-observation weight in it — reaches the output
only on `recovered` slots.**

**(c) The recovery gate is itself blind to outliers.** `recovered` requires
`max(errors) <= robust` across **all** `seen` views, including the bad
observation that caused the failure. So a failure *caused by an outlier* cannot
be recovered by construction; only failures caused by a missing view can.

**Where a confidence therefore still does work:** inside `triangulate_point`,
which weights DLT rows by `sqrt(max(conf, 1e-6))` (lines 324–336) and uses
confidence in eligibility and inlier-subset scoring. That path is live. It is
just not the path the σ arm was believed to be testing.

**Three mechanisms, not two channels:**

| mechanism | where it lives | live? |
|---|---|---|
| **eligibility gate** — confidence below `minimum_confidence` drops the observation | `triangulate_point`, line ~319 | **yes, and it is the largest term** |
| **DLT row weighting** — `sqrt(conf)` on the linear system, plus inlier-subset scoring | `triangulate_point`, lines 324–336 | **yes, and it is what the σ arm actually measured** |
| **robust-loss weighting** — `observed_weight` in the soft-l1 reprojection residual | `solve_sequence_positions` | **no — verified inert** |

---

## 3. What was measured

**Direct differential on the third mechanism.** One noise realisation, one
confidence field, `weight_before_loss` flipped, with a spy confirming the solver
received `True`. Output position arrays **bit-identical**; `max |Δ| = 0.000000000
mm`; NaN patterns identical.
*Instrument:* `reconstruct_multiview` on `artifacts/synthetic-truth`, 84 frames,
2 subjects, 17 scored joints, 4 cameras.
*Blindness:* **this fixture cannot exercise the sequence solve at all**, and the
reason is more specific than "nothing fails". Slots *do* fail to triangulate —
168 of 1,596 per subject — but they fail with **zero retained rays** (`support`
= 0), and `candidate = (~direct) & (support >= 1)` excludes exactly those. So
§2(a)'s early return fires anyway. **What exercises the solve is a single-ray
slot, not occlusion as such**: a joint seen by exactly one camera, which has
evidence but not enough to triangulate. This fixture has none.
*(Correction owed to Fable, who probed `candidate` directly rather than accepting
the "full visibility" story. The σ arms show `candidates = 0, recovered = 0`
under both noise models and both orderings; the visibility arm produces exactly
one candidate, and recovers it.)*
On real footage the solve does fire — 83% of the joints it recovers were seen by
exactly one camera — so the ordering fix is **untested, not disproven.**

**Gate-versus-weight, same information, one threshold moved.** Five seeds, paired
noise realisations, mixture noise 2.75 px clean / 7% bad at 36 px.

| arm | MPJPE | sd | buys | dropped | recovered |
|---|---:|---:|---:|---:|---:|
| no channel | 11.92 mm | 0.66 | — | 0 | 0.00% |
| informed drop, bad = 0.125 **below** gate | **8.84 mm** | 0.13 | **+3.09** | 794 | 0.06% |
| informed weight, bad = 0.2501 **above** gate | 11.02 mm | 0.58 | +0.90 | 0 | 0.00% |
| σ, the published arm (above gate) | **11.02 mm** | 0.58 | **+0.90** | 0 | 0.00% |
| **shuffled** drop, below gate *(control)* | 15.28 mm | 0.46 | **−3.35** | 794 | 0.09% |

Denominator: 11,424 observations per arm (4 cameras × 84 frames × 2 subjects × 17
scored joints); MPJPE over all frames × all scored joints, coverage 100.0% in
every arm, unrecovered slots counted as failures rather than absences.

*On the baseline moving:* 11.92 mm here against the published 11.62 mm is **five
seeds against three**, same fixture, same noise model, same scored joints — inside
the 0.66 mm seed spread. Nothing about the baseline is being contested; only the
sentence attached to the arms above it.

Two things make this readable rather than suggestive:

**The third and fourth rows land together to the decimal**, including their
standard deviations. That is the confirmation, not a coincidence: `{0.2501, 0.99}`
and `{0.250001, 0.99}` are the same weight field, so the published σ arm *is* the
above-gate arm under another name.

**The control fails, and fails in the right direction.** The shuffled arm drops
**the same 794 observations by count**, chosen at random, and *loses* 3.35 mm. So
the gate's +3.09 mm is not the value of dropping observations — it is the value of
knowing **which** to drop. No constant passes this band: dropping 7% blindly is
3.35 mm worse than not dropping at all.

*Blindness of the whole table:* the synthetic fixture contains none of
calibration error, lens distortion, sync error, soft-tissue artefact or
joint-definition error — all five first-order on real footage. It prices a
mechanism, never an accuracy.

---

## 4. What is withdrawn, and what survives

**Withdrawn:** *"σ and visibility cannot be ranked; the ordering reverses between
two noise models."* The reversal is not evidence about two channels. The arms
differ by mechanism — one gates, one weights — so the comparison violates this
lane's **same-denominator** rule at the level of the estimator path, not the
population. The specific numbers (σ +0.95 / +2.15, visibility +2.75 / +1.19,
wrong channel −3.6) are correct as measurements of what their arms did; the
sentence attached to them is not.

**One half of it is salvageable, and should not go out with the rest.** *(Fable's,
and it is the sharpest thing either advisor said about the correction.)* Under the
**lognormal** model the comparison is legitimate, because there the visibility arm
is a **quantile-coarsening of the same σ field** — same mechanism, same gate,
strictly less information — and σ still wins (+2.15 against +1.19). That is a real
**within-family** information result. It is only the **mixture** side that is pure
mechanism, because there the two arms straddle the gate. So the honest statement
is not "the ordering reverses" but: *within a family, finer information wins;
across the two published arms, the mixture comparison was measuring a gate.*

**Survives, and load-bearing:**

- **Battle 4 keeps the full μ/σ/visibility triple** — but see the ceiling below
  before pricing the visibility head. What this correction shows is that of the
  three mechanisms, only **gating** is live; it does **not** show that gating is
  worth much on real footage.

> ### ⚠ CEILING, measured 2026-08-30 — read before building a visibility head
>
> **A visibility head used as a gate *is* a veto**, and a perfect oracle veto was
> subsequently measured on real footage at **+0.05 mm median / +1.47 mm p95**
> (`tools/swap-harness/veto_ceiling.py`, plan §3). So the **+3.09 mm** below is a
> synthetic number with **no real-footage counterpart**, and the FPR spec derived
> from it prices a win that does not appear on this rig.
>
> **Why the two disagree, and it is not a defect in either.** The synthetic mixture
> puts 7% of observations at **σ = 36 px** — squarely inside the 14–45 px band where
> `triangulate_point`'s 14 px inlier gate is weakest, so an oracle drop helps.
> The real debiased residual is **bimodal**: 92.9% under 14 px, **3.6% in that
> ambiguous band, 3.5% at ≥45 px** where the gate discards them for free. Real
> tail events are either inside the threshold or trivially gated; the synthetic
> ones were engineered into the gap. **Only 1.9% of flagged real observations were
> ever consumed by the inlier subset.**
>
> **Scope of the ~0:** it is a property of *this rig's redundancy* — 4 cameras, high
> co-visibility, and even so **35.3% of slots have only two eligible views**, where
> the gate is helpless. On occlusion-heavy owned footage that fraction grows and the
> veto's value grows with it. The verdict holds for the **≥3-eligible-view regime**.
- **A wrong channel costs ≈ −3.6 mm** under both noise models (published), and
  **−3.35 mm** when reproduced here. Shuffled labels drop the same *number* of
  observations and still lose ground, so the gate's value is in knowing *which*
  to drop, not in dropping. That is the control that can fail, and it fails
  correctly — the one construct in this comparison that was never at risk of
  being an identity.
- **The +0.16 mm real-data visibility result** in the plan is consistent with
  this and always was: it was measured as a **weight**, on identical landmarks.
  Weight ≈ dead, gate ≈ alive — two instruments, same story.

---

## 4a. How good must a visibility head be? — the spec this hands Battle 4

**Read the ceiling box in §4 first: on real footage this whole prize measures ~0.**
What follows is the *shape* of the constraint — which axis binds — and that shape
survives; the millimetres do not transfer.

The oracle gate buys +3.09 mm and a random gate of the same size costs −3.35 mm,
so the number Battle 4 actually needs is **where the curve crosses zero.** Below
that line a visibility head makes the reconstruction *worse than no channel at
all.* Three seeds, same fixture and noise model, dropped observations set to 0.125
(below the gate).

| head quality | MPJPE | sd | buys |
|---|---:|---:|---:|
| no channel *(baseline)* | 11.62 mm | 0.66 | — |
| recall 0.00, fpr 0.00 *(degenerate check)* | **11.62 mm** | 0.66 | **+0.00** |
| recall 0.25, fpr 0.00 | 10.67 mm | 0.25 | +0.96 |
| recall 0.50, fpr 0.00 | 10.12 mm | 0.21 | +1.50 |
| recall 0.75, fpr 0.00 | 9.40 mm | 0.24 | +2.23 |
| recall 1.00, fpr 0.00 *(oracle)* | 8.87 mm | 0.16 | +2.75 |
| recall 1.00, fpr 0.02 | 9.29 mm | 0.12 | +2.34 |
| recall 1.00, fpr 0.05 | 10.09 mm | 0.92 | +1.54 |
| recall 1.00, fpr 0.10 | 11.18 mm | 0.52 | **+0.44** |
| recall 1.00, fpr 0.20 | 15.90 mm | 2.09 | **−4.27** |

*Instrument:* `tools/swap-harness/gate_quality_curve.py`. *Degenerate check:*
recall 0.00 with zero false positives flags nothing and **lands on the baseline to
the decimal** — 11.62 against 11.62. Had it not, the table would be seed noise.

*Reading this against §3:* the `recall 1.00, fpr 0.00` row **is** §3's "informed
drop" arm — the same oracle gate, run at **three seeds here against five there**,
which is why it reads +2.75 against +3.09. Both are inside the 0.66 mm seed
spread; it is one arm at two sample sizes, not two results.

**The two axes are wildly asymmetric, and that is the finding.**

- **Recall is cheap.** Even a head catching only a quarter of the bad
  observations buys +0.96 mm, and the curve is close to linear in recall.
- **False positives are expensive and non-linear.** The break-even sits between
  **10% and 20% false positives** — +0.44 mm at 10%, −4.27 mm at 20%. Past that
  the head is actively harmful, and the variance explodes with it (sd 0.52 → 2.09).

**So the pre-registered spec for a Battle 4 visibility head is a precision
constraint, not a recall target: keep the false-positive rate under ~10%, and
take whatever recall comes.** A head tuned the usual way — maximise recall of
occluded landmarks — walks straight into the harmful half of this table.

*Blindness, and it bounds the number hard in two directions:*

- **Four cameras, on a rig we cannot ship constants off.** At four views a joint
  has little redundancy to spare, which is exactly why a false positive costs so
  much; the break-even moves with camera count and this curve does not predict
  where. The fixture's rig is **MPI-derived and barred from shipping** (plan §7),
  so by this project's own rule — structural go/no-go may cite it, shipped numbers
  may not — **the axis is the finding and the number is provisional until
  re-derived on the owned rig.** Same treatment as the 15° majority gate.
- **The false positives here are iid, which is the friendliest possible structure.**
  `flag = bad | (~bad & (rand < p))` draws independently per camera, frame,
  subject and joint. A real visibility head fails **correlated** — persistently in
  time on one joint, and plausibly across views sharing an appearance failure —
  and correlated false positives can remove several views of the *same* joint at
  once, which at four cameras is the lethal case that iid at 10% almost never
  produces. **So ~10% is a ceiling measured under the friendliest noise, not a
  target.** Under a real head the break-even sits below this bracket, by an
  unmeasured margin.
- **Synthetic throughout:** "bad" is a known label here, and on real footage it is
  precisely the thing being estimated.

---

## 5. Disposition of the uncommitted diff

`src/autoanim_gnm/commercial_multiview.py` carries an uncommitted
`weight_before_loss` flag adding the statistically correct ordering (weight
before the soft-l1 compression, as `fit_hand_sequence` already does).

**Keep the code. Do not flip the default. Do not commit it with its current
comment**, which reads *"Measured on the synthetic fixture before changing the
default"* — that measurement is void, because §3 shows this fixture cannot
exercise the branch.

Correct status: **the ordering fix is right in principle, reaches only recovery
slots, and no instrument for it exists yet.** It belongs in OPEN.

Note also that `scripts/measure_detector_outputs.py` is **committed while the
parameter it passes is not**, so a clean checkout of `HEAD` cannot run it —
it raises `TypeError`. Whatever lands, land them together.

---

## 6. A discrepancy, recorded rather than resolved

`docs/BATTLE2_SYNTHETIC_TRUTH_FIXTURE.md` §0 and the artifact
`artifacts/synthetic-truth/detector-outputs.json` **agree to the decimal on every
row except σ**:

| row | doc | artifact |
|---|---:|---:|
| no channel (mixture) | 11.62 | 11.6237 ✓ |
| visibility (mixture) | 8.87 | 8.8719 ✓ |
| visibility shuffled (mixture) | 15.18 | 15.1794 ✓ |
| **σ (mixture)** | **10.67 (+0.95)** | **11.0495 (+0.57)** |
| **σ (lognormal)** | **12.23 (+2.15)** | **12.1149 (+2.27)** |

The plan of record quotes the doc. Not diagnosed here; flagged because the plan's
§3 numbers do not match the artifact they came from, and because both σ rows
inside the artifact are byte-identical across an ordering that provably does
nothing.

---

## 7. New rows for the instrument table

| instrument | readable to | blind to | banned uses |
|---|---|---|---|
| synthetic fixture through `reconstruct_multiview` | the **eligibility gate** and **DLT row weighting** inside `triangulate_point` | **the sequence solve entirely** — full four-camera visibility means `candidate` is empty and the solve early-returns | pricing anything that acts through `solve_sequence_positions`, including the robust-loss weight ordering and the temporal/limb priors |
| any confidence-channel arm | the mechanism its values actually trigger | the distinction between gating and weighting, unless the gate position is held fixed | **comparing two arms that straddle `minimum_confidence`** — that is a mechanism comparison wearing a channel comparison's clothes |

---

## 8. OPEN — created by this correction

- **What the robust-loss ordering is worth**, on a fixture where the sequence
  solve actually fires. Needs occlusion, i.e. slots seen by fewer than two
  cameras. Nothing on disk currently provides it.
- **Where `minimum_confidence` should sit.** It is now the largest single lever
  measured on this fixture and its value 0.25 has never been swept against
  truth. `scripts/measure_sigma_value.py` already carries a `--floors` sweep
  written for exactly this question and its results were read as a σ result.
- **Whether the gate's advantage survives on real footage**, where "bad" is not a
  known label. The +2.44 mm assumes an oracle. The honest version of this
  question is the plan's own §9a oracle-veto ceiling, and it is now the direct
  continuation of this finding rather than a separate errand.
