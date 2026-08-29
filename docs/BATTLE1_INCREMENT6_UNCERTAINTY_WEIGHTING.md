# Battle 1, increment 6 — does weighting the evidence close the gap?

**Written before the run. No numbers in this document existed when the bands
below were fixed.**

## The question this decides

Increment 5 ended with a hypothesis: every remaining hand problem traces to the
detector, because SOMA-77 emits no per-landmark uncertainty and no visibility, and
MAMMA's MammaNet emits both. If that is right, the next battle is training a
detector, which needs a synthetic-data campaign.

Both advisors rejected it as premature, and for the same reason: it is the shape
of the finger-triangulation mistake — a claim about what is *possible*, made
before ruling out that our own estimator is the waster. Their evidence was ours:

- the largest gains of increment 5 were plain estimator defects a σ-emitting
  detector would not have fixed — a units mismatch worth 98% of the objective, an
  unregularised wrist block, and a degeneracy that froze the hand;
- the "no temporal weight reaches MAMMA's operating point" curve was measured on
  a prior that has since been replaced, and the replacement does not behave that
  way;
- and the σ we said we lacked is **already computed and thrown away**. The gate
  reduces a continuous cross-view agreement distance to one bit.

So the hypothesis gets tested before it gets written down.

## What changed in the code

`cross_view_weights` replaces the hard veto with a Lorentzian inverse variance,
`w = 1 / (1 + (d / 9 px)²)`, on each observation's **median** epipolar distance to
the other views that see it. It enters the fit as `√w` on the reprojection
residual, applied *before* soft-l1 so the robust loss sees `r/σ` rather than `r`.

The veto's failure was measured, not assumed: four cameras make six pairs, and one
disagreeing pair killed both observations. On subj0's left hand that discarded 63%
of the evidence, and **2,790 of the 6,954 rejected observations had their own
median distance under the threshold** — a quarter of all evidence thrown away for
agreeing with everyone except one view.

## Three arms, to attribute the change rather than just observe it

| arm | observations | temporal prior | isolates |
|---|---|---|---|
| **A** | hard veto | angle only, weight 2 | the increment 5 baseline |
| **B** | hard veto | + position, weight 1.0 | what the prior alone buys |
| **C** | **inverse-variance weights** | + position, weight 1.0 | what the weighting adds |

Every arm is scored on **all four hands and all four leave-one-camera-out folds** —
sixteen held-out fits per arm — plus a four-camera fit per hand for amplitude and
jitter. Increment 5's headline moved on a single hand and a single fold, and that
is not a basis for a strategic decision.

## Pre-registered bands

Arm A is known: **35.0 mm** mean held-out, 26–35 mm jitter, 48.8–58.9 mm amplitude.

Arm C is scored against:

| verdict | condition |
|---|---|
| **pass** | mean held-out ≤ 30 mm **and** mean jitter ≤ 2 mm **and** mean amplitude ≥ 20 mm |
| **partial** | mean held-out ≤ 30 mm **and** jitter 2–6 mm **and** amplitude ≥ 20 mm |
| **fail** | mean held-out > 30 mm, **or** amplitude < 20 mm, **or** jitter > 6 mm |

The amplitude floor is MAMMA's own minimum on this fixture (21.0 mm, rounded
down). It is there because of the standing rule this increment adopts.

### The standing rule: no gate a constant can pass

Twice in increment 5 a metric improved because something had stopped working — a
bone-length gate that read 0.00% because bone lengths are constants in this
estimator, and a jitter gate that read 0.05 mm, better than MAMMA's, on a frozen
rest-pose hand at 98 mm of error. So every acceptance band now ships with a
demonstration that a degenerate solution fails it.

**Demonstration for the band above:** the rest-pose hand measured at
`pose_smooth_weight` 1.0 before the warm-start fix scores 98.3 mm held-out and
0.0 mm amplitude. It fails on two independent clauses. The band is not satisfiable
by a constant.

## What each outcome means for the strategy

**Pass or partial** — the estimator was the waster, the detector hypothesis was an
over-generalisation, and the σ we needed was geometric all along. The
synthetic-data campaign keeps its place in the plan but loses its urgency.

**Fail, with the recovered evidence and both priors in** — the hypothesis has
survived its first genuine test rather than being an inference from a confounded
sweep, and "we need a detector that emits μ, σ and visibility" is earned with data.

Either way the plumbing is not wasted: a detector that emits σ is useless until
the fit consumes per-observation weights, and this is that code.

## Result — **FAIL** against the pre-registered band

| | measured | band | |
|---|---:|---|---|
| mean held-out | **33.5 mm** | ≤ 30 | ✗ |
| mean jitter | **4.57 mm** | ≤ 2 (partial 2–6) | partial |
| mean amplitude | **39.0 mm** | ≥ 20 | ✓ |

**Verdict: fail.** The held-out clause misses by 3.5 mm, and jitter lands in the
partial band rather than the pass band. Recorded as it fell.

## But the weighting works, and the failure is not its fault

Paired, on all sixteen cells — same hands, same folds, same held-out observation
set for both arms so that a change in what each arm *trusts* cannot move the test
set:

| hand | A001 | B001 | C001 | D001 |
|---|---:|---:|---:|---:|
| subj0 left | 23.0 → 22.4 | 29.3 → 29.5 | 28.2 → 31.4 | 36.3 → 41.5 |
| subj0 right | 24.9 → 26.7 | 28.0 → **25.0** | 34.3 → **25.5** | 29.3 → 29.5 |
| subj1 left | 39.9 → 42.0 | **91.9 → 60.0** | **63.1 → 35.1** | **80.7 → 41.7** |
| subj1 right | 22.4 → 22.1 | 37.4 → 33.5 | 47.8 → **37.9** | 31.8 → 31.6 |
| | | | | |
| **mean** | | | **40.5 → 33.5 mm** | **−7.1** |
| **worst fold** | | | **91.9 → 60.0 mm** | **−31.9** |

Arm C wins 10 of 16 folds, takes 7.1 mm off the mean, and takes 31.9 mm off the
worst. It also holds 2.3–3.9× more evidence: 4,106 → 10,655 observations on
subj0's left hand, 2,316 → 8,950 on subj1's left.

**And the gain lands exactly where the mechanism predicts.** subj1's left hand is
the evidence-starved one — 2,316 observations under the veto, the fewest of the
four — and it is the hand that improves most, by 28, 32 and 39 mm on three of its
four folds. The veto was starving the hardest hand hardest, because a hand that is
difficult to see is a hand whose camera pairs disagree.

Jitter improves in the same direction: 4.57 mm mean against arm B's 6.70, and
against increment 5's 26–35 mm.

## The like-for-like baseline: 50.3 mm, not 35.0

Arm A finished on all four hands. It changes how the verdict should be read.

| | held-out mean | worst fold | jitter | amplitude |
|---|---:|---:|---:|---:|
| **A** — the shipped baseline | **50.3 mm** | 93.0 mm | 32.18 mm | 52.6 mm |
| **B** — + position prior | 40.5 mm (−9.8) | 91.9 mm | 6.70 mm | 47.3 mm |
| **C** — + inverse-variance weights | **33.5 mm (−16.9)** | **60.0 mm** | **4.57 mm** | 39.0 mm |

**Arm C beats the baseline on 14 of 16 folds, takes 16.9 mm off the mean, 33 mm
off the worst fold, and cuts jitter sevenfold.** The increment moved the hand fit
by a third. That is the result; "fail" is the verdict on a band, and the band was
set against a number that was wrong.

**The band was mis-set, and that is my error, not the experiment's.** It asked for
≤30 mm because I believed the baseline was 35.0 — a 14% ask. Against the real
baseline of 50.3 it was a 40% ask. The band still fails, and it stands as written;
but the reason it looked reachable was that the figure it was calibrated against
did not exist.

(Consistency check: arm A restricted to subj0-left across its four folds is
34.4 mm, against increment 5's 35.0 for the same hand and folds. The small gap is
the warm-start change. The two measurements agree, which is why the four-hand
figure is the one that was missing rather than wrong.)

## A comparison I nearly got wrong

Increment 5's headline of **35.0 mm was subj0's left hand only**, across its four
folds. It was never a four-hand figure. Setting it beside a sixteen-cell mean
would have made arm C look like a 1.5 mm improvement on a baseline that does not
exist.

On the hand the 35.0 mm was actually measured on:

| | subj0 left, four folds |
|---|---:|
| increment 5 baseline | 35.0 mm |
| arm B — veto + position prior | **29.2 mm** |
| arm C — weights + position prior | 31.2 mm |

Both beat it; arm B beats arm C on *this* hand. Which is the same lesson as
before, pointed at myself: **subj0-left is an easy hand, and it is the one every
earlier number came from.** The four-hand mean is 40.5 and 33.5. Arm A is now
running on all four hands so the baseline is a like-for-like figure rather than
an extrapolation from the easiest case.

## Arm D — the same weighting at temporal weight 4

| arm | held-out mean | worst fold | jitter | amplitude |
|---|---:|---:|---:|---:|
| **A** baseline | 50.3 mm | 93.0 mm | 32.18 mm | 52.6 mm |
| **C** weights, prior at 1.0 | 33.5 mm | 60.0 mm | 4.57 mm | 39.0 mm |
| **D** weights, prior at 4.0 | 33.9 mm | **53.0 mm** | **1.44 mm** | 35.7 mm |

**Arm D clears the jitter clause** — 1.44 mm against the pre-registered 2 mm — and
takes another 7 mm off the worst fold, at no cost in mean held-out error. Two of
the three clauses now pass. The band still fails, on held-out alone.

End to end, the increment moves the hand fit:

| | baseline | arm D | |
|---|---:|---:|---:|
| mean held-out | 50.3 mm | 33.9 mm | **−33%** |
| worst fold | 93.0 mm | 53.0 mm | **−43%** |
| jitter | 32.18 mm | 1.44 mm | **−96%** |

Arm D is optimistic by construction — temporal weight 4.0 was chosen after seeing
the sweep, on this fixture. It needs confirming on data it was not selected on.

## What this does and does not settle

Pre-registered reading: *fail, with the recovered evidence and both priors in* →
the detector hypothesis survives its first genuine test.

**It survives, provisionally, and one qualification is load-bearing.** The
temporal weight for these arms was fixed at 1.0 before the sweep finished. The
sweep then found 4.0 gives 1.76 mm of jitter at the same held-out error — inside
the pass band, where 1.0's 4.57 mm is not. So the jitter clause may have failed on
a configuration choice rather than on anything about uncertainty.

**Arm D** — the same weighting at temporal weight 4.0 — is running against the
same band. It is a configuration chosen after seeing the sweep, so it is
optimistic by construction and cannot settle the strategic question on its own;
what settles it is the synthetic fixture, which measures against truth on data no
configuration was selected on.

What is already settled, and does not depend on arm D: **the veto was throwing
away information that mattered, and the estimator was the waster of it.** 7.1 mm
of mean held-out error and 31.9 mm on the worst fold came back from nothing but
keeping observations the old gate discarded. That is not an argument for a new
detector. It is an argument that we had not finished using the one we have.
