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

## Result

Pending.
