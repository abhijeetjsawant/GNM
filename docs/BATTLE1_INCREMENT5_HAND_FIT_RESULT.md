# Constrained hand-chain fit — measured

Status: first measured result, 2026-08-29. Design: `BATTLE1_INCREMENT5_HAND_FIT_DESIGN.md`.
Prior negative result and its amendment: `FINGER_TRIANGULATION_GATE.md`.

## The headline

**Held-out finger reprojection: 6.28 px = 35.0 mm at the subject.**

Leave-one-camera-out: fit the hand on three views, measure how well it predicts
the fourth, which it never saw.

| held-out camera | train obs | test obs | held-out reprojection | at the subject |
|---|---:|---:|---:|---:|
| A001 | 1,067 | 365 | 4.62 px | 25.8 mm |
| B001 | 1,283 | 149 | 3.14 px | **17.5 mm** |
| C001 | 712 | 720 | 9.24 px | 51.5 mm |
| D001 | 1,234 | 198 | 8.14 px | 45.3 mm |
| **mean** | | | **6.28 px** | **35.0 mm** |

Body joints, in-sample, are 2.91 px = 16.2 mm for comparison.

**This lands squarely inside the predicted band.** Before any of it was built,
the estimate was 30–60 mm on free gesture and 60–100 mm during clasp. The
measurement is 35 mm, between MAMMA's 16.9 mm singles and its 47.95 mm
two-person contact.

That clears *pose-plausible for dialogue*: an open hand and a closed fist differ
by 60–80 mm at the fingertips, so grasp state, pointing, spread and coarse curl
sit above the noise. It does **not** clear contact precision, exactly as
predicted, and N5.1's `review_required` rule already covers that case.

## The agreed gate turned out to be vacuous, and I replaced it

Both advisors set the acceptance gate at **finger bone-length stability ≤ 10%**,
against MAMMA's 7.3% on this footage and independent triangulation's failed 226%.

Measured: **0.00%**, on every hand.

That is not a pass. It is a tautology. Bone lengths are *constants* in this
estimator — the entire reason it works where triangulation failed — so they
cannot vary, and a gate they cannot fail measures nothing. The gate was designed
when triangulation was the estimator and did not survive the change of estimator.

Leave-one-camera-out replaces it because it **cannot be satisfied by
construction**. A chain that merely satisfies its own bone lengths while sitting
in the wrong pose will reproject badly into a camera it never saw. It is also the
same shape as MAMMA's own held-out-marker protocol.

## What the fit actually did

Four hands, 150 frames, ~190 s each on CPU.

| | subj0 L | subj0 R | subj1 L | subj1 R |
|---|---:|---:|---:|---:|
| observations kept by the cross-view gate | 37% | 35% | 23% | 40% |
| mean joint angle | 27.7° | 23.0° | 21.5° | 28.8° |
| angle sd per DoF | 19.7° | 18.3° | 18.8° | 24.0° |

**The gate rejects 60–77% of finger observations.** That is high, and it is the
mechanism working as intended: SOMA-77 reports high heatmap confidence on
occluded fingers, and geometry is the only substitute we have for the visibility
channel MammaNet is trained to emit. It also means the fit is running on roughly
a third of the raw finger evidence.

## The number that should worry us

**Articulation variance is 18–24° per DoF. MAMMA's is 0.098 rad = 5.6°.**

Ours varies three to four times more. The agreed gate asked for a *band* around
MAMMA's figure — not collapsed, not thrashing. We are not collapsed; we may be
thrashing. Three candidate causes, untested:

- the temporal weight is too low for observations this sparse;
- gauge freedom in the redundant Euler triples at the proximal joints, which the
  synthetic test already showed (8° angle spread at under 19 mm of position
  error) — in which case the angle variance overstates the pose variance and the
  metric is partly misleading;
- genuine per-frame noise from fitting to ~a third of the observations.

The held-out reprojection is the trustworthy number; this one needs
disambiguating before any claim about motion quality. The obvious next
measurement is fingertip *position* variance rather than angle variance, which
is gauge-invariant.

## Honest limits

- One fixture, one 5-second take, two performers, on footage we may not use
  commercially. Nothing here is ground truth; that is Battle 2.
- Leave-one-camera-out on a 4-camera rig trains on 3, so it measures a *harder*
  configuration than production. Read 35 mm as conservative.
- The C001 fold is twice the B001 fold (51.5 vs 17.5 mm). Per-camera variation
  that large on four folds means the mean is not well determined.
- No qualitative review against the retained MAMMA take yet, which the design
  lists as an acceptance gate.
- No test for the flip-basin failure the design predicts during a clasp.
