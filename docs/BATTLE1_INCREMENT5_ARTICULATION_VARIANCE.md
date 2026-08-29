# Battle 1, increment 5 — is the fitted hand moving, or thrashing?

The hand fit measured 35.0 mm held-out
(`BATTLE1_INCREMENT5_HAND_FIT_RESULT.md`). One number in that report was left
unresolved: **angle standard deviation of 18–24° per degree of freedom, against
MAMMA's 0.098 rad = 5.6°.** Three to four times more variable. Either the fit is
tracking real finger motion, or it is thrashing between frames and the held-out
number is being carried by a minority of well-observed joints.

This document resolves that. It is written **before our numbers exist** — the
MAMMA reference and the pass/fail bands below were fixed first, and the run that
produces our figures was already executing when they were written.

## The old comparison was never valid

MAMMA's 0.098 rad is the per-DoF spread of **SMPL-X hand pose parameters**:
axis-angle triples in a space whose origin is MANO's mean hand and whose
distribution is shaped by a learned pose prior. Ours is the per-DoF spread of
**MHR Euler channels**, where a hinge carries its joint's entire rotation in one
number and no learned prior compresses anything.

These are not the same quantity, and no ratio between them means anything. A
hinge that swings 60° reads ~20° of spread on its single channel; the same swing
distributed over an axis-angle triple with the mean subtracted reads less. The
comparison was confounded from the moment it was written down. What follows
**replaces** it rather than explaining it.

## The measurement

Fingertip **position**, in a wrist-local frame, is gauge-invariant: it does not
care how the rotation was parameterised, so it is directly comparable between the
two systems.

The frame is built **identically in both**, from joint positions only, so that
neither estimator's solved rotation leaks into the result:

- origin at the wrist;
- first axis toward the mean of the four knuckles;
- third axis the palm normal, from that axis crossed with (index1 − pinky1);
- second axis completes the right-handed set.

Wrist translation and wrist orientation both vanish, which is correct and matched:
SMPL-X's hand pose parameters exclude the wrist too (it is body joint 20), so both
sides are measuring *articulation* and nothing else.

Three numbers, over the five fingertips:

| | |
|---|---|
| **amplitude** | temporal standard deviation of local tip position, mm and % of the wrist→middle1 distance |
| **jitter** | median norm of the discrete second difference, mm — high-frequency content, the signature of thrashing |
| **roughness** | jitter ÷ amplitude, dimensionless |

Amplitude answers *is it moving*. Jitter answers *is the motion smooth*. Real
articulation is low-frequency; a solve thrashing between frames is not. Hand size
differs between MHR's default hand and MAMMA's shape-fitted SMPL-X hand, so
amplitude is also reported normalised.

## The reference, measured

MAMMA's retained fit on this same fixture — same four cameras, same 150 frames —
read from `pred_joints` in `verts_joints_body_id-*.npz`. The SMPL-X index layout
was verified structurally rather than recalled: exactly the 15 hand joints and 5
tips of each side fall within 15 cm of the matching wrist, and every finger chain
is monotone in distance from it.

| | hand | amplitude | jitter | roughness |
|---|---:|---:|---:|---:|
| subj0 left | 108.9 mm | 21.0 mm (19.3%) | 0.19 mm | 0.009 |
| subj0 right | 107.2 mm | 32.2 mm (30.0%) | 0.26 mm | 0.008 |
| subj1 left | 119.0 mm | 22.8 mm (19.1%) | 0.18 mm | 0.008 |
| subj1 right | 117.1 mm | 26.5 mm (22.6%) | 0.19 mm | 0.007 |

Two things follow immediately. **MAMMA's hands are genuinely articulating** — 19
to 30% of hand length of fingertip excursion is a hand that opens and closes, not
a frozen prior. And **MAMMA's hands are extraordinarily smooth**: a fifth of a
millimetre of frame-to-frame acceleration, roughness under 0.01.

## Pre-registered verdict

Bands fixed before our numbers were known:

| verdict | condition | what it means |
|---|---|---|
| **trustworthy** | jitter ≤ 0.52 mm (2× MAMMA's worst) **and** amplitude 10–65 mm | the angle variance was gauge; the motion is coherent |
| **collapsed** | amplitude < 10 mm | the fit is not moving; a static hand would also pass a reprojection gate |
| **thrashing** | jitter ≥ 0.78 mm (3× MAMMA's worst) | confirmed; next lever is `smooth_weight` or locking redundant proximal channels |
| **unresolved** | anything between | report as unresolved, claim neither |

A caveat that holds whichever way it lands: **MAMMA's low jitter is partly its
temporal prior, and a smooth answer can still be a wrong one.** This measurement
separates *coherent* from *thrashing*. It says nothing about accuracy — that
claim rests on the 35 mm held-out number and nothing else.

## Result — **thrashing, confirmed on all four hands**

Amplitude and jitter are in the wrist-local frame (articulation only). The two
right-hand columns remove only wrist *translation*, so they also carry wrist
rotation — see the section below on why that distinction turned out to matter.

| hand | length | amplitude | jitter | roughness | ampl. +wrist | jitter +wrist |
|---|---:|---:|---:|---:|---:|---:|
| **ours** subj0 l | 82.1 mm | 48.8 mm (59.5%) | **26.11 mm** | 0.535 | 108.1 mm | **50.65 mm** |
| MAMMA subj0 l | 119.0 mm | 22.8 mm (19.1%) | 0.18 mm | 0.008 | 107.7 mm | 3.20 mm |
| **ours** subj0 r | 82.1 mm | 51.4 mm (62.6%) | **31.53 mm** | 0.613 | 116.0 mm | **81.85 mm** |
| MAMMA subj0 r | 117.1 mm | 26.5 mm (22.6%) | 0.19 mm | 0.007 | 114.6 mm | 3.13 mm |
| **ours** subj1 l | 82.1 mm | 58.9 mm (71.8%) | **30.06 mm** | 0.510 | 130.1 mm | **119.67 mm** |
| MAMMA subj1 l | 108.9 mm | 21.0 mm (19.3%) | 0.19 mm | 0.009 | 68.8 mm | 1.62 mm |
| **ours** subj1 r | 82.1 mm | 53.1 mm (64.6%) | **35.21 mm** | 0.664 | 121.5 mm | **65.18 mm** |
| MAMMA subj1 r | 107.2 mm | 32.2 mm (30.0%) | 0.26 mm | 0.008 | 112.1 mm | 2.98 mm |

**26–35 mm of fingertip jitter against MAMMA's 0.18–0.26 mm**, on a pre-registered
thrashing threshold of 0.78 mm. Between 100× and 190×. Not close to the band, not
ambiguous, and not one bad hand. The angle variance was **not** gauge freedom —
the fingertips genuinely jump between frames. Amplitude is 60–72% of hand length
against MAMMA's 19–30%, which is not a hand articulating; it is a hand flailing.

### The wrist-local frame was hiding half of it

The knuckles hang rigidly off the wrist, so the local basis turns with the wrist.
That is exactly what makes it the right frame for comparing *articulation* — and
it also makes it blind to a thrashing wrist. The fit's temporal term covers
`state[:, 3:]`, the joint angles; the three wrist-orientation parameters per frame
have **no prior at all**.

Removing only wrist translation exposes it: **50–120 mm of jitter, against
MAMMA's 1.6–3.2 mm.** The wrist block is thrashing *harder than the fingers*. Any
fix that raises `smooth_weight` alone would drive the local-frame number into
MAMMA's range and leave the hand still flailing in world space. Both blocks need
the term, and `_jacobian_sparsity` needs the matching three-frame span over the
wrist columns or the solve misconverges silently.

### Two findings that fell out of the comparison

**Our body track agrees with MAMMA's to 19–29 mm at the wrists.** Subject pairing
was done by wrist separation in the shared world frame, and it is unambiguous —
19 mm and 29 mm for the true pairing against 1.19 m and 1.20 m for the swap (our
subj0 is MAMMA's `body_id-01`; the ordering is crossed). That number was not the
point of the exercise, but it is the first direct check of our body reconstruction
against MAMMA's on the same footage, and it is a good one.

**MHR's hand root is not SMPL-X's wrist.** Our phalanges are the right size — the
knuckle-to-fingertip span is 97% of MAMMA's — but our wrist→knuckle offset is
7.4–8.6 cm against MAMMA's 10.4–11.9 cm. The whole 15% chain-length difference
sits in that one segment. It is a joint-definition difference, not a scale error,
and it is the same class of discrepancy Battle 0 found between Apple Vision and
MediaPipe. It matters when the fingers are wired onto the body track, because the
chain is driven from *our* wrist estimate.

### Agreement with MAMMA's fingers: 43–88 mm

Anchoring at the knuckle centroid and normalising by knuckle-to-tip span — which
removes the wrist-definition difference above — our fingertips sit 43.4, 75.4,
88.2 and 71.0 mm from MAMMA's. That is roughly the length of a finger. Consistent
with the thrashing, and clearly labelled as **agreement with a reference system,
not accuracy**: MAMMA is not ground truth here, and its own two-person hand error
is ~48 mm.

## Why — the temporal term is 1.9% of the objective

Evaluated at the fitted solution for subj0's left hand:

| block | residuals | sum of squares |
|---|---:|---:|
| reprojection | 8,212 | ~62,600 |
| temporal | 3,996 | 1,185 |

`smooth_weight` multiplies a second difference **in radians** while the data
block is soft-l1 **in pixels**. Those units were never reconciled, so the nominal
weight of 2.0 buys the temporal prior under 2% of the cost. It cannot resist
anything. The measured median second difference is 5.69° per DoF per frame.

The obvious guess — that the thrashing concentrates in frames the cross-view gate
left evidence-poor — is **wrong**, and worth recording because it would have sent
the fix in the wrong direction:

| | jitter, evidence-poor third | jitter, evidence-rich third | correlation |
|---|---:|---:|---:|
| subj0 left | 29.6 mm | 27.5 mm | **+0.15** |
| subj0 right | 28.9 mm | 44.3 mm | **+0.32** |

The correlation is *positive*: well-observed frames thrash as much or more. So
this is not an under-determined-frame problem to be solved with better gating. The
fit is chasing per-frame detector noise everywhere, on a ~20 px hand, with no
prior strong enough to average it out. That predicts something specific and
testable: **raising the temporal weight should reduce held-out error, not just
smooth the motion**, because noise independent across frames averages down. If
held-out error instead degrades, the smoothing is destroying signal and the
weight is the wrong lever.

## What this does to the 35 mm

It does not retract it, and it does not rescue the motion.

Leave-one-camera-out is a **per-frame** test: it asks whether a frame's pose,
fitted from three views, predicts the fourth. A solution can pass that in every
frame and still jump between frames, because nothing in the test looks across
time. 35 mm remains an honest statement about view generalisation. It was never
a statement about temporal coherence, and the two must now be quoted separately:

- **view generalisation: 35.0 mm held-out** — stands.
- **temporal coherence: fails** — 26 mm of frame-to-frame acceleration against a
  0.78 mm threshold.

The increment's headline should therefore be read as *the geometry is right and
the motion is not yet usable*, not as a pass.
