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

## Result

Pending — the fit that produces it is running.
