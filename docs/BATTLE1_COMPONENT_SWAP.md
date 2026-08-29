# One component at a time — swapping MAMMA's 2D into our geometry

The whole session argued about whether our gap to MAMMA is the **detector** or the
**fitter**, using instruments that could not separate them: reprojection sees only
our own detections, epipolar disagreement sees only the cross-view-incoherent
component, and the synthetic fixture's coherent bias turned out not to transfer to
real footage.

All of that is the same disease. **If part A is judged while B, C and D are also
new, nothing about A has been measured.** The remedy is substitution — hold
everything else fixed and change one thing.

## It costs nothing, because the interface already exists

MAMMA's retained run has a per-camera 2D artifact, and it is richer than expected:

```
landmarks     (150, 2, 512, 3)    frames, subjects, landmarks, (x, y, log sigma)
visibilities  (150, 2, 512)       MammaNet's visibility channel
```

Native 3840×2160 coordinates, and **already split by subject** — so a swap holds our
association stage out of the comparison as well. Using it needs **none of MAMMA's
code**: it is a retained output, read the same way its 3D fit already is as a
reference.

Two things worth noting from the file itself. The third channel is negative
throughout, median −4.89 — a **log sigma**, exactly the per-landmark uncertainty
this project has spent a day trying to price. And **64% of landmark-views carry a
visibility below 0.5**: MammaNet's own answer to how much of a 512-point body
surface any one camera can see, and the same order as the 60–77% our geometric veto
discards on fingers.

And a third, which may matter more than either. With **uniform** confidence all 512
landmarks triangulate to ~10 mm — **including the 64% MammaNet flags as invisible**.
Its predictions for occluded landmarks are *amodal and good*. SOMA-77's predictions
for occluded landmarks are the confidently-wrong ones the entire cross-view veto
exists to filter. **Amodal quality on occluded landmarks is a detector property
nobody here has priced**, and a large share of the 25–34 mm may live in it.

## The swap

Our `triangulate_point`, unchanged, fed MAMMA's 2D landmarks and its visibility as
the confidence. Reference is MAMMA's own fitted mesh — not truth, but it is what
*its* 2D produced through *its* fitter, so the distance measures how far our
geometry lands from theirs given identical input. 30 frames spanning the take, both
subjects.

### First pass, and why its metric was wrong

The first version compared each triangulated point to the **nearest vertex** of a
10,475-vertex mesh and read 7.6 / 7.0 mm. That metric cannot see tangential error:
a point sliding along the surface is always near *some* vertex. Measured
inter-vertex spacing is 5.4 mm median, so a purely tangential error reads at most
~2.7 mm — the 7.3 mm was above that ceiling, so the metric was not saturated, but
sideways error was invisible to it regardless.

**`verts_512.pkl` turns out to be on disk** — a (512, 10475) regressor whose rows
sum to 1, so MAMMA's own landmark positions come exactly from its fitted mesh. No
approximation, and it also un-confounds the two channels the first pass credited
together, since it fed MAMMA's `visibilities` as the confidence.

| confidence fed to our solver | subject | landmarks triangulated | error vs MAMMA's own landmarks | p90 |
|---|---:|---:|---:|---:|
| MAMMA's visibility | 0 | 291 / 512 | 9.7 mm | 23.2 mm |
| MAMMA's visibility | 1 | 342 / 512 | 8.2 mm | 19.4 mm |
| **uniform** | 0 | **512 / 512** | 11.1 mm | 31.9 mm |
| **uniform** | 1 | **512 / 512** | 9.2 mm | 24.5 mm |

**Our geometry, given MAMMA's 2D, lands 8.9 mm from MAMMA's own landmarks — and
10.2 mm at full coverage with no visibility gating at all.**

Two things follow. The exact correspondence moves the number from 7.3 to 8.9 mm, so
the nearest-vertex metric was understating by about a fifth rather than
catastrophically. And **the survivorship caveat is answered**: taking every one of
the 512 landmarks, including the ones MammaNet marks as invisible, costs only
1.3 mm. The result was not resting on the well-seen subset.

### The visibility channel is worth 0.16 mm, not 1.3 — the rest was composition

That 1.3 mm compares 8.9 mm over the ~317 landmarks that survive gating against
10.2 mm over all 512 **including the hard tail the other arm dropped**. Different
populations. Scored on identical landmarks:

| | same 18,972 landmarks |
|---|---:|
| MAMMA's visibility as confidence | **8.83 mm** (p90 21.2) |
| uniform confidence | **8.98 mm** (p90 22.8) |
| **the channel is worth** | **+0.16 mm** |
| the 11,748 landmarks it *dropped*, under uniform | 12.62 mm (p90 36.9) |

**Almost all of the 1.3 mm was the composition shift.** MammaNet's visibility does
identify genuinely harder landmarks — the ones it drops are 1.4× the median error of
the ones it keeps — but **used as a weight it buys 0.16 mm**, and dropping them
improves the reported median by removing hard cases, which is survivorship rather
than accuracy.

This is the first pricing of a visibility channel on **real** data rather than a
synthetic noise model, and it does not overturn the fixture — it lands *between* the
corrected predictions of 2.75 mm (mixture) and 1.19 mm (lognormal), on the low side.
If anything it mildly favours the lognormal, which is the noise model under which
**sigma outranked visibility**. The corrected §0 of the fixture doc said the ordering
could not be determined; this nudges it, and does not settle it.

Against the same reference, our geometry given **our** 2D disagrees by **24.7 mm of
per-frame spread plus 33.9 mm of per-joint systematic bias**
(`BATTLE1_BODY_PROFILE_VS_MAMMA.md`).

## What that settles

**Triangulation is not the bottleneck.** Change only the detector and our
reconstruction lands within 9–10 mm of MAMMA's; keep our detector and it is out by
25–34 mm. One swap, one variable, and it separates in a single measurement what four
instruments this session could not.

**Stated deliberately narrowly.** It licenses "triangulation is not the bottleneck",
not "the estimator has little headroom left" — the two are different claims and only
the first is measured. The swap stops at `triangulate_point`. It never exercises
`solve_sequence_positions`, the limb-length constraint, or the temporal smoother —
and §7g of the fixture doc measured that smoother imposing **4.4 mm of median and
34.9 mm of p95 lag at realistic acting speeds, with perfect 2D**. That is estimator
error this swap cannot see and does not refute.

It does, though, **reconfirm the project's original thesis by a cleaner method.**
The research memory's second fact, written before any of this, was "MAMMA's accuracy
is its detector, not its maths". That was an inference from its ablations. This is
the same conclusion by substitution on our own data.

## What it does not settle, stated plainly

**Coverage, and it is the important caveat.** Only 291–342 of 512 landmarks survive
our confidence and inlier gates. The survivors are the well-seen ones, so 7.3 mm is
measured on the easy subset — the same survivorship pattern this project has now hit
four times. Raising coverage would almost certainly raise the number.

**The reference is a fit, not truth.** MAMMA's mesh is regularised, so part of the
7.3 mm is its fit pulling away from raw triangulation rather than our error. That
cuts in our favour, but it means 7.3 mm is not an accuracy figure.

**It compares geometry, not the whole downstream.** The swap holds association out
(MAMMA's 2D is pre-associated) and stops at triangulation — it does not exercise
`solve_sequence_positions`, the limb-length constraint or the temporal prior.

**And not every component can be swapped this way.** Our 19-joint contract and
MAMMA's 512 surface landmarks do not meet at any other interface, so a
stage-by-stage ladder is not available. The one interface that lines up happens to
be the one that mattered.

## The reverse swap, and why it is not free

Feeding **our** 2D into **MAMMA's** fitter would complete the 2×2 and isolate the
fitter directly. It needs MAMMA's code to run, which raises two things the plan
already cares about: its licence bars producing artifacts for commercial purposes,
and Battle 4 requires a clean-room reimplementation *by someone who has not read
MAMMA's source*. Running is not reading, but whoever does it should not be whoever
implements Battle 4.

Given the result above, it is also no longer the urgent question.
