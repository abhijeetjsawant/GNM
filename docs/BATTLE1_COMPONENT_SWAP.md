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

## The swap

Our `triangulate_point`, unchanged, fed MAMMA's 2D landmarks and its visibility as
the confidence. Reference is MAMMA's own fitted mesh — not truth, but it is what
*its* 2D produced through *its* fitter, so the distance measures how far our
geometry lands from theirs given identical input. 30 frames spanning the take, both
subjects.

| subject | landmarks triangulated | distance to MAMMA's own fit | p90 |
|---|---:|---:|---:|
| 0 | 291 / 512 | **7.6 mm** | 16.4 mm |
| 1 | 342 / 512 | **7.0 mm** | 14.5 mm |

**Our geometry, given MAMMA's 2D, lands 7.3 mm from MAMMA's own fit.**

Against the same reference, our geometry given **our** 2D disagrees by **24.7 mm of
per-frame spread plus 33.9 mm of per-joint systematic bias**
(`BATTLE1_BODY_PROFILE_VS_MAMMA.md`).

## What that settles

**The gap is the 2D, not the geometry.** Change only the detector and our
reconstruction lands within 7 mm of MAMMA's; keep our detector and it is out by
25–34 mm. One swap, one variable, and it separates in a single measurement what
four instruments this session could not.

It also retires an open question. Increment 6 and the fixture spent a day trying to
price whether our *estimator* had headroom left. It does not have much: given good
2D it already reproduces a 13.5 mm-class system to 7 mm.

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
