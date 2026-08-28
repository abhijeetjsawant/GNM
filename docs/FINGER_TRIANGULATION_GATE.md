# Can we triangulate fingers from four wide cameras?

Status: **amended 2026-08-29.** The original conclusion — "no, the physics
forbids it" — was **wrong, and overclaimed from a correct measurement.**

What the measurements below actually establish is narrower and more useful:
**independent per-joint triangulation of fingers fails at this framing.** They do
not establish that fingers are unrecoverable. MAMMA recovers them from *these
exact four videos*, which is the counter-example that settles it.

The amendment is at the bottom, under "What MAMMA does differently". Read it
before acting on anything above it.

## Why it was asked

The SOMA-77 detector adopted in Battle 1 increment 4 emits **15 articulated
finger joints per hand**, which the AutoAnim-19 intermediate contract discards.
The 55-joint contract downstream already carries 30 finger joints and the MPFB
rig has the deforming bones. So the fingers looked like free capability.

Two advisors split on it, which is why it was measured rather than argued.

## The tests, in the order they were run

**1. Are the fingers a rigid template hanging off the wrist?** No. Normalised
finger offsets vary with a standard deviation of 0.53 of hand size — they move.

**2. Is that motion signal or argmax jitter?** Temporally coherent: consecutive
frames are ~4× more similar than random pairs (ratio 0.22–0.27, against 0.11 for
a trusted body joint). Looked promising.

*But coherence cannot distinguish an image-measured finger from a
wrist-conditioned prior.* The model is per-frame, and its input — the crop, the
arm pose inside it — is already temporally smooth, so a deterministic prior would
produce exactly this. The discriminating test is cross-view geometry: one real 3D
hand must agree epipolarly across four cameras; four independent priors will not.

**3. Cross-view reprojection residual.** Passed, apparently:

| | camera support | median residual |
|---|---:|---:|
| body joints (elbows, knees) | 3.29 | 3.52 px |
| finger joints | 3.31 | **4.23 px** |

Same camera support, 1.2× the residual. Against a gate of "≤2× body", this looks
like a clear pass.

**4. Bone-length stability — the physical invariant.** It is not a pass.

| segment | mean | sd | sd % |
|---|---:|---:|---:|
| forearm L / R | 345 / 330 mm | 295 / 218 | 85% / 66% |
| shin L / R | 433 / 427 mm | 120 / 111 | 28% / 26% |
| L index p1 / p2 | 134 / 68 mm | 320 / 106 | **240% / 156%** |
| R index p1 / p2 | 124 / 150 mm | 319 / 405 | **257% / 269%** |
| R middle p1 / p2 | 196 / 84 mm | 469 / 205 | **239% / 245%** |
| **body mean** | | | **51%** |
| **finger mean** | | | **226%** |

**Ratio 4.4×**, against a gate of 2×. And the absolute values are anatomically
impossible: a proximal phalanx is 25–45 mm, and these triangulate to 68–196 mm
with standard deviations *exceeding their means*. Those are not measurements of a
finger; they are scattered 3D points that happen to reproject acceptably.

## Why reprojection passed and geometry failed

Because a finger is small and the cameras are 4.7 m away, the rays from different
views are nearly parallel *relative to the feature size*. Depth is almost
unconstrained, so the triangulator can place a point that reprojects within a few
pixels of all four observations while sitting decimetres off in depth.

**This is the third time in this lane that reprojection error has flattered a bad
result** — after Battle 0's resolution survivorship and increment 4's clipped
boxes. The pattern is now reliable enough to state as a rule: *when reprojection
and a physical invariant disagree, the invariant is right.*

Caveat on the harness: this test used raw per-frame triangulation with a 40 px
gate and none of the pipeline's temporal or acceptance gating, so the body
control is also poor here (51%, against 3.5% in production). Absolute numbers are
not comparable to production. The **ratio** is the comparison that was asked for,
and the anatomical implausibility is independently damning.

## The physics behind it

A hand in this footage is **~33 px at the detector width and ~20 px inside the
model's 192-wide input** — about **5 heatmap cells for 15 finger joints**, where
one cell is roughly the length of a phalanx at the subject. The published
literature agrees this is out of envelope: RTMW's hand AP falls 66.4 → 61.0 when
input resolution drops while body AP barely moves; COCO-WholeBody's single-network
hand AP is 0.300 against 0.401 for a re-cropped model; InterHand2.6M used 80–140
cameras; and MAMMA — with twice our usable views, 2056×1504, and dense landmarks
deliberately weighted toward hands — still says *"the accuracy of hand-motion
recovery is still lower than we would like."*

Native-resolution re-cropping does not rescue it either: it turns a ~20 px hand
into ~50 px, where the dedicated hand models expect 150–250 px.
**Pixels-on-hand is a rig decision, not a model decision.**

## What MAMMA does differently — the correction

MAMMA ran on **this same fixture**: the same four cameras, the same 4.7 m stage,
the same 150 frames. Its output is retained at
`artifacts/mamma/mamma-4cam-five-second-v2/`. Measured directly from it:

| | |
|---|---:|
| SMPL-X hand pose variation per DoF | 0.098 rad, against 0.186 for the body |
| — i.e. hand articulation is **53% of body articulation** | not a frozen prior |
| pairwise distance stability, MAMMA's 512 triangulated landmarks, extremity region | **7.3%** |
| same, core region | 10.5% |
| **our independently-triangulated SOMA fingers** | **226%** |
| our body control, same harness | 51% |

MAMMA's 3D points are **5–20× more geometrically stable than ours on identical
footage**. The information is in these videos. Our estimator was throwing it away.

**Why.** MAMMA never triangulates a finger joint as a free 3D point. It fits
SMPL-X — a parametric body whose hand is a constrained 45-parameter pose space —
jointly across all views and all frames, with shape shared over the sequence,
temporal smoothing, and per-landmark uncertainty and visibility weighting. A
finger position is therefore never 3 unconstrained degrees of freedom fitted to
near-parallel rays; it is the consequence of a kinematic chain with fixed bone
lengths, driven by every view at once.

That is precisely the estimator my test lacked. Independent per-joint DLT is the
*worst possible* estimator for a small feature at 4.7 m, because depth is almost
unconstrained — which is exactly what the 68–196 mm phalanges show. The failure I
measured is real; the conclusion I drew from it was not.

**So the honest statement is:** fingers at this framing require a
**model-constrained fit**, not triangulation. We already have both halves —
SOMA-77 supplies per-view finger 2D, MHR supplies an articulated hand with fixed
bone lengths — and `solve_sequence_positions` already implements a sequence-level
fit with limb-length and temporal terms. Extending that solve to the hand chain
is the same architecture MAMMA uses, in miniature.

The physics section above still stands as a caution on *accuracy*: ~20 px of hand
inside the model input is genuinely tight, and MAMMA itself reports hand error
well above its body error. Expect pose-plausible fingers, not contact-precise
ones. But "expect limited accuracy" is a different claim from "the information
isn't there", and only the first is supported.

## Decision (superseded — see the amendment above)

~~**Do not wire SOMA-77's fingers into the body track.**~~ Do not wire them
through **independent triangulation**, which is what was tested. The pipeline's own rule
exists for this case: *"missing or untrusted hand observations remain explicitly
`review_required`; they do not fall back to an unreported canned gesture."*
Prior-driven fingers would break that rule with our own detector, and for a
product whose value is fidelity, plausible-but-wrong fingers are worse than an
honest rest pose with a review flag.

Note also that the premise which motivated this was stale: N5.1's status is
*"detailed-hand milestone verified"*. The fingers-stay-in-rest-pose defect is
already fixed; what remains is that the hand lane is preview-qualified.

**What would reopen it:** a rig change that puts 150+ px on hands — a closer or
longer-lens hand camera, or tighter framing. At that point the commercially clean
re-crop lane is ready and waiting: MediaPipe Hand Landmarker is Apache-2.0 on
Google-owned data, and Meta's SAM 3D Body ships a hand-decoder refinement under a
commercial licence, predicting into the MHR rig this plan already targets.
