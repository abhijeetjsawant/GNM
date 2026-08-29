# Where the body lane stands — 2026-08-29

One day, sixty-six commits, three strategic reversals and six
correct-measurement-wrong-claim instances. This document exists because the findings
are spread across nine documents written in fifteen-minute intervals, and the next
session — or the decision about whether to fund Battle 3 and 4 — needs one place that
says what is known, by what instrument, with what caveat.

**Every claim below carries the instrument that produced it.** Where a number was
retracted it is marked, because the retractions are the most useful part.

---

## 1. The one number that survived every correction

**Our detector's error tail is 7.5× MAMMA's.**

| | p50 | p95 |
|---|---:|---:|
| MAMMA's 2D residual, visibility ≥ 0.5, native 3840 | 4.29 px | **13.08 px** |
| ours, cross-view ladder scaled to 3840 | 9.33 px | **98.40 px** |

*Instrument:* MAMMA's fitted landmarks reprojected minus its own 2D, against our
cross-view epipolar disagreement. *Caveat:* different statistics, and MAMMA's is
deflated because its fit consumed the landmarks it is scored against — so the median
ratio is unfair to us by an unknown amount. **The tail ratio is too large to be
entirely artefact**, and it agrees with everything else measured today.

**Our failures are catastrophic where MAMMA's are merely imprecise.** That is the
gap, stated as narrowly as the evidence allows.

---

## 2. The estimator is not the bottleneck — three independent instruments

| instrument | finding |
|---|---|
| **substitution** — MAMMA's 2D into our triangulation | lands **8.9 mm** from MAMMA's own landmarks, **10.2 mm** at full coverage. MAMMA's *own* fit sits 9.3–11.7 mm from its *own* triangulation, so we are as close to their fit as they are |
| **Cramér-Rao** on this rig | single-frame bound is 33.2 mm at four clean views; we measure 16.4 mm. We are at half the memoryless bound, because the temporal and limb priors aggregate across frames |
| **smoother isolated** on real fast motion (p95 1.93 m/s) | it **helps**: 10.2 → 9.8 mm median, 39.8 → 34.0 p95, improving every speed band except the fastest 5% where it costs 0.7 mm |

**Scoped deliberately:** this licenses *"triangulation is not the bottleneck"*, not
*"the estimator has no headroom"*. The swap stops at `triangulate_point` and never
exercises `solve_sequence_positions`. And **fable's saturation finding bounds all of
it: this instrument cannot read below ~10 mm against MAMMA's fit** — smaller deltas
measure reference wobble.

**The smoother result is scoped too:** measured on *MAMMA's* 2D. Whether it helps on
SOMA-77's noisier 2D at speed is untested.

---

## 3. What a better detector is worth — and what was retracted

**Retracted.** The first pricing said μ 10.4 mm, visibility 4.5 mm, σ 0.4 mm, and
this plan was reordered around it. The noise model had been fitted to
`_epipolar_distance_px`, which returns the **symmetric** distance, as though it were
one-sided — an overstatement of exactly 2.00×, measured at 1.962 over 23,987 pairs.

**Corrected**, and weaker:

| | mixture noise | lognormal noise |
|---|---:|---:|
| baseline | 11.62 mm | 14.38 mm |
| a σ head buys | +0.95 | **+2.15** |
| a visibility head buys | **+2.75** | +1.19 |
| a *wrong* channel costs | −3.56 | −3.64 |

**The ordering reverses between two noise models that both fit our data**, so σ and
visibility cannot be ranked. **Battle 4 keeps MammaNet's full μ/σ/visibility triple**
— the reordering is withdrawn.

Measured directly on real data, a visibility channel used as a weight is worth
**+0.16 mm** on identical landmarks. The 1.3 mm first reported was a composition
shift between different landmark populations.

---

## 4. Levers tested and closed

| lever | verdict |
|---|---|
| input resolution 1280 → 3840 | **not a lever.** 14.5 → 14.8 px spread. At 1280 the person crop is already 286 px against the model's 256 px input, so it is downsampled either way. Would still matter for **hands**, at ~20 px |
| a per-joint calibration offset | **body-fixed within the synthetic domain**, 85% removable, frame buildable from positions alone — but it **does not transfer to real footage** (cosine +0.039 against the real bias field), so the synthetic renders cannot measure detector bias |
| SAM 3D Body as a 2D drop-in | **worse than SOMA-77**: 1.83× median, 2.70× p95. Its 2D are reprojections of a monocular fit, so four plausible bodies disagree in 3D and therefore epipolarly |

---

## 5. SAM 3D Body — integrated, licence-clean, one shape refuted

Downloaded, sha256-verified, running at ~45.8 s per person-frame on CPU. **MPS is
impossible** — float64 inside the TorchScript MHR module, out of reach of any
external shim. Worker: `workers/commercial_multiview/sam3d_body_pose.py`.

**Why it remains interesting despite the negative:** it emits `pred_joint_coords`
on **exactly our 127-joint MHR rig** (checked), so no convention gap exists; it emits
`pred_global_rots`, the per-joint **orientations** our reconstruction cannot
estimate; and it takes our calibrated intrinsics, retiring the monocular FoV problem.

**Licences verified against their text**, not their tags: SAM License and DINOv3
License, neither with a non-commercial clause, an MAU cap, a field-of-use
restriction, or any restriction on using outputs to train other models.

**Two shapes remain untested, and both are more expensive than they first looked:**

* **fuse four per-view MHR estimates with known extrinsics** — this means optimising
  *through* the MHR forward model, and the only differentiable MHR here is the
  float64 TorchScript graph that just refused MPS. CPU-only optimisation through a
  696 MB compiled graph, per frame;
* **prompt it with our 2D** — **not available through the public API.**
  `process_one_image` has no keypoint-prompt parameter, and the internal machinery
  samples prompts from the model's *own* predictions. Reaching external keypoints in
  means going below the entry point into internals.

---

## 6. What we still cannot claim, and the standing blocker

**No accuracy claim in this lane is ground-truth-verified.** Every number here is
relative to MAMMA, or to synthetic truth that carries none of calibration error,
lens distortion, sync error, soft-tissue artefact or joint-definition error — all
five first-order on real footage.

**Battle 2, the marker session, is still unbooked**, and it is the only path to an
accuracy claim. Its checklist has existed since this morning:
`BATTLE2_ACTION_CHECKLIST.md`. The three items with real queues are the marker
reference (weeks), genlock (days–weeks) and performer releases covering ML training
use. **These are user actions and none has been taken.**

---

## 7. The method that produced most of today's real findings

The user's: **if you judge part A while B, C and D are also new, you have measured
nothing.** Substitution — hold everything fixed, change one thing — separated in
single measurements what four purpose-built instruments could not. The harness is at
`tools/swap-harness/`.

Two standing rules came out of the failures:

* **no gate a constant can pass.** Four constructs this session could not fail: a
  bone-length gate reading 0.00% because bone lengths are constants; a jitter gate
  reading better-than-MAMMA on a frozen hand; gate G6 comparing against a constant σ,
  which is an algebraic identity; and a `both` arm byte-identical to the arm it was
  meant to be compared against;
* **score both arms on the same denominator.** A composition shift masqueraded as an
  effect twice.

**Six times today a correct measurement carried a claim it did not support.** The
pattern is stable enough to plan around: the measurement is usually right, and the
sentence attached to it is where the error lives.
