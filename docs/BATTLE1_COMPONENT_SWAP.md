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

## Rung 1 — the temporal smoother, isolated on real footage

Gate G10 in the fixture doc measured the smoother imposing **4.4 mm of median and
34.9 mm of p95 lag** with perfect 2D on synthetic motion played back at 6x, and I
recorded that as estimator error no detector could touch. The substitution harness
tests it on real data: MAMMA's 2D through our triangulator gives per-landmark tracks
on the one genuinely fast clip, MAMMA's own landmarks are the reference, and the
smoothing is applied per track — 512 landmarks need no joint contract.

Measured speed p50 0.59 m/s, **p95 1.93 m/s**, so the motion is genuinely fast.

| landmark speed | n | raw | smoothed | change |
|---|---:|---:|---:|---:|
| 0.00–0.23 m/s | 30,720 | 11.2 mm | 10.8 mm | −0.4 |
| 0.23–0.47 | 30,720 | 10.4 | 9.9 | −0.6 |
| 0.47–0.71 | 30,720 | 9.0 | 8.7 | −0.3 |
| 0.71–1.11 | 30,720 | 9.8 | 9.2 | −0.6 |
| 1.11–1.93 | 23,040 | 10.6 | 10.0 | −0.6 |
| **1.93–5.29** | 7,680 | 11.8 | **12.5** | **+0.7** |
| **all** | 153,600 | **10.2 mm** | **9.8 mm** | **−0.4** |
| p95 | | 39.8 mm | **34.0 mm** | −5.8 |

**The smoother helps.** Median 10.2 → 9.8 mm and p95 39.8 → 34.0, and it helps in
every speed band except the fastest 5%, where it costs 0.7 mm.

### So §7g of the fixture doc overstates, and here is why

G10 measured the lag with **noiseless** 2D, where a smoother can only hurt — there
is no noise for it to remove, so the measurement captured one side of a
bias–variance trade and reported it as a cost. On real data the 2D is noisy and the
variance reduction outweighs the lag at every speed that occurs.

The lag is real; it is just small relative to what the smoothing buys.
"4.4 mm of median error the detector cannot touch" should read: **at 1.93 m/s and
above the smoother costs 0.7 mm, and below that it pays for itself.** The
velocity-adaptive window is still the right refinement — it would recover that
0.7 mm in the top 5% — but it is a 0.7 mm refinement, not a 4 mm defect.

This is the substitution method paying for itself twice: it isolated the smoother,
and in doing so it corrected a number produced by an instrument that could not
isolate it.

## Rung 2 — the resolution confound, and it is not a lever

The swap has an uncontrolled variable inside it: **our detector consumed 1280-wide
frames and MAMMA's ran at 3840 native.** Nine times the pixels. And the architectural
argument said it should matter — SOMA-77 is **top-down**, cropping the person and
resampling into a 256×192 input, so unlike Battle 0's whole-image Apple Vision it
could plausibly be starved at 1280. This was ranked the highest-value next move
precisely because it might close part of the gap for free.

One variable. Same source video, same ffmpeg quality, same person boxes in
normalised coordinates, same model. Reference is MAMMA's fitted joints projected
into A001 — not SOMA's joint definitions, so the bias term is meaningless, but it is
*identical between the arms*, so **spread** is a fair comparison and spread is what
the ablation is about. Reported in native-3840 pixels so the arms are commensurable.

| | mean bias | mean spread |
|---|---:|---:|
| 1280 wide | 12.2 px | **14.5 px** |
| 3840 wide | 11.3 px | **14.8 px** |

**Nine times the pixels buys nothing** — the spread is 0.2 px *worse*, which is
noise. Joint by joint it is a wash: seven improve, seven degrade, none by more than
2.6 px.

### The mechanism, so the negative is conclusive rather than mysterious

| | person crop side |
|---|---:|
| at 1280 | **286 px** median (166–343) |
| at 3840 | 857 px median |
| the model's input | **256 px** |

**At 1280 the crop is already 1.1× the model's input — it is being *downsampled*,
not upsampled.** There was never any lost detail for 3840 to restore. The
architectural argument was right about the mechanism and wrong about which side of
the threshold we sit on.

**The generalisation, which is worth more than the result:** for a top-down crop
model, input resolution stops mattering once the person crop exceeds the model
input, and our 1280 lane is already past that threshold at this framing. It would
still matter at wider framing, at greater subject distance, or for **hands**, where
the crop is ~20 px and nowhere near 256 — which is consistent with everything the
finger work found.

So Battle 0's conclusion — resolution is not a lever — survives its extension to a
detector with a completely different architecture, for a reason Battle 0 did not
know.

## Rung 3 — SAM 3D Body, assessed

Both advisors named it as the strongest untested asset, and both said verify rather
than trust the option-space survey's one-liner. Verified, 2026-08-29.

**Licence: usable, and more so than the survey recorded.** Meta's custom SAM
License, last updated 19 November 2025. Commercial use permitted; **no**
non-commercial clause, **no** monthly-active-user cap, **no** field-of-use
restriction. Obligations are attribution in publications, redistribution only under
the same terms with a copy attached, no reverse engineering, trade-control
compliance, and termination on breach. **And there is no clause restricting the use
of outputs to train other models** — which is the clause that matters for Battle 3,
and its absence is worth more than the rest.

**It cannot replace our pipeline.** It is single-image human mesh recovery: 72.5 mm
PVE on EMDB for the DINOv3 variant, state of the art for its category and an order
of magnitude worse than the ~10 mm our multiview geometry already achieves against
MAMMA's. A monocular model has depth ambiguity that four calibrated cameras do not.
Anyone reading "state-of-the-art 3D human mesh recovery" as a replacement for a
13.5 mm multiview system has compared two different problems.

**Where it could help is precisely the term we measured as dominant.** Two reasons:

* **It predicts into MHR** — the same Apache-2.0 body model this plan already chose.
  So its landmark definitions *are* our body model's, which directly attacks the
  largest systematic term we have measured: SOMA-77's convention offsets against the
  reference, 39.9 px at the neck, 27.8 at the ankles, 20–25 at the hips, and no ear
  landmarks at all.
* **It is a full-body model with a learned shape and pose prior**, so its predictions
  for occluded landmarks should be *amodal*. That is the property this session
  identified as unpriced and as the likeliest home for much of the 25–34 mm gap:
  MAMMA's occluded predictions are good enough that all 512 landmarks triangulate to
  ~10 mm with no visibility gating, where SOMA-77's occluded predictions are the
  confidently-wrong ones the entire cross-view veto exists to filter.

**And we now have the instrument to test it in one variable.** Project its MHR mesh
joints into each view, drop them into the same substitution slot the swap used, score
against the same reference. If its 2D lands closer than SOMA-77's 18.6 px bias +
16.6 px spread, it is the cheapest available move on the dominant term — no training,
no corpus, no GPU campaign.

**Status: one download away.** The dependency audit records the code worktree as
incomplete; that is stale — `.cache/autoanim_gnm/gem-x/third_party/sam-3d-body` is
present at 23 MB with `LICENSE`, `INSTALL.md` and `tools/`. Only the checkpoint is
missing, at `facebook/sam-3d-body-dinov3`, behind licence acceptance on HuggingFace.
That acceptance is a user action, and it is the gate.

**Caveats before it is treated as the answer.** Its accuracy is reported on
single-image benchmarks in the wild, not at 4.7 m on a four-camera stage — the
framing this project actually has, where the subject is ~286 px tall. Its per-view
outputs would need fusing, which is a design question our triangulation does not
answer. And the licence, while permissive, is bespoke: the plan's standing rule is a
licence-verification pass before download, and this document is that pass for the
terms, not a substitute for the legal review the dependency audit already schedules.

### What the ComfyUI `utility_sam3d_body` workflow shows

Read from the running graph, 2026-08-29. It is a complete **monocular video →
animated MHR** pipeline, and it answers several questions the licence page could
not.

| stage | node | model |
|---|---|---|
| person boxes | Run Real-Time Detection (RT-DETR), class `person`, max 100 | `rt_detr_v4-o-hgnet_fp32` |
| multi-person tracking | Run SAM3 Video Track, threshold 0.50, max_objects 4 | `sam3_1_multiplex_fp16` |
| field of view | Run MoGe Inference → Get FoV from MoGe Geometry | `moge_2_vitl_normal_fp16` |
| **body** | **Run SAM3D Body Prediction** — takes image, `track_data`, `bboxes`, `fov`, and a **`run_hand_refinement`** toggle | **`sam_3d_body_dinov3_bf16`** |
| face | Face Expression to SAM3D Body — *"sam3d-body does not detect face expressions, this node adds them through MediaPipe"* | — |
| smoothing | Smooth SAM3D Body Pose Data, method `gaussian` | — |
| export | Create 3D Animation File — `glb`, `include_hands` on, `bone_smooth_window`, fps 24 | — |

**Four things follow that matter to us.**

**It emits `mhr_pose_data`.** Not landmarks — **MHR pose parameters**, our body
model, straight out of the node. The joint-convention offsets that are our largest
systematic term (neck 39.9 px, ankles 27.8, hips 20–25, no ears) do not arise,
because there is no convention to cross.

**It has a dedicated hand refinement stage**, exposed as a toggle. Battle 1's
increments 5 and 6 built a 27-DoF constrained hand chain by hand and finished at
33.9 mm held-out. This is a learned alternative into the same rig.

**Its weakest link is our strongest.** The whole MoGe stage exists to *guess the
camera* from a single image. We have a bundle-adjusted four-camera rig with
byte-verified intrinsics. That stage is not merely unnecessary for us — the
uncertainty it is compensating for is a large part of why monocular HMR sits at
72.5 mm PVE.

**And that reframes how it should be used.** The workflow is monocular and
per-frame. We would run it **per view and fuse four MHR estimates with known
extrinsics** — a parametric estimate per camera rather than 2D points, which is
strictly more to fuse than the landmark triangulation we do now. That is a
different and much better-conditioned problem than the one the workflow solves,
and it aims directly at the term this document measured as dominant.

The model files are named, so a local reproduction needs no reverse engineering of
the graph: `sam_3d_body_dinov3_bf16`, `sam3_1_multiplex_fp16`,
`moge_2_vitl_normal_fp16`, `rt_detr_v4-o-hgnet_fp32`. Note these are **Comfy-Org
mirrors**; the SAM License governs the SAM materials wherever they are hosted, and
SAM 3 and MoGe carry their own terms that this pass has **not** checked.

### For our rig, only one of the four models is needed

The workflow loads four. Three of them exist to compensate for things we already
have, so the ask is **one asset, not four**:

| workflow model | what it is for | do we need it |
|---|---|---|
| `moge_2_vitl_normal` | guess the camera's field of view from one image | **no** — bundle-adjusted rig, intrinsics byte-verified against MAMMA's |
| `sam3_1_multiplex` | track multiple people across frames | **no** — cycle-consistent cross-view association, zero identity switches on this take |
| `rt_detr_v4-o-hgnet` | person bounding boxes | **probably not** — we already source boxes, and the Apple Vision dependency is separately audited |
| **`sam-3d-body-dinov3`** | **the body model itself, predicting into MHR** | **yes — this is the whole ask** |

**It is promptable, and that changes the integration.** From the model's own README:
*"3DB employs an encoder-decoder architecture and supports auxiliary prompts,
including 2D keypoints and masks, enabling user-guided inference similar to the SAM
family of models."* So it need not be run blind per view — it can be **prompted with
the 2D we already have**, which means SOMA-77's detections stop being the final word
and become a hint to a model with a full-body prior. It also opens a multi-view
option the workflow does not use: prompt all four views toward a consistent pose.

**Take `facebook/sam-3d-body-dinov3`**, not `sam-3d-body-vith`: it is the variant the
workflow loads, the one carrying the published EMDB figure, and by download count
the one in use (8.04k against 787). The model card confirms what matters — *"body,
feet, and hands using the Momentum Human Rig parametric mesh representation"*.

**Not** `facebook/sam-3d-body-dataset` (5.66M rows). That is training data. It is
interesting for Battle 3 and it is a separate decision with separate terms; it is
not needed to evaluate anything.

**Two flags before download.** Access requires accepting terms and sharing contact
information — a user action, and the accepted terms should be archived per the
dependency audit's standing requirement. And the model card declares the SAM License
but is **silent on whether DINOv3's own licence applies in addition**; the backbone
is Meta's own, so the SAM License plausibly governs the whole bundle, but "plausibly"
is exactly what the asset gate exists to replace.

## Rung 4 — MAMMA's empirical 2D residual, and the fork it was meant to settle

Queued to settle the mixture-versus-lognormal fork the sigma/visibility ordering
hung on. Run, and **the value came from somewhere else** — the fork had already
stopped mattering, because the swap priced visibility on real data at 0.16 mm
directly, without needing a noise model at all.

MAMMA's fitted landmarks reprojected through the rig, minus its own 2D. Native
3840 px, 307,200 observations:

| | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all 512 landmarks | 1.87 | 3.30 | 5.84 | 10.18 | 19.33 | 36.23 | 122.51 |
| **visibility ≥ 0.5** | **1.44** | **2.50** | **4.29** | **6.99** | **10.37** | **13.08** | 20.74 |
| *ours, cross-view ladder, scaled to 3840* | 1.59 | 4.11 | 9.33 | 18.10 | 34.50 | 98.40 | — |

**The tail is the story, not the bulk.** In the median we are ~2.2× wide of MAMMA;
at p95 we are **7.5× wide** — 98 px against 13. That is the same shape every other
instrument has pointed at: our detector's failures are catastrophic where MAMMA's
are merely imprecise, which is what "amodal on occluded landmarks" buys.

**Read the bulk comparison with care.** These are different statistics — MAMMA's is
a fit residual, ours a cross-view disagreement — and MAMMA's is *deflated* because
its fit consumed the very landmarks it is scored against. The median ratio is
therefore unfair to us by an unknown amount. The tail ratio is large enough that it
is unlikely to be entirely artefact, but it is not clean either.

### The fork, answered weakly

| shape fitted to the visibility ≥ 0.5 residual | best fit | quality |
|---|---|---:|
| two-point mixture | clean 4.50 px, **2%** bad at 14 px | |log ratio| **0.147** |
| lognormal | median 4.50 px, σ_log 0.30 | 0.186 |

**The mixture wins, but neither fits well** — both are far worse than the 0.043 the
mixture achieved on our own cross-view ladder. That is expected: a fit residual is
shaped by the regulariser as much as by the detector, so it is the wrong quantity to
fit a *detector* noise model to. The fork is nudged, not closed.

What the fit does say clearly is that **MAMMA's residual is nearly
homoscedastic** — 2% bad at 14 px, against the 7% at 36 px our own ladder fits.
Tight and uniform, which is the same finding as the tail comparison from the other
side.

### And a units hypothesis for the log-sigma channel

The channel's median is −4.801, so exp is 0.00822. Against a residual median of
4.29 px:

| if the units are | implied σ |
|---|---:|
| normalised by image width (3840) | 31.58 px — far larger than the residual, wrong |
| **normalised by the model crop (256)** | **2.11 px — within 2× of the residual** |
| raw pixels | 0.008 px — absurd |

**Crop-normalised is the only plausible reading**, and it puts σ at about half the
observed residual, which is the right side to be on given the residual carries fit
error the detector's own estimate does not. Suggestive, not settled — the actual
crop size is unverified.

### The durable output

Not the fork. **A target spec.** `1.44 / 2.50 / 4.29 / 6.99 / 10.37 / 13.08 px at
3840` is what a 13.5 mm-class detector's 2D looks like through its own fit. When a
candidate goes into the substitution slot — SAM 3D Body first — *"does its residual
distribution look like this, particularly in the tail"* is the acceptance question,
and it is now written down before the candidate arrives rather than after.
