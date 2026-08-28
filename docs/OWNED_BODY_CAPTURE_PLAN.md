# Owned body capture — execution plan

Status: plan, 2026-08-26. Companion to `docs/OWNED_BODY_CAPTURE_RESEARCH.md`,
which holds the findings and the sources. This doc holds only the sequence, the
gates, the owners and the costs. It will churn; the research doc should not.

## The target, confirmed

MAMMA-parity at four cameras, or better:

| Cameras | Single-person MPJPE | Two-person MPJPE |
|---|---|---|
| 2 | ~24 mm | ~47.5 mm |
| **4 — our rig** | **~13.5 mm** | **~20 mm** |
| 8 | ~13.5 mm | ~17.5 mm |
| 12–16 | ~13 mm | ~17.3 mm |

Plus the end-state gate: **held-out marker error within ~2 mm of a marker-based
pipeline**, which is the protocol MAMMA used to claim Vicon parity (22.481 mm vs
21.619 mm).

**State the metric on every claim.** MPJPE, PVE and held-out marker error are
three different numbers. On the same data MAMMA scores 12.96 / 17.18 / 22.48.
Marker-honest runs 5–10 mm behind joint-centre MPJPE. A gate that doesn't name
its metric is not a gate. Full caveat in research §2.

---

## The dependency that shapes everything

**Battle 2 is the critical path for the *final gate*.**

*(Restated 2026-08-28. It was "the critical path for measurement", which is too
strong: reference-free progress on the detector — Battles 3 and 4 — can begin
before we own a capture. What Battle 2 gates is any claim of accuracy.)*

Research §4.1 concluded the MAMMA fixture cannot referee commercial development —
its licence bars production of artifacts for commercial purposes, so benchmarking
a product we intend to sell against it is out of scope of the grant. Until we own
a reference capture:

- **No accuracy claim in this plan can be evaluated.** Every mm figure above is
  unmeasurable for us today.
- Battle 1's exit gate must therefore be **reference-free**: reprojection error
  in mm, temporal jitter, bone-length consistency, joint coverage, identity
  switches. Not "distance to MAMMA".

So Battle 2 runs **in parallel with Battle 1**, not after it. And the MAMMA
fixture gets a retirement date: **it is deleted from the evaluation path the day
Battle 2 delivers**, and from that point exists only as a historical note.

---

## The on-device constraint is gone

The pipeline may run on a bigger GPU; it does not have to run on the MacBook.
A 72-agent survey established what that unlocks — see
`docs/GPU_DETECTOR_OPTION_SPACE.md`. The short version:

- **No commercially-clean whole-body 2D checkpoint exists.** The menu was
  licence-gated, not compute-gated, so this unlocks zero new shippable models.
  COCO-WholeBody is research-only and its 133-keypoint skeleton *is* the
  whole-body format, so every open checkpoint inherits the taint.
- **What it does unlock is density.** Dense landmarks cost ~80× sparse-keypoint
  inference — exactly what an on-device budget could never afford, and exactly
  where MAMMA's accuracy comes from.
- **Cost is not the obstacle.** Our shot is $0.10–$0.49 of GPU time; a full
  ViT-Base training run is ~$600–720. The unpriced item is the synthetic corpus.
- **Three clean assets found:** Meta's SAM 3D Body (commercial checkpoints,
  predicts into MHR), SAM 2 (Apache-2.0 code *and* weights), Unity
  PeopleSansPeople (Apache-2.0 synthetic humans).

## Sequence

### Battle 0 — the detector-width experiment ✅ DONE 2026-08-27

**Result: `docs/BATTLE0_DETECTOR_WIDTH_FINDINGS.md`. The hypothesis was
wrong, two of my own analyses were wrong, and the actionable win turned out to
be something else entirely.**

- Naive reading said 3840 was 2.7× better. **Survivorship bias**: the hard-coded
  `inlier_threshold_px = 14.0` is a *shrinking physical* gate as resolution
  rises (78 mm at 1280, 26 mm at 3840) and discarded 41.8% of observations.
  Stratified by inlier count the residual actually *rises* with width.
- Controlled (all detections renormalised to a common frame), **precision is
  flat**: 25.8 → 25.8 → 25.5 mm.
- A direct sweep at 320–3840 px shows **Apple Vision saturates below 1280** —
  3.9% jitter change for 8× the pixels. At most ~6 mm was ever available.
- **The real win is JPEG quality, not resolution.** `_extract_frames` passes no
  `-q:v`, so bigger frames got *more* compression. Pinning `-q:v 2` at 1280
  gives valid joints 82.8% → **88.2%**, temporal rejections 33 → **14**, bone sd
  38.0 → **33.7 mm** — beating the 1920 lane at 1/2.25 the pixels.

**What it means for the plan:** Apple Vision carries ~26 mm of 2D error at the
subject *at any resolution*. That triangulates to ~25 mm median / ~59 mm P95 —
a large share of the 60–80 mm gap. It is model-limited, not pixel-limited, so no
rig or resolution change reaches past it. **The detector is the programme**, and
Battle 4 is where the target is won.

**Applied 2026-08-27** — `-q:v 2` pinned at 1280; pixel gates now scale with
detector width against `REFERENCE_DETECTOR_WIDTH_PX`; `detector_width` and
`frame_jpeg_quality` recorded in `run-report.json`; frame cache invalidated by an
`extraction.json` stamp; pre-interpolation triangulation persisted as
`raw_triangulated_world_positions_z_up_m`. Threshold change proven a no-op at
1280 (bit-identical baseline); fresh run lifts valid joints 82.8% → **88.2%** and
cuts temporal rejections 33 → **14**; verifier passes; 33 tests green, with the
four scaling terms mutation-tested.

**Closed 2026-08-28:** the per-person ROI crop is **not** a Battle 1 experiment.
Apple Vision is model-limited, so a crop cannot rescue it. Crop resolution is a
real lever — published ablations put it above a backbone step — but on *our own*
detector, in Battle 4.

---

### Battle 1 — sparse-pipeline upgrades (weeks, $0, no licensing exposure)

Every component Apache/BSD, no training, nothing to buy. See research §10.

- SAM2 masks with temporal identity propagation (Apache-2.0)
- RTMDet for person boxes — **not Ultralytics YOLO** (AGPL)
- ✅ **Interim detector adopted 2026-08-28: NVIDIA GEM-X's SOMA-77**, a ViT-Huge
  whole-body model already in `.cache` under the NVIDIA Open Model License,
  trained exclusively on NVIDIA-owned synthetic data. 2D error at the subject
  25.6 → **16.2 mm**, bone instability 9.4% → **3.5%**, temporal rejections
  14 → **0**, and 99.3% direct triangulation over the 17 joints it emits.
  Integrated behind `--detector soma77` and verified end to end: the acceptance
  gate passes, retarget endpoint median 157.0 → **120.0 mm** and p95 359.7 →
  **249.0 mm**. Bone instability 9.4% → **3.5%** is a large gain and still does
  not meet the 2% gate, which stays with Battle 4. See
  `docs/BATTLE1_INCREMENT4_SOMA77_DETECTOR.md`
- ~~MediaPipe Pose Landmarker as the interim detector~~ — **closed negative
  2026-08-27.** Measured head-to-head on our own fixture it is far worse than
  Apple Vision (32.6% vs 88.2% joint coverage). Apple Vision stays. See
  `docs/BATTLE1_INCREMENT3_DETECTOR_COMPARISON.md`. The viability blocker was
  also wrong — the recorded abort is the *legacy* graph, not the Tasks runtime.
- ✅ **Cycle-consistent Hungarian matching on epipolar affinity, replacing the
  exhaustive per-frame permutation search** — landed 2026-08-27, identical
  output on every fixture frame, and the only tractable path beyond two
  subjects *on phantom-free footage* — a surplus detection still routes a frame
  into the exhaustive search, which is now bounded rather than unbounded. See
  `docs/BATTLE1_INCREMENT1_ASSOCIATION.md`
- ✅ **AniPose-style spatiotemporal triangulation with per-shot bone lengths** —
  landed 2026-08-27. Recovers single-ray joints from evidence instead of
  interpolating them: coverage 0.8823 → **0.9398**, interpolation halved, every
  pre-existing metric unchanged. See `docs/BATTLE1_INCREMENT2_SEQUENCE_SOLVE.md`

**Ceiling, restated 2026-08-28.** The original text said Battle 1 lands ~40 mm
because the residual is systematic label-convention bias. The bias claim was
right and the ceiling claim was too pessimistic: swapping to a detector that
predicts skeletal joint *centres* rather than surface landmarks cut 2D error 39%
and bone instability 41%. Battle 1 still cannot reach 20 mm — dense landmarks are
Battle 4's job — but it got materially closer than "de-risk the plumbing".

*Exit (reference-free):* zero identity switches on the two-person fixture ✅;
triangulation jitter ≤ 5 mm median; **direct + constraint-recovered joint
coverage ≥ 90%** ✅ (0.9398).

**Gate restated 2026-08-27.** Two changes, both from measurement:

1. Coverage was "direct ≥ 90%". A joint resolved from one ray plus limb and
   temporal constraints is *evidence-based*, unlike an interpolated one, so it
   counts — but under its own metric, `constraint_recovered_joint_fraction`, not
   by redefining `valid_joint_fraction`, which is unchanged at 0.8823.
2. **"Bone lengths stable within 2%" moves to Battle 4.** Measured on frames
   where both endpoints triangulated directly, limb length varies by a median of
   **10.4%**, while per-frame scatter of those same joints is only 3–7 mm. The
   instability is the detector's joint definition moving with pose, not the
   geometry. No solver reaches 2% on Apple Vision. This is the same
   model-limited conclusion Battle 0 reached, measured a second, independent
   way.

---

### Battle 2 — own the reference, and build the rig properly (one shoot)

Unblocks every accuracy claim in the plan. Research §9b sets the spec, and it is
not optional — the physical terms are what make 20 mm reachable at four cameras.

- **Genlock, plus a sync check in every take.** MAMMA's genlocked rig still
  slipped two frames at 60 fps. Budget ≤5 ms residual.
- 60 fps, fast-readout sensors, 1/500–1/1000 s shutter, enough light to support it
- Board + wand bundle-adjusted calibration per shoot
- Cameras surrounding the volume, not in an arc
- Fitted wardrobe; a body scan of each performer
- **Simultaneous marker reference** — rented suit or a hired session
- Consented performers, with releases covering ML training use

Shoot list should cover: dialogue two-hander, hands near face, seated acting,
crossing identities, occlusion, one extreme-range-of-motion block (the MOYO gap —
a yoga or movement performer, one day, owned outright).

*Exit:* sync verified ≤5 ms per take; calibration validated by **triangulating
known lengths**, not by reprojection RMS (wand-only calibration overfits its own
recording); marker reference recorded and time-aligned; the whole set usable
commercially with documented provenance.

---

### Battle 3 — the data factory (the real programme)

Motion data turned out to be free (research §9a). The cost here is rendering and
engineering, not acquisition.

- **Bodies:** MHR (Apache-2.0, weights included)
- **Motions:** CMU + Eyes JAPAN + ACCAD + 100STYLE ≈ 7,300 clips, retargeted onto
  MHR with ordinary skeletal retargeting. **Never ship the motion files** — CMU
  bars reselling "even in converted form"; renders are inside the grant
- **Hands:** ContactPose (MIT) grasps + Feix-taxonomy poses, spliced per-frame —
  no hand mocap needed
- **Occlusion:** co-locate independent single-person clips in one volume, as
  MammaSyn did with "2–6 subjects at random positions"
- **Environments:** Poly Haven CC0 HDRIs
- **Renderer:** Blender/Cycles. **Not Unreal/MetaHuman** — Epic's position bars
  training models on MetaHuman content
- **Clothing and hair:** a purchase decision — Daz AI-training program vs
  Reallusion Enterprise. Both negotiated, neither public

Render at high resolution. MAMMA used 2056×1504 explicitly for hand detail.

*Milestones:* ~150k crops for a first single-person model (Microsoft's proven
scale); ~1M crops ≈ 3k sequences × 8 views for multi-person parity.

*Exit:* a versioned corpus with a per-asset licence manifest, and a held-out
synthetic validation split.

---

### Battle 4 — the detector (the programme)

Reimplemented **from the published papers, by someone who has not read MAMMA's
source.** Its licence bars reverse engineering, and clean-room hygiene points the
same way.

- ViT-B backbone + CNN mask encoder, features summed
- N learnable per-landmark queries through a transformer decoder (512 landmarks,
  farthest-point-sampled from the MHR surface, weighted to hands/feet/head)
- Outputs: μ, uncertainty σ, visibility p. Gaussian NLL on landmarks, BCE on
  visibility
- Then: robust reprojection fit to MHR weighted by σ and p, shared per-subject
  shape, temporal term, L-BFGS

**Skip floor-contact optimisation** — MAMMA's own evidence says it does nothing
in the multiview case. Person-contact terms only if penetration becomes visible.

*Exit:* **two-person ≤20 mm MPJPE at 4 cameras against our own marker
reference**; single-person ≤13.5 mm. Then the end-state gate: held-out marker
error within ~2 mm of the marker pipeline.

---

### Battle 5 — hands

Per-hand high-resolution re-crop pass seeded by wrist/palm predictions, then
triangulation and IK onto our own hand rig with per-subject finger bone-length
calibration.

*Exit:* ~10° finger joint angles at dialogue framing. This closes the N5.1
blocker at acting quality, not at contact quality — set that expectation with
reviewers up front.

---

## Standing rules — how sol and fable stay involved

Not a vibe; specific checkpoints.

**fable (research agents):**
1. **Asset gate.** Every newly introduced asset — dataset, checkpoint, texture
   pack, motion library, renderer — gets a fable licence-verification pass
   *before* download or purchase, with a URL per claim. This is the gate that
   caught Ultralytics (AGPL), RTMW's tainted checkpoints, Mixamo and Reallusion.
   No exceptions, including for things that "obviously" look fine.
2. **Battle-opening deep dive.** Battle 3: MHR-in-Blender pipeline specifics and
   CLO3D output-ownership terms. Battle 4: implementation references for the
   Gaussian NLL head and per-landmark query decoder. Battle 5: hand re-crop
   architectures.

**sol (advisor review):**
3. Before each battle's design commits.
4. Before each battle is declared done.

**Existence proof before a negative result.** Before recording "X cannot be
done" as physics, check whether anything demonstrably does X under the same
conditions. This lane recorded "fingers cannot be triangulated from four wide
cameras" while MAMMA's finger output on those exact four videos sat on disk
unexamined. The measurement was right; the generalisation was not, and only the
user's domain knowledge caught it. The review machinery tests claims that were
made — it does not generate the counter-example nobody thought to look for.

**Isolation.** Review workflows run read-only or with worktree isolation. A
review agent once left a mutation-test stub in tracked source and did not
restore it; it was caught before it reached a commit, but it should have been
structurally impossible. Any mutation of tracked files by a review agent is a
defect in the review.

---

## Actions that need you, not me

| Action | Why | When |
|---|---|---|
| **FTO counsel engagement** | US10395411B2 (MPG, ~2036) claims runtime use of the blendshape+skinning formulation; US9189886B2 (Brown, ~2032) reads on image-based capture regardless of body model. Apache-2.0 from Meta cannot clear Max Planck's patents | Start now, in parallel. Blocks launch, not engineering |
| **Request EgoSuite-Open100K access** (`huggingface.co/datasets/LightwheelAI/EgoDemo`) | Best hand-articulation motion source evaluated — 21-joint non-MANO hand pose at scale, retargetable to MHR. Licence text is not public; manual review, 2–3 business days. Read it against four clauses: do derivatives (renders, trained weights) fall under "no resale"; are training outputs unrestricted; termination; consent warranty | Before Battle 3 |
| Email KIT (`motiondatabase@lists.kit.edu`) | 4,600 motions with **no published licence at all** — cheapest large win available | Before Battle 3 |
| Email HDM05 (`HDM05@mpi-inf.mpg.de`) | Confirm the archived CC BY-SA 3.0; live host 403s | Before Battle 3 |
| Email Truebones | Confirm ML training is permitted before relying on the $99 pack | Optional |
| Book suit rental or marker session | Battle 2 | Now — it is the critical path |
| Clothing/hair purchase decision | Daz AI program vs Reallusion Enterprise, both negotiated | Before Battle 3 |
| Performer consent and releases | Must cover ML training use explicitly | Before Battle 2 |

---

## Cost shape — estimates, not quotes

| Battle | Cost | Basis |
|---|---|---|
| 0 | $0, same day | Rerun an existing fixture |
| 1 | $0, weeks of engineering | All components Apache/BSD |
| 2 | Suit ~$300–1,000/week + ~$315/month software; or a hired marker session | Published rental rates |
| 3 | **Motion data: $0.** Render at MAMMA scale (955k frames) is a **$10k-class** Modal bill; the first 150k-crop model is a fraction of that. Clothing/hair assets: negotiated | Per the compute-split memory, all GPU runs on Modal |
| 4 | ~4× A100 × 3 days per training run | MAMMA's published training cost |
| 5 | Engineering only | Reuses Battle 3/4 infrastructure |

Every figure is an estimate from a published or quoted source, not a quote for
our configuration.

---

## Immediate next action

**Battle 0.** Rerun the existing fixture at `--detector-width` 1920 and 3840,
report reprojection error and jitter **in mm**. Same day, $0, and it either
hands us a 3× improvement or tells us early that the detector is the binding
constraint — which is the thesis of this whole plan.

Say the word and I'll run it.
