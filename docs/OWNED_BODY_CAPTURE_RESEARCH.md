# Building our own MAMMA-class body capture

Status: research finding, 2026-08-26. No code. This doc answers one question —
**can we build a commercially clean markerless body capture system that is as
good as MAMMA?** — and says what each part would cost.

It supersedes nothing. `docs/COMMERCIAL_MULTIVIEW_BODY.md` made the ship/no-ship
decision and sketched phases P0–P5; this doc is the paper-level layer underneath
its P1–P3, and it corrects two license claims made elsewhere in the repo.

Every number below carries the source it came from. Claims marked `UNVERIFIED`
were not confirmed against a primary source in this pass and must not be built on.

---

## 1. The short answer

**Yes, and the hard part is narrower than it looks.**

MAMMA's accuracy does not come from clever fitting maths. Its fitting stage is
deliberately plain: L-BFGS over SMPL-X shape/pose/translation with a robust
reprojection term and L2 regularisers, and the paper states outright that
"optimization does not rely on pose priors or pose initialization". We already
have a comparable optimiser.

The accuracy comes from **the observations**: 512 dense surface landmarks per
person per view, each carrying a predicted uncertainty and a predicted
visibility, produced by a network conditioned on a segmentation mask. On Hi4D,
that one change is the difference between 41.29 mm (4D Association, sparse
keypoints) and 12.44 mm (MAMMA) — same class of triangulation, three times
better input.

So the programme is: **replace 19 sparse joints with a dense, uncertainty-aware
landmark set on a rig we own.** Everything else we either have or can assemble
from permissively licensed parts.

Two things that looked like blockers are not:

- **The body model.** We assumed we'd have to license SMPL-X or build our own.
  Meta released **MHR** in November 2025 under **Apache-2.0, weights included** —
  a full parametric body with articulated hands and learned pose correctives,
  derived from 600k scans. See §8.
- **The supporting stack.** SAM2, MMPose, Pose2Sim, OpenSim, AniPose and the
  association literature are all Apache/BSD. See §5.

The genuinely hard part is the **training data** — MAMMA's synthetic corpus is
built from seven motion sources and every one is non-commercial, with AMASS as
the hidden gate underneath them all. But **AMASS turns out to be routable
around**: it is a derivative compilation, and CMU + Eyes JAPAN + ACCAD + 100STYLE
give ~7,300 commercially-clean clips at the original hosts, free — 2.4–3.6× what
BEDLAM used. Hands too: Microsoft *splices* static hand poses in, and ContactPose
(MIT, 2,306 real grasps) fills that. See §9a. So the remaining cost is rendering
and engineering, not data acquisition.

**Two things stand between us and 20 mm, and neither is the rig geometry.**
Measured on our own footage: the four-camera ring is near-ideal, but Apple
Vision's ~26 mm of 2D error at the subject triangulates to ~25 mm median /
59 mm P95 — a large share of our 60–80 mm gap. Battle 0 then showed that error
is **model-limited, not resolution-limited**: more pixels do not reduce it. So
the detector is the programme. Separately,
synchronisation is a hard physical gate — half a frame at 30 Hz costs 29–48 mm at
our measured body-joint speeds, and MAMMA's own genlocked rig still slipped two
frames. See §9b.

One risk survives everything: **the SMPL patents are live to ~2036, and a second
family reads on image-based body capture itself regardless of which body model
we ship.** Apache-2.0 from Meta cannot clear Max Planck's patents. This needs
counsel. See §8.

---

## 2. What "just as good" has to mean

Setting a number matters, because "as good as MAMMA" is currently unfalsifiable.

| System | Error | Source |
|---|---:|---|
| Vicon + MoSh++ (the pipeline that built AMASS) | 21.619 mm held-out marker error | [MAMMA v4 §5.5](https://arxiv.org/pdf/2506.13040) |
| **MAMMA** | **22.481 mm** (+0.862 mm vs Vicon) | same |
| MAMMA MPJPE, its own evals | 12.96 – 22.95 mm | MAMMA v4 Table 4 |
| MAMMA MPJPE, hard close interaction (Harmony4D) | 45.26 mm | same |
| Theia3D — a shipping commercial markerless product | <25 mm joint centres, 36 mm hip | [systematic review](https://www.theiamarkerless.com/blog/systematic-review-evaluates-the-accuracy-and-reliability-of-theia3d-markerless-motion-capture) |
| Pose2Sim — fully permissive open stack | 3.0–4.1° joint angle error | [Pagnon 2022](https://www.mdpi.com/1424-8220/22/7/2712) |
| OpenPose multiview | 47% of MAE <20 mm, 10% >40 mm | [PMC7739760](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7739760/) |
| **AutoAnim P0 clean-room (today)** | **60.8 / 79.8 mm** median vs MAMMA | `docs/COMMERCIAL_MULTIVIEW_BODY.md` |

Our clean path is roughly **3× off MAMMA and 2× off shipping commercial
markerless**.

**The target is Vicon parity: ~20–22 mm held-out marker error.** That is the
first row of the table — the marker-based pipeline that built AMASS — and MAMMA
sits 0.86 mm behind it. Commercial-markerless parity (~25 mm) is a waypoint, not
the goal.

**But be precise about what that number is, or the target is unfalsifiable.**
20.49 mm is *held-out marker error*: fit the body to most markers, then measure
how well it predicts markers it never saw. It is not distance from anatomical
truth. Marker-based reference itself carries soft-tissue artifact — skin moves
relative to bone — so "20 mm vs Vicon" is *agreement with a noisy reference*, not
absolute accuracy. Three consequences:

1. The gate must name a protocol: agreement with our own marker reference, on a
   named held-out marker set, on our own footage.
2. We cannot measure it at all until Battle 2 — we have no reference we may
   legally use.
3. Do not mix metrics. Held-out marker error, MPJPE (joint centres), and PVE
   (surface vertices) are three different numbers and MAMMA reports all three.

Note a version discrepancy: the arXiv **v1 HTML** and the **v4 PDF** report
different numbers (v1: Hi4D 13.43 mm, marker gap 1.611 mm, ~8 h runtime; v4:
12.44 mm, 0.862 mm, ~26 h). **Cite v4** — it is the CVPR camera-ready.

---

## 3. MAMMA, decomposed

Stage graph (from the public README, `ma_cap → ma_masks → ma_2d → ma_3d → ma_vis`):

| Stage | What it is | Is it the hard part? |
|---|---|---|
| `ma_masks` | SAM2, initialised once then **propagated temporally** — gives both per-person conditioning and identity labels | No. Replaceable, and SAM2 is Apache-2.0 (UNVERIFIED — confirm before relying) |
| `ma_2d` **MammaNet** | ViT-Base (init from ViTPose-B) + CNN mask encoder, features summed; 512 **learnable per-landmark queries** through a transformer decoder; outputs pixel coords μ, uncertainty σ, visibility p, person-contact and floor-contact probabilities | **Yes. This is the whole system.** |
| association | symmetric epipolar distance over jointly-visible landmarks → affinity → **Hungarian** per view pair → **cycle-consistent** correspondence graph, connected components = people | Cheap to reimplement; published math |
| `ma_3d` | L-BFGS, 4 stages, `E = E_ldmks + E_shape + E_temp + E_cont`, 16 shape coefficients | No. We have ~this already |
| `ma_vis` | Overlays | No |

The 512 landmarks are farthest-point-sampled SMPL-X vertices, **weighted toward
hands, feet and head** — precisely the regions where our sparse pipeline is
worst. Training: 300k iterations, 4× A100, batch 24/GPU, ~3 days, Gaussian NLL
on landmarks, BCE on visibility, focal loss on contact. **Purely synthetic data.**

### Their own ablation, ranked by what it buys

1. **Dense landmarks are the foundation.** Hi4D MPJPE: MvP 92.77 → Faster
   VoxelPose 68.40 → MVPose 42.63 → 4D Association 41.29 → AvatarPose 32.10 →
   **MAMMA 12.44**. Every method above MAMMA is a sparse-keypoint method.
2. **Mask conditioning is an occlusion/association mechanism, not a general
   accuracy mechanism.** Two-person 2D error: Harmony4D 31.96 → 18.33 px,
   CHI3D 6.22 → 4.36 px. Single-person: it slightly *hurts* (RICH 8.55 → 8.83).
3. **Stage S2 does all the work.** Global-transform-only S1 sits near ~290 mm
   MPJPE; adding pose+shape+translation reprojection drops it to ~30 mm.
   Uncertainty re-weighting (S3) and contact (S4) barely move MPJPE.
4. **Contact optimisation buys plausibility, not accuracy.** Mean penetration
   depth 10.50 → 8.46 mm (the Vicon ground truth is itself 9.84 mm). MPJPE is
   essentially unchanged. **Floor-contact optimisation does not help at all**
   in the multiview case — the paper says so explicitly.
5. **Four cameras is already a good regime.** Accuracy saturates around 12
   cameras; two cameras degrades sharply. Our rig is not the bottleneck.
6. **Association is a solved sub-problem for them** — 100% identity success even
   at two cameras, attributed to dense landmarks plus temporal mask propagation.
7. **Data mix beats data volume.** MammaSyn + BEDLAM beats either alone; training
   *longer* on BEDLAM alone slightly degrades.

### What this tells us to build, in priority order

1. Densify the landmark set on our own rig. Biggest single lever.
2. Predict per-landmark **uncertainty and visibility**, and actually use them in
   a robust reprojection objective. This is what lets fitting converge with no
   learned pose prior.
3. Replace greedy per-frame association with temporal track propagation +
   Hungarian on epipolar affinity + cycle consistency.
4. Add a sequence-level optimisation with shared per-subject bone lengths.
5. Hands: high-resolution per-hand re-crop pass, then IK onto our own hand rig.
6. **Skip floor-contact optimisation.** Their evidence says it does nothing here.

---

## 4. License findings that change what the repo currently says

### 4.1 MAMMA cannot be our internal quality reference

`docs/PIPELINE_MAP.md` states MAMMA's license "blocks *shipping* its
code/weights/outputs, not internal use." Reading `.cache/mamma/LICENSE` directly,
that is not defensible:

- The grant is for "non-commercial scientific research, non-commercial education,
  or non-commercial artistic projects" **only**.
- "Any other use, in particular any use for commercial … purposes is prohibited.
  This includes, without limitation, incorporation in a commercial product, use
  in a commercial service, or **production of other artifacts for commercial
  purposes**."
- "This license also prohibits the use of the Data & Software to **train
  methods/algorithms/neural networks/etc. for commercial … use of any kind**."
- "By downloading the Data & Software, **you agree not to reverse engineer it**."

Using MAMMA outputs to benchmark a system we intend to sell is a commercial
purpose. Two consequences:

- **We need our own reference capture.** Section 7 covers how.
- **Whoever writes the replacement must not read MAMMA's source.** The
  no-reverse-engineering clause plus ordinary clean-room hygiene both point the
  same way. This research pass deliberately stopped at MAMMA's public README,
  paper, and LICENSE. Everything in section 3 comes from the published paper.

### 4.2 Every ingredient in MAMMA's training data is non-commercial

MAMMA's own LICENSE names the motion sources in MammaSyn. Verified:

| Source | License | Commercial training |
|---|---|---|
| Hi4D | ETH Zurich non-commercial | Explicitly forbidden, naming neural networks |
| Inter-X | CC-BY-NC-SA 4.0 + explicit clause | Explicitly forbidden |
| InterHand2.6M | CC BY-NC 4.0 | No |
| SignAvatars | CC BY-NC-SA | No |
| MOYO | MPI non-commercial | No |
| Harmony4D | **No license declared at all** | Worse than restricted — no grant exists |
| BEDLAM | see §6 | see §6 |

This is the real blocker. The recipe is public; none of the ingredients are ours.

### 4.3 The SMPL patent is live until 2036

[US10395411B2](https://patents.google.com/patent/US10395411B2/en) "Skinned
Multi-Person Linear Model" — priority 2015-06-24, filed 2016-06-23, granted
2019-08-27, **anticipated expiry 2036-08-12**, assignee Max-Planck-Gesellschaft,
**status active**. Family: EP3314577A1, WO2016207311A1, continuation
US20180315230A1.

Claim 1 is a *method for generating* a body model from multi-pose data aligned to
high-resolution 3D scans. That targets the act of **training** an SMPL-style
model — which is exactly what a clean-room parametric body model would involve.
Copyright-clean weights do not protect against a live method patent.

MAMMA's own license text says "Any copyright **or patent right** is owned by …
MPG", and its warranty disclaimer explicitly declines to warrant that use "will
not infringe any patents". Move AI holds patents in this space too (e.g.
US11867901), so the field is patented generally, not just by MPI.

**This must go to counsel before any owned-body-model work is costed.** It is the
single largest unpriced risk in the plan.

---

## 5. A commercially clean stack already exists

This is the most encouraging finding. Verified licenses:

| Component | License | What it gives us |
|---|---|---|
| [MMPose](https://github.com/open-mmlab/mmpose) RTMPose / RTMW | Apache-2.0 **(code only)** | Whole-body 2D architecture, 133 keypoints including fingers. **Adopt the architecture and training code; do not ship the released checkpoints** — they are trained on research-only data (§9) |
| [Pose2Sim](https://github.com/perfanalytics/pose2sim) | BSD-3-Clause | Calibration → sync → multi-person association → robust triangulation → filtering → LSTM marker augmentation → OpenSim scaling and IK. **No SMPL anywhere.** |
| [OpenSim core](https://github.com/opensim-org/opensim-core) | Apache-2.0 | Owned scaled skeleton, inverse kinematics and dynamics |
| [AniPose](https://github.com/lambdaloop/anipose) | BSD-2-Clause | Spatiotemporal triangulation: reprojection + temporal smoothness + **constant bone lengths auto-estimated per shot** |
| [mvpose](https://github.com/zju3dv/mvpose) (Dong et al. CVPR 2019) | Apache-2.0 | Cycle-consistent multi-way matching — the published fix for greedy per-frame association |
| [MVGFormer](https://github.com/XunshanMan/MVGFormer) | Apache-2.0 | Learned association that generalises across camera setups |
| [VoxelPose](https://github.com/microsoft/voxelpose-pytorch) / Faster-VoxelPose | MIT | Association-free 3D proposals (accuracy ceiling too low for hero shots) |
| [OpenCap](https://github.com/stanfordnmbl/opencap-core) core | Apache-2.0 | LSTM marker augmenter: sparse → dense virtual markers before IK |
| RapidPoseTriangulation | CC BY-SA 4.0 | Fast whole-body triangulation reference — **share-alike, treat with caution** |
| [SAM2](https://github.com/facebookresearch/sam2) | Apache-2.0 | Per-person masks **and** temporal identity propagation — the exact mechanism behind MAMMA's 100% identity matching |
| [rtmlib](https://github.com/Tau-J/rtmlib) | Apache-2.0 | ONNX-only RTMPose runner — no mmcv/mmdet build, practical for Mac local |

All licenses in the table above were confirmed via the GitHub API in this pass,
not inferred.

> **Trap: do not use Ultralytics YOLO.** MAMMA's mask stage pairs SAM2 with
> YOLO, and [ultralytics](https://github.com/ultralytics/ultralytics) is
> **AGPL-3.0** — network-copyleft, and a hard blocker for a proprietary service.
> Use RTMDet (Apache-2.0, MMDetection) for the person boxes that prompt SAM2, or
> prompt SAM2 manually on the first frame as MAMMA itself supports.

**Blocked — ideas only, reimplement the maths, do not read the code:** MAMMA,
EasyMocap (custom non-commercial), HaMeR (MIT code but requires non-commercial
MANO), WiLoR (CC BY-NC-ND), Sapiens (CC BY-NC), InterHand2.6M (CC BY-NC),
OpenPose (CMU non-commercial), 4D Association (no license file), AvatarPose (no
license file).

Pose2Sim reaching 3–4° joint-angle error against a marker-based reference, on a
BSD+Apache stack with no MPI asset in it, is the existence proof that a
license-clean pipeline can be biomechanics-grade.

---

## 6. Fitting without SMPL — the published blueprints

The key realisation from MAMMA's own formulation: **nothing in its landmark
energy requires SMPL-X.** It requires (a) K surface points rigidly associated
with a skinned surface, and (b) a detector trained to find them in images. Our
own rig satisfies (a). MAMMA's evidence is that the dense-landmark term, not the
body prior, is what wins.

Concrete routes, strongest first:

1. **OpenSim-style scaled skeleton + IK** (the Pose2Sim recipe) — scale a rig
   once per subject from a static trial, then per-frame weighted least-squares
   IK. Entirely permissive, validated to 3–4°.
2. **Differentiable biomechanics** ([Cotton et al.](https://arxiv.org/abs/2402.17192),
   [with uncertainty](https://arxiv.org/abs/2502.06486)) — a differentiable
   MuJoCo biomechanical model with an implicit trajectory representation,
   optimised end-to-end against multi-view keypoint reprojection, with
   **per-subject skeleton scaling, marker offsets and poses solved jointly**.
   This is precisely the sequence-level bundle optimisation our P0 lacks, on a
   self-owned rig. The follow-up adds a variational posterior giving
   **confidence intervals** — genuinely useful for an animator-facing tool.
3. **AniPose spatiotemporal triangulation** — cheapest immediate upgrade to our
   temporal repair, and BSD-licensed.
4. **4D Association** fits a plain kinematic skeleton at 30 fps with no mesh at
   all — proof that skeleton-only multi-person capture works. Maths reusable,
   code unlicensed.
5. **Hands without MANO** — [IK of a biomechanical hand onto foundation-model
   keypoints](https://arxiv.org/abs/2605.09258): ~10° finger joint angles, ~6 mm
   positions.

---

## 7. Hands from four wide cameras — the honest ceiling

This is our named N5.1 blocker, so it deserves a straight answer.

- BlazePose GHUM states the problem plainly: a 256×256 body input is
  "insufficient to capture hand details," and a single pixel of error matters far
  more for a fingertip than for a hip. Their fix is a **separate hand model run
  on high-resolution re-crops** seeded by palm predictions from the body pass:
  normalised error 11.8% → 9.7%, 3D error 27 → 18 mm.
  [Paper](https://arxiv.org/pdf/2206.11678)
- MAMMA rendered its training data at 2056×1504, "roughly twice the resolution of
  BEDLAM… to ensure much finer detail in the hand regions."
- Triangulating fingers from wide shots is measurably worse than body:
  RapidPoseTriangulation reports 41.8 mm hand MPJPE vs 29.1 mm body on H3WB.
- Even MAMMA concedes "the accuracy of hand-motion recovery can still be
  improved," and notes most marker-based captures ignore hands entirely because
  finger marker cleanup dominated their Vicon baseline at 4.95 h per sequence.

**Expect: body-grade wrists, pose-plausible fingers (~10°, tens of mm),
unreliable finger contact.** The three published ways to beat that are more
cameras, more resolution, or a per-hand re-crop pass — and the re-crop pass is
the only one we can ship without changing the rig. No paper states a canonical
"N pixels across the hand" threshold; every serious system instead re-crops the
hand to a full network input.

The clean route: our own detector's hand branch (RTMW-style architecture,
**retrained on our synthetic corpus** — the released RTMW checkpoints are not
usable, see §9) → per-hand high-res re-crop → existing confidence-weighted
triangulation → IK onto our own hand rig with per-subject finger bone-length
calibration.

---

## 8. The body model: we no longer need to build one

`docs/COMMERCIAL_MULTIVIEW_BODY.md` P3 offered two routes — license SMPL-X
commercially, or build an owned parametric body ("a dataset and model program,
not a small code phase"). **A third route opened in late 2025 and it is better
than both.**

### Apache-2.0 parametric body models that now exist

| Model | What it is | License | Verdict |
|---|---|---|---|
| **[MHR](https://github.com/facebookresearch/MHR)** (Momentum Human Rig, Meta, Nov 2025) | Full parametric body: skeleton, skinned mesh with LODs 0–6, **45 shape params** (20 body / 20 head / 5 hands), **204 pose params with articulated hands**, 72 FACS expression blendshapes, **neural pose correctives**. The released form of [ATLAS](https://arxiv.org/abs/2508.15767), "learned from 600k high-resolution scans captured using 240 synchronized cameras" | **Apache-2.0, weights included** — confirmed on the repo LICENSE *and* on `assets/LICENSE.txt` inside the release archive | **Strongest candidate** |
| **[SOMA-X](https://huggingface.co/nvidia/SOMA-X)** (NVIDIA) | Decouples identity from pose; ~18,095 vertices, 77 joints, optional MLP pose correctives; identity backends for SOMA-shape, SMPL, SMPL-X, MHR, Anny. Trained on SizeUSA + TripleGangers scans **commercially licensed by NVIDIA** | Apache-2.0; card says "ready for commercial use" | Strong — and **we already have a SOMA lane** |
| **[Anny](https://github.com/naver/anny)** (NAVER, Nov 2025) | Differentiable parametric body for all ages, built on MakeHuman's CC0 assets | Apache-2.0 code + CC0 assets — **but note** the repo's LICENSE is a composite custom file, so GitHub reports it as `NOASSERTION`, not a clean SPDX tag. Read it before relying on it | Good fallback; artist-built rather than scan-learned |
| **MakeHuman / MPFB2 base mesh + targets** | What AutoAnim's `hm08` asset already uses | **CC0 1.0** for all assets (app code is AGPL — don't link it) | Already in production here |

Two carve-outs to respect: SOMA-X ships `SMPL/base_body.obj` and
`SMPLX/base_body.obj` in its Apache-tagged repo — **exclude those two backends
pending legal review**; and Anny's optional SMPL-X-topology addon is marked
non-commercial — **exclude it**.

Non-commercial and therefore out: SMPL-X (default), SUPR, SKEL/BSM, STAR, Meta
Sapiens/Goliath/ca_body (CC BY-NC). GHUM's model is gated behind an offline form
with no published license — assume closed. Daz Genesis and Reallusion CC4 base
bodies are effectively prohibited as fitting targets: both EULAs bar shipping
derived meshes in open formats, and Reallusion explicitly bans use "for online
or in-software character generation."

This is a real de-risking. The existing `hm08`/MPFB body is already CC0, and MHR
is a drop-in upgrade path that brings articulated hands and learned pose
correctives — the two things our rig lacks — at no license cost.

### The official SMPL route, for completeness

**The Meshcapade route has closed.** Their homepage now reads, verbatim:
*"Meshcapade is now part of Epic Games. As of April 18th, the Meshcapade online
platforms have been shut down."* Max-Planck-Innovation
[states](https://www.eurekalert.org/news-releases/1117236) it "will now directly
take over the licensing of the SMPL technology," so the remaining channel is
`smpl@max-planck-innovation.de` (or `ps-license@tue.mpg.de` for the datasets).
No pricing has ever been published and no public case exists of anyone
announcing a completed commercial licence.

Given MHR exists under Apache-2.0 with weights, this is a fallback we probably
never exercise. One useful precedent if we ever negotiate: MPI already releases
**SMPL-Body** — exported body meshes and rigs *without* shape blendshapes —
under [CC BY 4.0](https://smpl.is.tue.mpg.de/bodylicense.html).

### The patent problem does not go away

This is the part that Apache-2.0 cannot fix, and it is the largest unpriced risk
in this whole document.

**Apache-2.0 grants you Meta's, NVIDIA's and NAVER's patent rights. It cannot
grant Max Planck's.**

| Patent | Covers | Expiry | Status |
|---|---|---|---|
| [US10395411B2](https://patents.google.com/patent/US10395411B2/en) + [US11017577B2](https://patents.google.com/patent/US11017577B2/en) + EP3314577B1 (MPG) | Claim 1: **runtime use** of template + shape-dependent blendshape + pose-dependent blendshape + blend skinning. Claim 13: the **method of learning** it from registered scans. Claim 22: a medium performing it | ~2036 | **Active**; EP divisional EP4571665A3 still in prosecution as of 2025 |
| [US9189886B2](https://patents.google.com/patent/US9189886B2/en) family (**Brown University**, Black et al.) | Estimating body shape from imprecise/partial sensor data using a low-dimensional parametric model that factors pose from shape | ~2032 | Active — **reads on the capture product itself, whichever body model we ship** |
| [US10755464B2](https://patents.google.com/patent/US10755464B2/en) (MPG) | Co-registration: simultaneous alignment and modelling of articulated 3D shapes — i.e. the build-your-own-model pipeline | ~2032 | Active |
| [US9710964B2](https://patents.google.com/patent/US9710964B2/en) (MPG) | MoSh: personalised body models from mocap markers | ~2035 | Active |

Claim 1 of US10395411B2 has **no limitation on how the model was trained**.
Whether MHR's neural correctives or Anny's morph-target construction fall inside
"applies a pose-dependent blend shape… according to a static soft-tissue
deformation" is a claim-construction question, and validity is arguable
(pose-space deformation prior art, Lewis et al. 2000, is cited on the patent's
face). Neither question is one we can answer ourselves.

Note also that the second family attaches to the *capture product*, not the body
model — so switching to MHR does not avoid it, and neither would building our
own model from scratch.

**Action: commission a freedom-to-operate opinion before any commercial launch.**
Not before engineering — the engineering is the same either way — but before
launch, and certainly before any investment decision that assumes the product is
sellable. Also note the field is patented generally: Move AI holds US11867901 in
the same space.

### If we did build our own model anyway

For completeness, the verified recipe is smaller than the P3 text implies:

- **Shape space:** ~3,800–10,000 static scans. SMPL used CAESAR registrations
  (~1,700 male / ~2,100 female). Buyable today: [WEAR](https://www.bodysizeshape.com/page-1855750)
  ships CAESAR NA/IT/NL scans for **€1,095** with "you can do anything you like
  with it"; caesar-database.com sells derivatives at $2,500–5,000.
  [Human Alloy](https://humanalloy.com/pages/license-3d-human-scan-data-for-ai-training)
  and [HumanDataset](https://humandataset.com/faq/) (Renderpeople's data arm)
  both have explicit negotiated ML-training lanes.
- **Pose correctives:** surprisingly cheap — SMPL used **1,786 registrations of
  40 individuals**, roughly 45 poses each.
- **Hands:** MANO used ~31 subjects across a 31-pose grasp taxonomy (~1–2k scans).
- **Face:** FLAME used ~3,800 neutral heads plus ~33,000 total scans, mostly 4D
  expression sequences. A 4D scanner matters more than subject count.

The moat is the co-registration pipeline, not the maths — and that pipeline is
itself patented to ~2032. **Given MHR exists under Apache-2.0, building our own
buys nothing on the copyright axis and nothing on the patent axis.** Do not do it.

Every mainstream academic scan corpus is contractually closed to us: THuman 2.0/3.0/4.0,
2K2K, Hi4D, X-Humans, MVHumanNet, CAPE, AGORA and DFAUST all prohibit commercial
use, several banning commercial neural-network training in so many words.
3D Scan Store goes further and bans the *end product* ("character generators or
digital humans created using AI training data derived from 3d scan store models").

---

## 9. The training data — this is the actual battle

Everything else in this document has an answer. This is the part that has to be
built.

### The problem, stated exactly

No off-the-shelf model gives commercially-clean dense surface landmarks, and
almost no dataset that could train one is usable.

| What we'd want | Why we can't have it |
|---|---|
| MammaSyn | MPI non-commercial; bars commercial training outright |
| BEDLAM | Same MPI clause, verbatim: bars training "for commercial… use of any kind" ([license](https://bedlam.is.tue.mpg.de/license.html)) |
| AGORA | Same clause |
| **AMASS** | Same clause — and it gates every BEDLAM-style pipeline, because BEDLAM's 2,311 motions are AMASS samples ([license](https://amass.is.tue.mpg.de/license.html)). **But see §9a — AMASS is a derivative compilation and its raw constituents are separately licensed. This one is routable around.** |
| Microsoft SynthMoCap | Research Use of Data Agreement v1.0: "you may not use the Data **or any Results** in any commercial offering," and §5.5 defines models trained on it as Results ([license](https://raw.githubusercontent.com/microsoft/SynthMoCap/main/LICENSE)) |
| SynBody | CC BY-NC-SA 4.0 |
| GTA-Human | Derived from Rockstar game content; `license:other` |
| COCO-WholeBody | "ONLY for research and non-commercial use" |
| Halpe-FullBody | No license; AlphaPose parent is academic-only |
| Human3.6M | "Licenses free of charge are limited to academic use only" |
| MPII | No license text found; images are YouTube frames. Risky |

And that contamination reaches the checkpoints. **RTMW's weights are not clean**
even though MMPose's code is Apache-2.0: RTMW is trained on "Cocktail14", which
includes COCO-WholeBody, Halpe and InterHand. Same for DWPose. ViTPose's weights
are Apache-2.0 but its training images are the same encumbered academic sets.
The only off-the-shelf pose model with verified-clean provenance is **MediaPipe
BlazePose GHUM** — Apache-2.0, trained on Google's own consented data (30k
consented + 85k fitness images, per its model card) — and it gives 33 keypoints,
which is nowhere near dense enough to be the end state.

**So: we train our own detector, on synthetic data we generate, from assets we
own or have explicitly licensed.** That is the battle.

It is a smaller battle than it first appears. §9a works through the motion
sourcing and finds ~7,300 commercially-clean clips available free — more than
BEDLAM used. What remains is rendering and engineering effort, not a data
acquisition problem.

### It is a proven battle, though

Microsoft has done this twice at production quality with wholly-owned assets,
and published how:

- **[Fake it till you make it](https://arxiv.org/html/2109.15102)** (ICCV 2021) —
  a face model from 511 scans, 512 hairstyles, 200 texture sets, 448 HDRIs,
  rendered in **Blender/Cycles**: 100,000 images at 512², 48 hours on 150 GPUs.
  No domain adaptation — the gap was closed at the source.
- **[Dense landmarks](https://arxiv.org/html/2204.02776)** (ECCV 2022) — 703
  dense landmarks, each a 2D Gaussian, trained on 100k synthetic images with the
  **Gaussian NLL loss** `Σ λᵢ(log σᵢ² + ‖μᵢ−μᵢ′‖²/2σᵢ²)`. **This is the exact
  loss and output head MAMMA's landmark branch descends from.**
- **[Look Ma, no markers](https://arxiv.org/html/2410.11520v1)** (SIGGRAPH Asia
  2024) — 1,428 body + 141-per-hand + 744 face landmarks; HRNet-w48; ~100k
  images per part from 3,572 body scans; L-BFGS fitting with a normalizing-flow
  pose prior.

The published evidence is unambiguous: **dense-landmark detectors trained on
~100k frames of well-crafted synthetic data transfer to real footage with no
real-image training at all.** That is the single most important derisking fact
in this document.

### Every ingredient has a verified-clean option

| Ingredient | Clean option | Status |
|---|---|---|
| **Body model / landmark surface** | **MHR** (Apache-2.0 incl. weights, §8). Define the 512 landmarks as farthest-point-sampled MHR vertices, exactly as MAMMA did on SMPL-X | **Solved** — this is what §8 buys us |
| **Motions** | **CMU + Eyes JAPAN + ACCAD + 100STYLE ≈ 7,300 clips / 34+ h, all free and commercially clean** — 2.4–3.6× what BEDLAM used. See §9a for the per-source verification | **Solved**. Never AMASS |
| **Environments / lighting** | **[Poly Haven](https://polyhaven.com/license)** — CC0, "any purpose, including commercial work" | **Solved** |
| **Renderer** | **[Blender/Cycles](https://www.blender.org/about/license/)** — "What you create with Blender is your sole property." The GPL covers the software, not the renders. This is literally Microsoft's stack | **Solved** |
| **Clothing** | CLO3D sim (BEDLAM's choice; output terms UNVERIFIED — read before buying), or Daz clothing via their AI program, or Reallusion **Enterprise** tier | Needs a purchase decision |
| **Hair** | Reallusion **Enterprise** license — their standard content license excludes "AI training and deep learning", but Enterprise is explicitly available "for special arrangements such as AI training, machine learning" | Needs a purchase decision |
| **Textures / scanned people** | [Daz AI training data program](https://www.daz3d.com/ai-training-data) ("Commercially safe training data with clear licensing"), or Renderpeople with written ML consent, or our own scans | Needs a purchase decision |
| **Eval set** | Our own consented capture. Unavoidable — see §10 | Needs a shoot |

> **Do not build the corpus on MetaHuman.** Epic's position is that MetaHumans
> may be used in workflows incorporating AI, "but not to train or enhance the AI
> models themselves." The Unreal EULA itself 403s to automated fetch and the
> MetaHuman FAQ answer text could not be retrieved, so treat this as flagged
> rather than settled — but Blender/Cycles has no such ambiguity, so there is no
> reason to take the risk.

### Scale and cost

Anchored to published numbers, not guesses:

| Milestone | Scale | Reference point |
|---|---|---|
| First working single-person dense-landmark model | **~100–200k rendered crops** | Microsoft's proven scale, all three papers |
| MAMMA-class multi-person, multi-view, contact-heavy | **~1M crops** ≈ 3k sequences × 8 views | BEDLAM 380k frames / 1M boxes; MammaSyn 955k images / 2.5M crops |

Asset counts to target, from the fetched papers: a few hundred identities and
outfits, ~25–100 hairstyles, ~100 skin textures, ~100–450 HDRIs, ~2–3k motion
clips. Training itself is cheap by comparison — MAMMA's was 300k iterations on
4× A100 for ~3 days, and Microsoft's face model was 150 GPUs × 48 h **in 2021**.
The render bill is the dominant cost and it is cloud-friendly and embarrassingly
parallel, which suits our existing Modal setup.

Render at high resolution. MAMMA used 2056×1504, "roughly twice the resolution
of BEDLAM… to ensure much finer detail in the hand regions" — and hands are our
named blocker.

---

### 9a. The AMASS workaround — it exists, and it is free

**AMASS is a derivative compilation.** It unifies pre-existing optical mocap
datasets and re-fits them to SMPL with MoSh++. The AMASS license binds the
SMPL-fitted output — but **the raw source data at the original hosts carries its
own, separate licence**, and we never need the SMPL fits because we retarget raw
joint angles onto our own rig anyway.

So the question is not "can we use AMASS" (no) but "which constituents are clean
at the source". Verified, per dataset, at the original host:

| Source | Licence at the original host | Clips | Verdict |
|---|---|---|---|
| **CMU Graphics Lab** | *"free for all uses"*; *"You may include this data in commercially-sold products"* | ~2,400–2,600 trials, 100+ subjects | **COMMERCIAL OK** |
| **Eyes JAPAN** (free tier) | CC BY 2.1 JP — the page also says BY-SA 2.1 JP elsewhere; assume the stricter | 4,671 free BVH files | **COMMERCIAL OK** |
| **ACCAD Open Motion Project** | CC BY 3.0 Unported | 258 motions, 20 subjects | **COMMERCIAL OK** |
| KIT / EKUT / WEIZMANN / CNRS | **No licence published anywhere on the site** | ~4,600 motions, 11.5 h | SILENT — one email away (`motiondatabase@lists.kit.edu`) |
| MPI HDM05 | Original host 403s; archived snapshot shows CC BY-SA 3.0 | 219 motions | UNVERIFIED — confirm before use |
| BMLrub, BMLmovi | Custom Troje licence: bars commercial use *and* commercial NN training | ~12,000 motions | RESEARCH ONLY |
| SFU | *"cannot be used for commercial products or resale"* | 44 | RESEARCH ONLY |
| TotalCapture | Surrey research agreement | 40 | RESEARCH ONLY |
| MPI sets (MoSh, PosePrior, HumanEva, DFaust, GRAB, SOMA, MOYO), DanceDB, SSM, Transitions | MPI/UCY non-commercial template | bulk of AMASS hours | RESEARCH ONLY |

**Plus non-AMASS sources that are outright free and clean:**

| Source | Licence | Scale |
|---|---|---|
| [100STYLE](https://www.ianxmason.com/100style/) (Edinburgh) | CC BY 4.0 | ~18.5 h, 100 locomotion styles, 4M+ frames |
| [ASPset-510](https://github.com/anibali/aspset-510) | **CC0-1.0** (verified) | 510 sports clips |
| [ContactPose](https://github.com/facebookresearch/ContactPose) | **MIT** (verified) | 2,306 real grasps, 50 subjects |

**The arithmetic works comfortably.** BEDLAM sampled 2,311 clips from AMASS;
MammaSyn used 2.8k sequences. **CMU alone matches that scale**, and CMU + Eyes
JAPAN + ACCAD + 100STYLE gives roughly **7,300+ clips and 34+ hours** — 2.4–3.6×
what we need, at zero cost.

> **The one CMU clause that constrains the product:** *"you may not resell this
> data directly, even in converted form."* Rendering training images from it and
> selling a model trained on those images is squarely inside "include this data
> in commercially-sold products". **Shipping the retargeted BVH/FBX itself is
> not.** Keep motion files internal to the render farm; never ship them. Carry
> the requested acknowledgement in pipeline metadata.

### Filling the three diversity gaps

MAMMA's ablation says the wins came from extreme poses, hand articulation and
interaction. CMU covers none of them well. Clean fills:

**A candidate that would close the hand gap properly — blocked at the licence
gate.** [EgoSuite-Open100K](https://huggingface.co/blog/LightwheelAI/egosuite-open100k)
(LightwheelAI, 2026) ships **hand pose as data**: 21 joints, positions `(21,3)`
and rotations `(21,4)` as quaternions in world coordinates, per-episode time
series. **Not MANO** — "MANO" and "SMPL" appear nowhere in any card — so it is
retargetable onto MHR's 27-DoF hands. 50 h downloadable now, ~12 TB in the gated
buckets, 100,000 h planned. Against ContactPose's 2,306 *static* grasps that is
orders of magnitude more articulation-over-time.

Its imagery is useless to us, but not for the obvious reason: viewpoint gaps
turn out to matter little for pose pretraining, whereas *resolution* does, and
our hands are ~20 px where theirs are close-ups.

**Blocked, and correctly so.** The HF metadata says
`license: "other"`, `license_name: "commercial-training-no-resale-v1.0"` — and
**no verbatim licence text exists at any public URL**. EgoStandard and EgoPro
contain one file each (a README); the data sits behind a manually-reviewed gate,
and the card points at "the repository license" which is not published. A licence
*name* and a blog sentence are not terms. The clause that decides it for us:
**does "no resale" reach derivatives** — synthetic renders generated from their
poses, and model weights trained on those renders? Derivative generation is our
entire use case. If it does, walk away.

**Hands — solved, and cheaper than expected.** I had assumed we needed hand
mocap. We don't. Microsoft's own paper states they *splice* hand poses in:
*"If the body pose sequence lacks articulated hands then we splice these in from
the hand pose library."* So what's needed is a **static hand-pose library**, not
hand animation. Clean sources: **ContactPose (MIT) — 2,306 real functional
grasps from 50 subjects**, retargeted onto our own hand rig; plus hand-authored
poses from the Feix GRASP taxonomy, which is a *paper*, not a dataset, so no
licence attaches. Everything else in this space (InterHand2.6M, SignAvatars,
GRAB, ARCTIC, DexYCB, FreiHAND, Re:InterHand) is non-commercial.
*Caveat:* ContactPose ships MANO parameters and the MANO **model** is
research-only — take the joint angles, retarget to our rig, don't use MANO.

**Extreme poses.** ASPset-510 (CC0) + CMU's "Physical Activities & Sports"
category + Eyes JAPAN's karate/kung-fu/gymnastics/acrobatics packs. For true
MOYO-grade contortion there is no commercially-clean equivalent — that's a
one-day capture session with a yoga performer, which we own outright.

**Two-person interaction.** CMU has verified two-subject sections (Subject #18
"human interaction and communication", #33 "throw and catch football") plus a
"Human Interaction" category. Beyond that: purchased paired-animation packs, or
the Battle 2 shoot. And note §3 — **contact optimisation buys plausibility, not
MPJPE**, and our use case is dialogue, not grappling. Much of MammaSyn-I's value
was *occlusion* robustness, which we can synthesise by placing independent
single-person clips in one render volume — MammaSyn itself instantiated "2–6
subjects at random positions."

### Motion sources to avoid — two are traps

| Source | Status |
|---|---|
| **Mixamo / Adobe** | **ML FORBIDDEN.** Adobe General Terms §17(C) bars using output "to directly or indirectly create, train, test, or otherwise improve any machine learning algorithms." Free, and a trap |
| **Reallusion ActorCore** | **ML FORBIDDEN** under the standard EULA §2.1(C): "You may NOT use the Content… For machine learning, AI training, or AI-generated output." Enterprise tier negotiable |
| **TurboSquid** | ML forbidden without written authorisation |
| **CGTrader** | **Permitted** for *paid* assets under their standard Royalty Free License §21A.2, which explicitly names ML training. Free downloads and NoAI-flagged assets are excluded |
| **Truebones** | "royalty free… any and all purposes even commercial"; ML not addressed. $99 for 20k+ motions. Confirm in writing first |
| **MoCap Online AI Permit** | Covers "Model Training" — **but restricts training "tools that would compete with professional motion capture services."** That is arguably exactly what we are building. Declare the model's output explicitly in the application; do not assume approval |

### Three emails worth sending

1. **KIT H²T** (`motiondatabase@lists.kit.edu`) — 4,600 motions with no published
   licence at all. Cheapest large win in this section.
2. **HDM05** (`HDM05@mpi-inf.mpg.de`) — confirm the archived CC BY-SA 3.0.
3. **Truebones** — confirm ML training before relying on the $99 pack.

---

## 9b. Error budget — what actually stands between us and 20 mm

Everything in this section that says "measured" was computed on **our own rig and
our own reconstruction**, not taken from a paper. Scripts are in the session
scratchpad (`geom_floor.py`, `tri_floor.py`, `velocity_budget.py`) against
`artifacts/commercial-multiview-mamma-comparison/`.

### Two ledgers, and only one of them is ours

**Ledger B — what "ground truth" is worth.** A marker-based reference is not
truth. Soft-tissue artifact moves skin markers relative to bone by
**>30 mm on the thigh**, 15–21 mm on the shank, and the acromion cluster slides
**28.7 ± 4.0 mm** during arm elevation. Hip-centre regression adds 5–12 mm per
axis. The Vicon instrument itself is excellent (0.15 mm static, <2 mm dynamic) —
the body is the problem. And the whole Vicon + MoSh++ pipeline predicts held-out
markers at **21.6 mm**.

So: **nothing optical can be validated below ~20 mm using a marker reference,
including marker systems.** 20 mm is not "20 mm from bone" — it is
*statistically indistinguishable from the reference*. That is the right goal and
it is achievable; anything below ~15 mm would be measuring agreement with the
reference's own artifact.

### Ledger A — our rig, measured

**Rig geometry is excellent and is not a limiter.** The four cameras sit at
84.7° / 92.0° / 87.7° / 95.6° apart with two opposed pairs at 179.5° / 176.7°,
baselines 6.3–9.7 m, depth 4.6–5.0 m to the volume centre. Near-ideal.

**Triangulation error at our current detector is ~25 mm.** Monte-Carlo, 4,000
trials, DLT over all four cameras, driven by our measured reprojection residual:

| Detector width | mm/px at subject | @ our median 4.63 px | @ our P95 11.05 px |
|---|---|---|---|
| **1280 (current default)** | 5.57 | **25.1 mm** | 59.2 mm |
| 1920 | 3.71 | 16.6 mm | 39.8 mm |
| 3840 (native footage) | 1.86 | 8.3 mm | 20.3 mm |

*(mm/px cross-checked two ways: geometrically from rig-centre depth, and
empirically from the median depth of triangulated joints along each camera's
optical axis. Both give 5.5–5.6 mm/px at 1280.)*

**So detector noise is a major term in our 60–80 mm gap, not a minor one.**
At 1280, Apple Vision's ~4.63 px median residual is ~26 mm at the subject, and
that alone triangulates to ~25 mm median and ~59 mm P95 of 3D error.

> **Correction.** An earlier draft of this section claimed triangulation
> contributes only ~6 mm and that reconstruction jitter is ~2 mm, and concluded
> the gap was almost entirely systematic bias. Both figures were wrong. The
> 6 mm came from a quaternion-ordering bug in the analysis script — the rig
> stores `wxyz` and `commercial_multiview.py:218` reorders to `xyzw`, which the
> script did not. And the ~2 mm "jitter" was measured on
> `triangulated_world_positions_z_up_m`, which is **post-`np.interp` and
> Savitzky-Golay smoothed** (`_fill_and_smooth_positions`), so it cannot measure
> raw triangulation noise. The corrected picture is above.

**But — and this is the finding that matters — that error does not fall when you
give the detector more pixels.** Battle 0 measured it directly
(`docs/BATTLE0_DETECTOR_WIDTH_FINDINGS.md`): controlled for the pipeline's
fixed-pixel gates, reprojection error is flat across 1280 / 1920 / 3840. Apple
Vision carries roughly 26 mm of 2D error at the subject **regardless of input
resolution**. It is model-limited, not pixel-limited.

That is precisely the argument for replacing the detector rather than tuning the
rig. And it is why dense landmarks are the answer specifically: a dense landmark
is a vertex of the body model, so it cannot be semantically inconsistent across
views the way an anatomically-vague "shoulder" keypoint can — the literature
reports 29–53 mm of systematic hip/knee error in sparse markerless systems
attributable to training-label convention alone.

### The hard physical gate: synchronisation

Measured P95 joint speeds on our own footage, **body joints only**:
right_wrist 2.85, left_wrist 2.74, left_ankle 2.72, right_ankle 2.27,
left_elbow 2.07, neck 2.02 m/s. Median P95 across body joints: **1.76 m/s**.

*Two caveats.* First, these are finite differences of a noisy 19-point
reconstruction, so the tail partly reflects detector jitter rather than real
motion. Second, the **facial landmarks are excluded** — right_eye reads
3.85 m/s P95 with a 9.03 m/s maximum, which is not plausible as bulk head
movement on a wide shot and is almost certainly detector noise. Excluding them
is the honest choice; it also lowers the headline number.

Third, these speeds come from `triangulated_world_positions_z_up_m`, which is
post-interpolation and Savitzky-Golay smoothed — so they are *under*estimates of
true joint speed. That is conservative in the right direction for a sync budget,
but it means the table below is a floor, not a ceiling.

The literature's design velocity for dialogue acting is ~1 m/s, rising to
2–4 m/s for emphatic gesture, which brackets what we measured.

| Sync offset | @ 1.76 m/s (our median body-joint P95) | @ 2.85 m/s (our fastest body joint) |
|---|---|---|
| <250 µs (verified software sync) | 0.4 mm | 0.7 mm |
| 5 ms | 8.8 mm | 14.2 mm |
| 8.3 ms (½ frame @ 60 fps) | 14.6 mm | 23.6 mm |
| 16.7 ms (½ frame @ 30 fps) | **29.3 mm** | **47.5 mm** |
| 33 ms (2 frames @ 60 fps) | **57.9 mm** | **93.9 mm** |

**The finding that should change how we shoot:** MAMMA's consumer rig — four
iPhone 17 Pro Max on a Blackmagic dock with Ambient LockIt genlock *and* LTC
timecode — still showed *"occasional synchronization deviations of up to two
frames"* at 60 fps. **Genlock is necessary and not sufficient. Sync must be
measured every take**, with an LED flash or clap visible to all four cameras.
A single silent two-frame slip explains most of a 60–80 mm error on its own.

### The full budget

Independent terms combine in quadrature. Dialogue-speed motion, 4K, ~4 m depth:

| Term | Sloppy rig | Well-run rig | How to fix |
|---|---|---|---|
| **2D landmark error → 3D** | 20–45 mm (sparse keypoints; 30–50 mm label-convention bias at hip/knee) | **8–15 mm** | Dense landmarks, 4K, performer large in frame. **This is the programme** |
| Synchronisation | 15–58 mm (½ frame @30 to 2 frames @60) | **<1 mm** | Genlock **plus per-take verification** |
| Rolling shutter | 10–30 mm (slow 4K30 readout) | **2–5 mm** | Fast-readout sensor, 60 fps |
| Motion blur | 8–17 mm (180° shutter) | **1–4 mm** | 1/500–1/1000 s + more light (MAMMA strobes at 12k lux) |
| Clothing offset | 10–30 mm systematic | **5–12 mm** | Fitted wardrobe, or pre-scan the performer |
| Occlusion, 2 performers | 10–25 mm episodic | **4–7 mm** | Surround placement, not an arc |
| Calibration + distortion | 5–15 mm (stale, wand-only) | **1–3 mm** | Board+wand bundle adjustment per shoot, validated on held-out geometry |
| Triangulation, given the detector (measured, ours) | 25 mm @1280 | **8–15 mm** | Falls out of a better detector; rig geometry is already near-ideal |
| Body model expressiveness | 2–5 mm | 2–5 mm | MHR ≈ 4.2 mm vs SMPL-X ≈ 4.6 mm vertex-to-surface — a wash |
| **Camera count, 4 vs 16** | — | **+0.5 mm single / +3 mm two-person** | Smallest term on the list |
| **RSS total** | **~35–75 mm** ← where we are | **~12–18 mm** | |

### Is 20 mm achievable at four cameras? Yes — with no margin.

MAMMA's own camera-count ablation, read off their Fig. 12:

| Cameras | Single-person MPJPE | Two-person MPJPE |
|---|---|---|
| 2 | ~24 mm | ~47.5 mm |
| **4** | **~13.5 mm** | **~20 mm** |
| 8 | ~13.5 mm | ~17.5 mm |
| 12–16 | ~13 mm | ~17.3 mm |

**Four cameras costs ~0.5 mm single-person and ~2.7 mm two-person versus sixteen.**
Camera count is not our problem. But two-person at four cameras lands *exactly*
on 20 mm — meaning every other term has to be well-run, not merely acceptable.
Note also that MAMMA published **no quantitative accuracy at all** for their
4-iPhone consumer rig, only that it looked plausible with flicker in 5 of 46
sequences. Studio discipline is doing real work in that 20 mm.

And expect the honest number to be worse than MPJPE: on the same data MAMMA
scores 12.96 mm MPJPE, 17.18 mm PVE, and **22.48 mm held-out marker error**.
Marker-honest runs 5–10 mm behind joint-centre MPJPE. **Report all three.**

### Order of operations, by millimetres-per-effort

1. **Measure sync on every take.** Cheapest large win. Assume the rig slips until
   proven otherwise — MAMMA's genlocked rig did.
2. **Dense-landmark front end.** Most of the gap. Note that detector *resolution*
   is not the lever — Battle 0 tested it directly (see
   `docs/BATTLE0_DETECTOR_WIDTH_FINDINGS.md`): going 1280 → 1920 → 3840
   leaves reprojection error flat at 25.8 → 25.8 → 25.5 mm, and a direct sweep
   shows Apple Vision saturates below 1280. It is model-limited, not
   pixel-limited. The lever that *did* pay was **JPEG quality** — pinning
   `-q:v 2` at 1280 beats the 1920 lane outright.
3. **Board+wand bundle-adjusted calibration per shoot**, validated by
   triangulating known lengths, not by reprojection RMS — reprojection RMS is a
   training error and wand-only calibration overfits it.
4. **Fast shutter, fast readout, more light.**
5. **Fitted costume, or pre-scan each performer.**
6. **Only then** consider cameras five and six. Worth ~3 mm for two-person
   occlusion; nothing else.

---

## 10. Recommendation

**Build it. The answer to the question is yes.**

The shape of the thing:

```text
own 4-cam capture
  |
  +-- RTMDet person boxes (Apache-2.0)  ->  SAM2 masks + temporal ID propagation (Apache-2.0)
  |
  +-- OUR dense-landmark net  <--- trained on OUR synthetic corpus
  |     512 MHR-surface landmarks + uncertainty + visibility
  |     ViT-B + mask CNN + per-landmark learnable queries, Gaussian NLL
  |
  +-- association: epipolar affinity -> Hungarian -> cycle-consistent graph
  |
  +-- sequence-level fit to MHR (Apache-2.0): robust reprojection weighted by
  |     visibility/uncertainty, shared per-subject shape, temporal term, L-BFGS
  |
  +-- per-hand high-res re-crop pass -> triangulation -> IK onto our hand rig
  |
  `-- AutoAnim-55 body track  ->  existing compositor / N5.1 assembly
```

Not one Max Planck asset in it. The only novel engineering is the detector and
its data factory; everything else is either already built here or available
permissively.

### Sequencing — one battle at a time

**Battle 1 — the sparse-pipeline upgrade, no new data required (weeks).**
Four independent changes, every component Apache/BSD, no training and no
licensing exposure:

- **Masks and identity:** add SAM2 for per-person masks with temporal
  propagation. This gives person separation and identity tracking — it does *not*
  produce keypoints.
- **Detector:** the only clean off-the-shelf upgrade available today is
  **MediaPipe BlazePose GHUM** (Apache-2.0, verified-clean provenance, §9) at 33
  landmarks versus Apple Vision's 19. A modest improvement, and it adds feet.
- **Association:** replace greedy per-frame permutation search with
  cycle-consistent Hungarian matching on epipolar affinity.
- **Triangulation:** AniPose-style spatiotemporal solve with per-shot bone-length
  estimation.

Plus one small measured change, **already applied** (2026-08-27): **JPEG quality
is pinned (`-q:v 2`) in `_extract_frames` and the detector stays at 1280.** Battle 0 measured this end-to-end — see
`docs/BATTLE0_DETECTOR_WIDTH_FINDINGS.md`. Frame extraction currently sets
no quality flag, so higher resolutions silently got *more* compression;
quality-pinned 1280 gives 88.2% valid joints and 14 temporal rejections, beating
the 1920 lane (85.5%, 20) at 1/2.25 the pixels. Do not build a 3840 lane.

**Be clear about the ceiling.** MAMMA's Hi4D table shows the *detector*
dominates: every sparse-keypoint method clusters at 32–93 mm regardless of how
good its association is, and §9b shows why — the residual is systematic
label-convention bias, not precision. Battle 1 should move us from 60–80 mm
toward the ~40 mm where the best sparse pipelines sit. **It cannot reach 20 mm.**
Only Battle 4 — our own dense-landmark detector — can. Battle 1 is worth doing
because it is cheap and risks nothing; it is not a shortcut.

**Battle 2 — own the ground truth, and build the rig properly (one shoot).**
We currently have no legal reference. Capture our own: four calibrated cameras,
consented performers, a rented Xsens suit (~$300–1,000/week plus ~$315/month
software) or a hired Vicon session for a simultaneous marker reference. This
unblocks *everything* downstream, because right now every accuracy claim we make
is measured against an asset we cannot use commercially.

§9b sets the rig spec, and it is not optional — the physical terms are what make
20 mm reachable at four cameras:

- **Genlock, plus a sync check in every take** (LED flash or clap visible to all
  four cameras). MAMMA's genlocked rig still slipped two frames.
- **60 fps, fast-readout sensors**, 1/500–1/1000 s shutter, and enough light to
  support it.
- **Board + wand bundle-adjusted calibration per shoot**, validated by
  triangulating known lengths — not by reprojection RMS.
- **Cameras surrounding the volume**, not arranged in an arc.
- **Fitted wardrobe**, and a body scan of each performer.
- Record a marker reference simultaneously, and define the gate as **held-out
  marker error** so it is comparable with the 21.6 mm Vicon number.

**Battle 3 — the data factory (the real programme).**
MHR bodies + the free clean motion pool from §9a (CMU + Eyes JAPAN + ACCAD +
100STYLE, ~7,300 clips) + ContactPose grasps spliced in for hands + Poly Haven
HDRIs + purchased clothing/hair, rendered in Cycles on Modal. Target ~150k crops
for a first single-person model. This is where the money and the months go —
though notably **not** on motion data, which turned out to be free.

**Battle 4 — the detector.**
ViT-B + mask conditioning + 512 learnable landmark queries, Gaussian NLL and BCE
visibility. Reimplemented from the published papers by someone who has not read
MAMMA's source.

**Battle 5 — hands.**
Per-hand high-res re-crop pass, then IK onto our own hand rig. Set expectations
at ~10° finger angles. This closes the N5.1 blocker at acting quality, not at
contact quality.

**In parallel, not blocking: get an FTO opinion.** §8 explains why. It does not
gate the engineering, but it absolutely gates a launch and any funding decision
that assumes one.

### What to stop doing immediately

1. **Stop treating MAMMA as an internal reference.** §4.1. Replace it with our
   own captured ground truth (Battle 2). Until then, the honest framing is that
   we have no legal benchmark.
2. **Don't plan to build a parametric body model.** §8. MHR exists under
   Apache-2.0 with weights. Building our own buys nothing on either axis.
3. **Don't ship Ultralytics YOLO** (AGPL) or RTMW/DWPose checkpoints (tainted
   training data). The MMPose *code* is fine; the weights are not.

---

## 11. Corrections to existing docs

| Doc | Claim | Correction |
|---|---|---|
| `PIPELINE_MAP.md` | MAMMA's licence "blocks *shipping* its code/weights/outputs, not internal use" | Not defensible. The grant is non-commercial-purpose only, and bars "production of other artifacts for commercial purposes". See §4.1 |
| `COMMERCIAL_MULTIVIEW_BODY.md` P1 | "Evaluate… an audited RTMW/RTMPose whole-body checkpoint" | The code is Apache-2.0 but every high-keypoint checkpoint traces to COCO-WholeBody/Halpe, which are research-only. Plan to *train* on our own data, not to adopt a checkpoint. See §9 |
| `COMMERCIAL_MULTIVIEW_BODY.md` P3 | Two routes: license SMPL-X, or build an owned parametric body | A third route now dominates both: adopt MHR (Apache-2.0, weights included). See §8 |
| `COMMERCIAL_MULTIVIEW_BODY.md` P3 | "Do not train from MAMMA output or its restricted training assets" | Correct, and now specific: AMASS is the hidden gate, but it is routable around — its raw constituents are separately licensed and CMU + Eyes JAPAN + ACCAD + 100STYLE give ~7,300 clean clips free. See §9a |
| *this doc, earlier draft* | "Email both `smpl@max-planck-innovation.de` and `support@meshcapade.com`" | Meshcapade's platforms **shut down 18 April** after the Epic acquisition. Max-Planck-Innovation is the only remaining channel. See §8 |
| *this doc, earlier draft* | MoCap Online's AI Usage Permit named as a primary motion source | Its permit bars training "tools that would compete with professional motion capture services" — arguably our exact product. Demoted to do-not-plan-around. See §9a |
| all | Silent on patents | US10395411B2 (MPG, ~2036) and US9189886B2 (Brown, ~2032) are live and the second reads on the capture product regardless of body model. See §8 |

---

## 12. Open questions

Flagged honestly — these were not resolved in this pass and must not be assumed:

- **Freedom to operate.** The biggest one. Needs counsel, not more research.
- Whether CMU's "may not resell this data directly, even in converted form"
  extends to synthetic renders driven by those motions. The same sentence grants
  "include this data in commercially-sold products", so renders read as clearly
  inside the grant — but confirm, because the whole data plan rests on it.
- Whether MoCap Online's AI permit would actually be granted to us. Its
  restriction on training "tools that would compete with professional motion
  capture services" arguably describes our product exactly. Not blocking — §9a's
  free pool already covers the volume — but do not plan around it.
- Eyes JAPAN's licence page contradicts itself (CC BY 2.1 JP vs BY-SA 2.1 JP).
  Assume share-alike until clarified.
- HDM05's live licence (host 403s; archived snapshot says CC BY-SA 3.0), and
  whether KIT will grant commercial terms on its 4,600 unlicensed motions.
- CLO3D / Marvelous Designer output-ownership terms.
- Daz AI-program pricing (negotiated, not public).
- Truebones' exact ML-training terms — their two-sentence royalty-free grant
  doesn't mention ML.
- DWPose's exact training pipeline.
- Patent landscape on dense-landmark methods specifically — not searched.
- Whether MHR's neural pose correctives fall inside US10395411B2 claim 1. This
  is a claim-construction question, and it is the one that decides whether §8's
  good news survives contact with a lawyer.

