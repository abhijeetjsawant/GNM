# SHELLS in-house — Stage −1 and Stage 0

Status: plan, revised 2026-08-25 after Sol's review. No hard deadline; treat as
priority with steady checkpoints. Technical design by Fable, critique by Sol,
restructured in response. Context: `docs/PIPELINE_MAP.md`, section "Can we
build a SHELLS-equivalent ourselves?".

## Revision note — why this plan changed

The first draft made Stage 0 a synthetic estimator test: train a regressor on
GNM-sampled heads, beat the landmark fitter, proceed. Sol's review identified
three defects that would have wasted the week:

1. **Circularity is fatal, not merely a risk.** Identity-disjoint splits are
   not *generator*-disjoint splits. Tiers A, B and C all decode from the same
   64-dimensional CVAE, so held-out cells and blended labels still occupy the
   same learned shape manifold. A PASS would have proven only "this
   architecture can invert synthetic renders of its own low-dimensional
   generator". Rendering identities previously fitted from real photographs
   does not fix this — those meshes are exactly in GNM space too.
2. **PASS condition 3 was incoherent, and our own docs said so.** Modes
   170–252 are **not** a high-frequency shape band. Verified directly from the
   NPZ: `identity_names` is `head × 170, eyes × 3, teeth × 80`. The gate asked
   the network to recover eyeball and teeth geometry from renders containing no
   eye or teeth assets, with the mouth closed. Cosine > 0.2 there would have
   measured CVAE correlation and prior guessing, nothing more.
3. **The target was conflated.** SHELLS predicts **free vertex positions**, and
   explicitly presents that as bypassing the expressiveness limit of linear
   3DMMs. Fixed topology means fixed connectivity and semantic correspondence —
   it does **not** mean confinement to a 253+383-dimensional linear model. A
   coefficient-only regressor, however good, is *a better learned GNM fitter*.
   That may be worth having, but it is not SHELLS.

Point 3 implies a prerequisite question that no architecture can answer and no
amount of synthetic data can dodge, so it now comes first.

## The prerequisite question — Stage −1

> **Is the best-possible GNM surface accurate enough on real heads for our
> product?**

This is a pure representation-floor measurement. **No training, no rendering,
no network.** Fit GNM as well as it can possibly be fitted — directly to real
head geometry, with full knowledge of the ground truth — and measure what is
left over.

- If the floor is **acceptable**: a GNM coefficient estimator is a legitimate
  target, Stage 0 below is a fair test of it, and we describe the result
  honestly as a learned fitter rather than a SHELLS reproduction.
- If the floor is **not acceptable**: the coefficient-only path is dead on
  arrival, and the output must become GNM **plus a topology-consistent
  free-form residual**, or direct vertices — which in turn means the training
  corpus must contain geometry *off* the GNM manifold. That is a materially
  different and more expensive project, and it is better to learn it now than
  after a week of synthetic training.

### What Stage −1 needs

10–20 real head scans whose **own geometric accuracy is well below the error we
are measuring**. That is the binding constraint, not licensing: we expect a
floor of 1–3 mm, so a scan carrying ~1 mm of capture error cannot measure it.

**DIY capture is disqualified on accuracy**, which settles the cheapest option
early:

| Method | Measured accuracy | Verdict |
|---|---|---|
| iPhone 14 Pro LiDAR apps vs 3dMD ground truth | **1.46–1.66 mm** mean surface error ([study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11594577/)) | Error ≥ the floor being measured. Useless here |
| iPad LiDAR apps | 2.2–4.4 mm | Worse |
| Phone/DSLR photogrammetry + Metashape | 0.22 ± **1.29 mm** on craniofacial models | The spread means it is not reliably sub-mm on a living subject (breathing, micro-motion) |
| Synchronised multi-camera DSLR rig (30+) | ~0.1–0.3 mm | The only DIY class that qualifies — but building one is a project, not a shortcut |

**Sources, ranked by how fast meshes land on disk** (scope: internal
evaluation only):

| # | Source | Cost | Capture rig / accuracy | Access |
|---|---|---|---|---|
| 1 | [3DScanStore 12× HD Head Scans bundle](https://www.3dscanstore.com/discount-packs/12x-hd-head-scans-bundle) | **£270**, instant download | 120-DSLR photogrammetry rig, "sub pore level" detail, 16K textures, 34 GB. No published mm figure `[unverified]`; same capture class as FaceScape, realistically ~0.1–0.3 mm | None — buy it. **Demographically diverse**, which matters for a gender/ethnicity-conditioned basis |
| 2 | [NPHM](https://github.com/SimonGiebenhain/NPHM) (TUM, CVPR'23) | free | **Two Artec Eva structured-light scanners**, 0.1 mm spec point accuracy, ~0.2–0.5 mm merged. 5,200 scans, 255 identities, ~20 expressions, good hair coverage | Google Form + ToS. Best *verified* sub-mm option. Non-academic approval `[unverified]` |
| 3 | [FaceScape](https://nju-3dv.github.io/projects/FaceScape/) (Nanjing) | free | 68-DSLR rig, pore-level displaced geometry, topologically uniform. 847 subjects × 20 expressions | Signed PDF + email. Licence reads *"internal, non-commercial research, evaluation or testing purposes only"* — a literal match for Stage −1. But **Chinese-subject-dominant**, so weak demographic spread. Metric accuracy unpublished `[unverified]` |
| 4 | [FaceVerse-Dataset](https://github.com/LizhenWangT/FaceVerse-Dataset) | free | Dense DSLR rig; 128 identities × 21 expressions | GitHub agreement, low friction. Asian-skewed; accuracy unstated `[unverified]` |
| 5 | [FaceHD-100 / ImHead](https://arxiv.org/html/2510.10793) (2025) | free? | 9-DSLR + 8-LED, 8K, 100 subjects ages 16–70 | Access path `[unverified]` — a fallback if 1–3 stall |

**Ruled out:** [Headspace](https://www-users.york.ac.uk/~np7/research/Headspace/)
(1,519 subjects, 3dMD 5-pod — but access restricted to *"verifiable university
employees"*, closed without an academic partner); Multiface (tracked
reconstructions, no published geometric accuracy); BU-3DFE and FRGC (~2005
capture, fail sub-mm regardless of access); Basel Face Model (ships the model,
not raw scans). Among commercial stores, Renderpeople's licence bars computer-
vision research outright, and Triplegangers' terms bar use *"for artificial
intelligence or machine learning (AI/ML) research or study without a licence"* —
broad enough to arguably catch even internal evaluation, so it is the wrong
vendor to start with even though it sells single heads at $69.99.

**Recommended acquisition:** buy the 3DScanStore bundle for immediate,
demographically diverse, studio-grade meshes, and submit the NPHM and FaceScape
requests the same day. Twelve heads within the hour; potentially hundreds
within one to three weeks. Total tier-A cost about £270–550.

**When this later becomes training data**, none of the above carries over —
academic sets forbid it and 3DScanStore's clause bars distributing AI-derived
generators. That path is a negotiated ML licence (Triplegangers sells these on
request) or commissioned scans with an explicit ML-use release. Keep any such
corpus **physically separate** from the tier-A evaluation set so the two can
never be confused.

Follow the existing fixture discipline from `docs/TEST_FIXTURES.md`: pinned
revision, verified SHA-256, source attribution, recorded terms, opt-in
download, and never checked into the repository.

### What Stage −1 measures

For each scan, solve for the GNM identity (and expression) coefficients that
minimise surface-to-surface distance against the raw scan — an *oracle* fit
with no image observation involved. Then report, per subject and aggregated:

- **Representation floor:** oracle GNM fit vs raw scan — median, p90, p95.
- Per-region breakdown via `vertex_groups`: nose, lips, eyes, ears, cranium,
  jawline. Ears and cranium are where linear head models usually fail.
- Visible-skin-only error alongside full-mesh error.

Then, and only then, does the estimation question below become meaningful,
because it separates cleanly into **representation floor** (oracle fit vs scan)
and **estimation gap** (prediction vs oracle fit).

### If real scans are genuinely unobtainable first

Weaker but non-zero: synthesise off-manifold geometry by adding smooth,
topology-preserving deformations **projected orthogonally to the complete GNM
identity and expression bases**, so the target provably cannot be represented
by any coefficient vector. This is an anti-circularity stress test, not a
substitute for real geometry, and must be labelled as such in any result.

---

# Stage 0 — the estimator test (conditional on Stage −1)

Run this **only if the Stage −1 floor is acceptable.** The question then is
narrow and fair:

Can a small multi-view regressor trained purely on GNM-sampled synthetic heads
recover held-out identities better than this repo's landmark-based fitter — and
does it extract information the 68 landmarks structurally cannot?

## Prior art check: `edualvarado/gnm-webcam-puppet`

Reviewed before planning. Apache-2.0. Browser-side MediaPipe (478 landmarks) →
least-squares GNM identity fit → per-frame expression and pose solve → WebGL
render. **No training of any kind; pure fitting and optimization.**

This is the same *class* of method AutoAnim already implements in
`multiview.MultiViewIdentityFitter` and the video lane — it is essentially the
Stage 0 baseline, shipped as a web demo. It does not overlap with the SHELLS
hypothesis, which is precisely about exceeding the landmark ceiling. Its real
value to us: it is permissively licensed prior art confirming the
landmark-fitting ceiling is well understood, and a possible reference for a
real-time browser preview lane later. It changes nothing about this plan.

## Verified foundations

Facts confirmed by reading, not assumed:

- `gnm/shape/data/semantic_sampler/` ships `identity_decoder_model.h5` (1.26 MB)
  and `expression_decoder_model.h5` (1.53 MB).
- `semantic_sampler.py`: `IdentitySampler.sample_identity(gender_class,
  ethnicity_class, num_samples, rng)` → `[N, identity_dim]`;
  `randomize_identities` blends one-hot labels;
  `ExpressionSampler.sample_expression(...)` and `randomize_expressions(...)`.
  All take a seeded `np.random.Generator` — fully deterministic. Requires
  TensorFlow.
- Mesh units are **meters**: template bbox x ∈ [-0.128, 0.128],
  y ∈ [0.066, 0.407] (neck to crown). Interocular ≈ 0.06 m, so 1 mm = 1e-3.
- `GNMAdapter.mesh(identity, expression, rotations, translation)` returns a
  validated `[17821, 3]` float32; `.landmarks(...)` returns HEAD_SPARSE_68.
- **`MultiViewIdentityFitter` fits only 68 sparse landmarks and only
  `compact_identity_basis[:170]`** — it asserts the basis tail is zero at the
  landmarks. That 68-point / 170-mode ceiling is the thing this experiment
  tries to beat.
- `run_multiview_pipeline` requires real photos through `FaceExtractor` and
  **cannot run on synthetic renders**. The baseline must call
  `MultiViewIdentityFitter.fit()` directly with oracle landmarks.
- `camera_bundle.CalibratedCameraView` holds `intrinsics_matrix [3,3]`,
  `distortion [5]`, `world_to_camera [4,4]`, `image_size`, per-landmark
  `visibility [68]`. `texture_baker._project` is standard pinhole. Synthetic
  bundles emit in exactly this format, drop-in compatible.
- `render.MeshRenderer` is weak-perspective CPU preview — QA only, **not** the
  dataset renderer.

## 1. Synthetic corpus

**Sampling** (one-time, TensorFlow isolated to this script so the render/train
stack never imports TF):

- **3,000 identities** = 2,500 train / 250 val / 250 test.
- Sample per (gender × ethnicity) cell with a fixed seed, but **hold out two
  cells entirely** for the hard test tier.
- Plus 250 `randomize_identities` blended samples reserved for tier C.
- **8 frames per identity**: 1 neutral + 4 `sample_expression` draws + 3
  `randomize_expressions` blends. Random head pose per frame: yaw ±35°,
  pitch ±20°, roll ±10°, driving neck/head only, eyes neutral.
- Cache `identities.npz [3000,253]`, `expressions.npz [3000,8,383]`, and a
  split manifest recording every seed.

**Camera rig.** 6 virtual views — SHELLS uses 13 but reports two are workable;
6 gives coverage without fetishising their rig. Ring at 1.0 m from the head
centroid, yaws {-90, -45, -10, +10, +45, +90}°, per-identity jitter ±3 cm and
±2° look-at. Zero distortion for Stage 0 — distortion is a calibration-realism
problem, not a shape-learnability one. Emit per-identity bundles readable by
`camera_bundle.load_camera_bundle`.

**Corrected framing arithmetic** (the first draft's numbers were wrong). The
template head bbox spans y ∈ [0.066, 0.407] = **0.341 m**. At 1.0 m with
f = 700 px it projects to 700 × 0.341 / 1.0 ≈ **239 px** — 93% of a 256 px
frame, which clips under pose and jitter. Either use **f ≈ 525 px at 1.0 m** or
keep f = 700 px and move the ring to **1.33 m**; verify the worst-case posed
bounding box stays inside the frame with margin before generating data.

**Resolution must respect the patch grid.** 384×256 is not divisible by
DINOv2's 14-px patch size (384/14 = 27.43). Render at **392×266** (28×19
patches exactly) and record the true intrinsics for that raster. Define crop
and pad conventions explicitly and update K accordingly.

**Renderer: not Blender Cycles.** Cycles buys global illumination and hair; we
have no hair assets and GI is irrelevant to whether geometry is learnable. Use
**nvdiffrast** on the 3090s (~1 ms/frame), with pyrender/trimesh as a CPU
fallback for Mac smoke tests. Verify projections match `texture_baker._project`
to sub-pixel before generating anything.

**Appearance — deliberately cheap.** GNM ships UVs but no albedo, hair, or eye
assets. Three randomized layers per (identity, frame):

1. Procedural UV albedo: skin-tone base × low-frequency noise; darker lips and
   brows painted via `vertex_groups` masks.
   `gnm/shape/visualization/vertex_colors.get_vertex_colors` is a zero-effort
   variant. **No curvature-baked AO in the albedo** — baking curvature into the
   texture leaks the geometry label directly into the input, and the network
   would learn to read the answer off the skin. Any AO must be randomized
   independently of the target geometry, or omitted.
2. Randomized lighting: 2–3 directional lights plus ambient, Lambert + Blinn.
3. Random backgrounds, including head-coloured ones.

**What this proves and does not prove.** It proves the regressor can read shape
from shading, silhouette and multi-view parallax — the geometry-supervision
half of the SHELLS recipe. It proves **nothing** about transfer to real
photographs: real skin BRDF, hair occluding scalp and ears, eyelashes,
specular eyes, sensor noise. Hair is the largest gap — SHELLS's real scans
include hair volumes; our renders show bare scalps. Cheap mitigation: composite
random occluder blobs over the scalp in ~30% of frames so the network cannot
rely on always seeing the cranium. Real-photo transfer is Stage 1's question.

**Budget:** 3,000 × 8 × 6 = **144k images** at 384×256. Under 2 h on one 3090,
about 30 GB as PNG.

## 2. The regressor

**Coefficient regression (253 + 383 = 636 outputs), not free vertices** —
against 53,463 free coordinates, ground-truth coefficients are known exactly by
construction, and free-vertex errors leave the GNM manifold, muddying
comparison with the coefficient-space baseline.

**Critical: the loss lives in vertex space**, not coefficient space:
`L = mean_v || B_id·(β̂−β) + B_exp·(ε̂−ε) ||₂`, on a 2k-vertex subsample per
step, full 17,821 at eval, plus small coefficient L2. Plain coefficient MSE
mis-weights PCA modes and would understate perceptible error.

**Architecture, one 3090** (revised after Sol's review — the first draft's
choices risked a **false FAIL**, which is the worst outcome for a
kill-the-idea experiment):

- **Frozen DINOv2 ViT-B/14 with LoRA adapters and multi-layer features**, not
  frozen ViT-S. The paper ablates this: removing LoRA costs ~20% synthetic V2V.
  A weaker backbone that fails tells us nothing about the hypothesis.
- **Projective feature sampling in v0, not as a contingency.** Ray-map tokens
  are not equivalent: they force the transformer to learn correspondence and
  triangulation implicitly. Ray maps built from K and R also omit camera
  *centres*, so metric geometry is not even recoverable. Predict coarse
  coefficients, pose the mesh, sample features at projected vertex locations,
  refine — the SHELLS-like iteration, and the actual mechanism under test.
- Cross-view fusion: 4-layer transformer, dim 384, 6 heads, plus a learnable
  `[SHAPE]` token → MLP head.
- **Stage 0 is identity-only.** Regressing identity and expression jointly
  against only their *summed* vertex displacement permits identity–expression
  cancellation — the two heads can trade error invisibly. Either fix
  expression to neutral for Stage 0, or add an explicit neutral-identity loss
  plus cross-expression identity consistency (the same subject under different
  expressions must decode to the same β).
- AdamW, batch 32, ~60k steps, **random view dropout (train on 2–6 views)** so
  the 2-view evaluation is in-distribution.

**Output head — state the target honestly.** Coefficient regression is the
cheap Stage 0 instrument, and it tests *estimation*, not *representation*. If
Stage −1 shows the GNM floor is the binding constraint, the head must become
coefficients **plus a topology-consistent free-vertex residual**, because that
is what SHELLS actually predicts.

## 3. Splits — identity generalization is the real test

Split by **identity**, never by view or frame. Three tiers, reported separately:

- **Tier A (easy):** 250 held-out samples from the six *seen* cells.
- **Tier B (harder):** everything from the two *entirely held-out* conditioning
  cells.
- **Tier C (hardest):** 250 blended-label identities, off the training simplex.

Every tier evaluated at both 6 views and 2 views (frontal ±45°).

## 4. Metrics, baseline, and the PASS/FAIL call

**Metrics**, in millimetres (model units × 1000):

- Primary: **V2V** over all 17,821 vertices of the *neutral, unposed*
  reconstruction (decode β̂ with ε=0) — isolates identity. Report unaligned;
  Procrustes-aligned as diagnostic.
- **Median alone is not enough** — it hides catastrophic subjects. Report
  subject-level **p90 and p95** alongside the median.
- **Visible-skin error** separately from full-mesh error, and per-region via
  `vertex_groups`: nose, lips, eyes, ears, cranium, jawline, teeth.
- **Variance explained / V2V-energy reduction over a zero-prior predictor**
  (i.e. always predicting the mean face), computed per region and **conditioned
  on actual visibility**. This replaces the discarded cosine gate.

**Baseline:** `MultiViewIdentityFitter(adapter).fit(...)` fed **oracle
landmarks** — ground-truth HEAD_SPARSE_68 projected through the synthetic
cameras with σ = 1.0 px noise, true intrinsics, stages up to 170. Deliberately
optimistic: real pipelines detect landmarks rather than being handed them.
**Measure it on day 2, before any training, and freeze thresholds against the
measured number.**

### The discarded gate, and why

The first draft required *coefficient cosine > 0.2 in modes 170–253*, on the
theory that these were shape modes the 68 landmarks cannot observe. Verified
against the NPZ, that is false:

```
identity_names = head × 170, eyes × 3, teeth × 80
index 170.. = eyes_000, eyes_001, eyes_002, teeth_000 ... teeth_079
```

Modes 170–252 are **eyeball and teeth** identity coefficients. Our renders
contain no eye or teeth assets and the mouth is closed, so any cosine there
would measure CVAE correlation and prior guessing, not evidence extracted from
images. The gate is removed. Cosine also ignores coefficient magnitude, which
made it a poor instrument regardless.

**Where the real signal is:** dense improvement *within* the 0–170 head modes.
Landmarks technically touch those modes but observe them at only 68 points;
dense multi-view imagery should recover them far better. That is the honest
version of the hypothesis, measured by variance-explained per visible region
rather than by a coefficient-band cosine.

**PASS** requires:

1. Median V2V on tiers A **and** B beats the oracle baseline by a margin set
   against the day-2 measurement — **not** the first draft's arbitrary 25%. The
   baseline is structurally capped at 170 modes while the regressor sees dense
   pixels, so a relative win is close to designed-in; the margin must be large
   enough to be meaningful and should be chosen once the baseline number exists.
2. **Absolute** tier-B error substantially below the floor measured in
   Stage −1. Note the first draft's 1.5 mm was borrowed from SHELLS and is not
   comparable: SHELLS reports 1.22 mm face-only / 1.59 mm full-mesh median on
   *synthetic* data and 1.50 mm against registrations on *real* capture, where
   the registrations are themselves imperfect. A clean, exactly-representable
   GNM-on-GNM problem should demand **substantially lower** error than that.
3. p90/p95 within a stated multiple of the median — no catastrophic subjects.
4. Variance-explained over the zero-prior predictor, per visible region,
   clearly above the baseline's.

**FAIL:** cannot beat an oracle 68-landmark fit on dense error, or tier B
degrades more than 2× versus tier A (memorised the sampler's cells rather than
learning shape from images).

**A PASS still means only this:** the architecture can estimate GNM
coefficients from multi-view renders of GNM heads. Generalisation to real
photographs is untested and untestable here — that is Stage 1. Generalisation
to geometry *off* the GNM manifold is answered by Stage −1, not by this
experiment.

## 5. Using the 3090 fleet

- **Rendering: embarrassingly parallel.** Shard 3,000 identities across every
  box via `render_worker.py --shard k/N`. No inter-node communication; rsync
  outputs to the training box. **This is where the fleet earns its keep.**
- **DINOv2 feature pre-extraction:** also shard-parallel.
- **Training: one 3090 is enough.** 15 M trainable params with a frozen
  backbone — DDP is not worth the WSL2 NCCL pain. Use the other boxes for a
  **hyperparameter sweep, one config per box**, not data parallelism.
- **Baseline fitting:** CPU-bound SciPy; multiprocess over test identities.
- **Modal:** fallback only — if nvdiffrast fails under WSL2, render on A10Gs;
  if the feature cache forces on-the-fly DINO, one A100.

## 6. Schedule and files

Everything under `experiments/stage0_gnm_corpus/` — a spike, deliberately
outside the shipped `src/autoanim_gnm/` package.

**Stage −1 comes first and is not on this clock** — it is gated on acquiring
10–20 rights-cleared head scans, which is an acquisition task, not a coding
task. Days below are Stage 0, and only start once Stage −1 has passed.

| Day | Work | Files |
|---|---|---|
| 1 | Sampling and camera bundles; verify nvdiffrast on WSL2; verify projection matches `texture_baker._project` to < 0.5 px | `sample_corpus.py`, `synthetic_rig.py` |
| 2 | Renderer plus 50-identity pilot; **and the baseline, measured** — it calibrates the thresholds before any training exists | `render_worker.py`, `appearance.py`, `baseline_oracle_fit.py`, `metrics.py` |
| 3 | Full 144k render across the fleet; sharded feature extraction | `extract_dino.py`, `dataset.py` |
| 4 | Model and training; overfit 10 identities as a sanity check, then a full overnight run | `model.py`, `train.py` |
| 5 | Evaluate three tiers × {2, 6} views, baseline table, QA renders of worst cases; write the verdict | `evaluate.py`, `RESULTS.md` |
| 6 | Contingency: free-vertex residual ablation if ambiguous | — |

## 7. Threats to validity

1. **Same-generative-model circularity — now handled structurally, not by
   mitigation.** Train and test identities decode from the same 64-dimensional
   CVAE. Identity-disjoint splits are not generator-disjoint splits, and tiers
   B and C do not escape the manifold. The first draft's "decisive probe" —
   rendering coefficients previously fitted from real photographs — **does not
   work**: those meshes are exactly in GNM space and, having come from the
   170-mode fitter, barely exercise the rest. This is why Stage −1 exists and
   why it runs first. Stage 0 no longer claims to answer the manifold question
   at all.
2. **False FAIL is the expensive failure mode.** For a kill-the-idea
   experiment, an under-powered model that fails for architectural reasons is
   worse than no experiment — it kills a good idea. Hence ViT-B + LoRA and
   projective sampling in v0 rather than as contingencies.
3. **Appearance shortcut.** With few albedos the network may regress shape from
   silhouette alone, and a bald GNM head's silhouette is unrealistically clean.
   Mitigations: head-coloured backgrounds, scalp occluders, and a
   silhouette-only ablation number. Stated honestly: scalp blobs do **not**
   reproduce hair statistics or semantic occlusion, and the absence of eye and
   teeth assets bounds what any eye/mouth-region metric can mean.
4. **Geometry-label leakage through appearance.** Any texture derived from the
   target geometry — baked curvature AO being the obvious case — hands the
   network the answer. Appearance must be randomized independently of shape.
5. **Identity–expression cancellation.** Supervising only the summed
   displacement lets the two heads trade error invisibly. Stage 0 is
   identity-only, or carries explicit neutral-identity and cross-expression
   consistency losses.
6. **Unverified, must check on day 1:** nvdiffrast under WSL2 CUDA; exact
   `vertex_group_names` for lip/brow masks; worst-case posed bounding box
   inside the corrected frame; disk budget for the feature cache (larger with
   ViT-B — may force on-the-fly extraction).

## Reuse discipline

**Reused:** `GNMAdapter`, `semantic_sampler`, `camera_bundle`,
`MultiViewIdentityFitter` (as baseline), `MeshRenderer` (QA only),
`texture_baker._project` (projection reference).

**Deliberately not used:** `run_multiview_pipeline` (needs real-photo face
detection), `calibrated_retarget.py` (out of scope), Blender Cycles (deferred
to Stage 1 realism work).

---

# Costing the stages after the scans arrive

Estimates, 2026-08-25. Dollar costs are small; **time and the appearance asset
library dominate**. Everything below assumes the 4× RTX 3090 fleet is
available, since that is what keeps the cash cost near zero.

## Stage −1 — representation floor

| Item | Cost |
|---|---|
| Scans | £270 (3DScanStore bundle), free if NPHM/FaceScape land |
| Compute | **≈ $0** — surface-to-surface fitting is CPU/SciPy, runs on the Mac |
| Time | **2–4 days** to write the oracle fitter and produce the report |

The oracle fitter is the only real work: non-rigid GNM fit to an arbitrary scan
mesh, plus per-region error reporting. No GPU, no rendering, no network.

## Stage 0 — estimator test

| Item | Cost |
|---|---|
| Rendering 144k images at 392×266 | rasteriser, minutes to ~2 h on one 3090 |
| Training ~15–30 M params, frozen backbone | one 3090, overnight |
| Cash | **≈ $0** local; ~$50–150 if pushed to Modal |
| Time | **5–6 days** |

## Stage 1 — the appearance pipeline (the real investment)

This is where the money and most of the risk live, because GNM ships UVs and
nothing else.

| Item | Realistic cost |
|---|---|
| **Skin albedo / displacement** | Largely **already paid for** — the 3DScanStore bundle ships 16K textures and displacement for 12 diverse heads. A Texturing.xyz-style micro-detail pack adds ~$100–400 if needed |
| **Hair grooms** | The genuinely hard asset. Purchased groom libraries run ~$50–300 each; a usable randomisable set is ~$500–1,500, or weeks of procedural Blender work instead |
| **Eyes** | ~$50–150, or free-with-effort |
| **HDRI lighting** | free (Poly Haven) |
| **Assets subtotal** | **~$700–2,000** |
| **Engineering time** | **3–6 weeks.** Procedural material graph, randomisation ranges, hair placement and simulation, a Cycles render harness, and the domain-randomisation sweep. This is the schedule driver, not the cash |

## Stage 2 — scaled training

Two costs people forget: **rendering the corpus is bigger than training it**,
and the corpus size is uncertain.

**The uncertainty, stated plainly.** The paper reports "300,000 data pairs".
If a *pair* is one multi-view **set**, then at 6 views that is **1.8 M renders**
— not 300 k. I could not resolve this from the paper `[unverified]`, and it
swings the estimate by roughly 6×.

| Scenario | Cycles renders | On 4× 3090 @ ~5 s/frame | On Modal A10 @ $1.10/hr |
|---|---|---|---|
| 300 k images | 300 k | **~4 days** | ~$460 |
| 300 k sets × 6 views | 1.8 M | **~3–4 weeks** | ~$2,750 |

Mitigations if it lands at the high end: fewer identities, fewer frames per
identity, lower sample counts with denoising, or an Eevee/raster hybrid for
everything except the hair passes.

**Training itself.** The 3090-vs-H100 gap is larger than a first estimate
suggests. Dense BF16/FP16 tensor throughput is ~35.6 TFLOPS on a 3090 against
~495 TFLOPS on an H100 SXM5 (~14×), memory bandwidth 936 GB/s against
3,350 GB/s (~3.6×), and Ampere has no FP8 path. Real transformer training sits
between compute- and bandwidth-bound, so **5–10× slower end to end** is the
honest figure `[estimate]`. The paper's ~20 GB training footprint does fit in
24 GB, but the tighter batch forces gradient accumulation, costing more
efficiency again.

| Route | Full paper config | Cash |
|---|---|---|
| Modal H100 80GB | ~2 weeks | **~$1,330 per run** at $0.001097/sec |
| **1× RTX 3090** | **~10–20 weeks** — impractical | electricity only |
| **4× RTX 3090 in one chassis, DDP over PCIe** | **~3–6 weeks** (no NVLink, so ~3.2–3.5× scaling, not 4×) | electricity only |
| **4 separate machines, one 3090 each** | **See the caveat below — probably not viable for DDP** | electricity only |

**Topology caveat, and it matters.** The GPUs are described as living in
*separate Windows computers*, not one chassis. Multi-**node** DDP synchronises
gradients over the network every step. At ~30 M trainable parameters that is
~120 MB of all-reduce traffic per step; over 1 GbE that is on the order of a
second, against a step time of a few hundred milliseconds — communication would
dominate and four machines could end up **slower than one**. Multi-node DDP is
only worth attempting over 10 GbE or better, ideally RDMA.

Practical consequence: treat the fleet as **four independent workers**, not one
cluster. Rendering shards perfectly across machines with zero communication,
and hyperparameter sweeps run one config per box. Neither needs a fast
interconnect. Reserve genuine data-parallel training for a single multi-GPU
box, if one exists, or for Modal.

**Therefore: do not run full scale on the 3090s.** Run a *reduced* config there
first — fewer identities, fewer frames each, smaller model, fewer steps. About
a quarter of the compute is **4–8 days on 4× 3090**, and it answers whether the
recipe learns at all. Then buy one full-scale H100 run once the recipe is
known.

That splits the work by each machine's strength: the 3090s do **rendering**
(embarrassingly parallel, and expensive on Modal), reduced-scale runs, and
hyperparameter sweeps one config per box; Modal H100 does the one or two
full-scale runs that genuinely need the throughput. If the fleet has already
narrowed the recipe, budget **1–2 full runs ($1,300–2,700)** rather than 3–5.

**Electricity, for completeness:** 4× 3090 ≈ 1.4 kW. Three weeks continuous ≈
700 kWh — tens of euros, not a real line item.

**Nobody gets it right in one run.** With the fleet used for reduced-scale
iteration first, budget 1–2 full Modal runs (**$1,300–2,700**) rather than the
3–5 ($4,000–6,650) a cloud-only approach would need.

## Summary

| Stage | Cash | Wall-clock |
|---|---|---|
| −1 | £270 (scans) | 2–4 days |
| 0 | ~$0 | 5–6 days |
| 1 | ~$700–2,000 (assets) | 3–6 weeks |
| 2 | $0 local (reduced) / $1,330 per full run on Modal | 4–8 days reduced; 2 weeks–6 weeks full |
| **Total to a trained prototype** | **~$1,000–2,500 local fleet; ~$6,000–9,000 all-Modal** | **2–4 months** |

The local fleet turns this from a five-figure cloud project into a roughly
£1,000 hardware-already-owned project. The cost that does not shrink is
**engineering time on the appearance pipeline**, and that is also the part most
likely to decide whether the result transfers to real photographs at all.
