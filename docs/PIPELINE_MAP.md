# AutoAnim pipeline map

Status: orientation map, 2026-08-25. This document does not add capability and
does not supersede any other doc. It answers one question: what lanes exist
today, where each one runs, what state it is in, and where they converge.

Every per-lane detail lives in the linked doc. Claims below are marked
`[code]` when verified against the repository in this pass and `[doc]` when
taken from a lane document without re-verification.

## The one-line picture

```text
FACE LANES  (unified)          BODY LANES  (not unified)
audio ─┐                       4-cam video ─ MAMMA ──────┐
photo ─┼─ autoanim CLI ─────┐  1-cam video ─ GEM-X/SOMA ─┼─ body track:
video ─┘   ApplicationService│  audio+text ─ GestureLSM ──┤   autoanim.body-track/1.0 (25-joint)
multiview ─────────────────┐│  4-cam video ─ clean-room ─┘   autoanim.body-track/1.1 (55-joint)
                           ▼▼                             │
                    artifacts/jobs/<job-id>               │
                    production-readiness report           │
                           │                              │
                           └──── body_compositor / unified_gltf ────┐
                                                                    ▼
                                          promoted character revision (N5.1)
                                          connected 55-joint GLB
```

The face side is one product. The body side is four research lanes that share
an output contract and nothing else. That asymmetry is the mess.

## Hardware reality

- **MacBook (Apple Silicon)** owns everything except learned body inference:
  request preparation, GNM evaluation, face pipelines, importers, retarget,
  projection, composition, review, export. `[code]`
- **NVIDIA work runs on Modal**, never locally. All three GPU workers declare
  the same fallback order `["L40S", "A100-40GB"]`. `[code:
  workers/*/modal_app.py]`
- The GestureLSM doc still describes a local RTX 3090/3070 split; the committed
  worker is the Modal app. Treat Modal as the actual execution surface. `[doc vs code]`
- Modal input is always a sealed, hash-bound request directory; outputs are
  downloaded, hashed, and converted locally. `[code]`

## Face lanes — unified, go through the CLI

All four dispatch through `src/autoanim_gnm/cli.py` into
`ApplicationService` (`service.py`), write into `artifacts/jobs/<job-id>`, and
are covered by the `autoanim.production-readiness/1.2` report. `[code]`

| Lane | Command | Core modules | Runs where | State |
|---|---|---|---|---|
| Audio → face | `autoanim audio IN --out OUT` | `a2f.py`, `a2f_v3_local.py`, `a2f_v3_postprocess.py`, `audio_pipeline.py`, `speech_alignment.py` | Mac local | Working prototype; the `performance` release gate can never pass on phone-span proxy diagnostics alone `[doc: PRODUCTION_READINESS]` |
| Photo → face | `autoanim image IN --out OUT` | `image_pipeline.py`, `fitting.py`, `identity_qualification.py` | Mac local | Conservative visible-geometry fit works; monocular ambiguity is a stated hard limit `[doc: COMPLETION_AUDIT]` |
| Photos → shared identity + texture | `autoanim multiview IN... --out OUT` | `multiview_pipeline.py`, `calibrated_retarget.py`, `camera_bundle.py`, `texture_baker.py` | Mac local | Calibrated core is synthetic-tested; no rights-cleared real calibrated subject exists `[doc]` |
| Video → face | `autoanim video IN --out OUT` | `video_pipeline.py`, `video_capture.py`, `visual_face_retarget.py`, `video_observation.py` | Mac local | Working experimental pipeline; source contact is still a landmark heuristic `[doc]` |

Supporting commands: `health`, `qualify-lipsync`, `character` (list / show /
promote / material-template / import-material / revoke), `direct`, `job`,
`inspect-identity-qualification`, `material`, `serve`. `[code]`

Google GNM is the sole owner of face, jaw, eyes, tongue, teeth and oral
contacts in every lane, including the body ones. `[doc, consistently]`

## Body lanes — four of them, no CLI, no shared job model

**None of these has an `autoanim` subcommand.** Each is driven by a script in
`scripts/`, an optional Modal worker in `workers/`, and its own directory under
`artifacts/`. This is the single biggest structural gap. `[code]`

| Lane | Input | Driver | GPU | Output | State |
|---|---|---|---|---|---|
| **MAMMA** (internal reference) | 4 calibrated cameras | `workers/mamma/modal_app.py` (5 stages: cap → masks → 2d → 3d → vis), `mamma_motion.py`, `mamma_compositor.py` | Modal L40S/A100 | 55-joint track, per person | Real 4-cam GPU smoke passed 2026-08-04, 2 people, frames 60–90, `artifacts/mamma/mamma-4cam-smoke-v3/` `[doc]`. Retained as the **internal** quality reference and best available capture; its licence blocks *shipping* its code/weights/outputs, not internal use `[doc: COMMERCIAL_MULTIVIEW_BODY]` |
| **GEM-X / SOMA** | single video | `workers/gem_x/modal_app.py`, `scripts/prepare_gem_x_request.py`, `soma_motion.py`, `body_projection.py` | Modal | SOMASKEL77 → 25-joint track | Qualification take `research-squat-640` retained; **hands collapse to wrists** — the N5.1 release blocker `[doc: N5_1]` |
| **GestureLSM** | audio + transcript + word alignment | `workers/gesturelsm/modal_app.py`, `scripts/build_gesturelsm_candidates.py`, `speech_motion*.py` | Modal (doc says RTX 3090) | SMPL-X55 → AutoAnim-55, 4 seeds | 4 corrected candidates pass automatic gates; review-only, no animator approval `[doc]` |
| **Clean-room multiview** | 4 calibrated cameras | `scripts/build_commercial_multiview_comparison.py`, `commercial_multiview.py`, `workers/commercial_multiview/apple_vision_pose.swift` | **Mac local**, Apple Vision | 55-joint track | Working sparse baseline; no dense shape, no articulated fingers; the only commercially clean body path `[doc]` |

The one thing they do share: the body-track contract declared in `body.py` —
`autoanim.body-track/1.0` (25 core joints, the legacy preview contract GEM-X/SOMA
emits) and `autoanim.body-track/1.1` (55 detailed joints with articulated
fingers, used by MAMMA and GestureLSM). Both are validated against a matching
skeleton schema pair. `[code: body.py:23-25, 566-572]` The
`*.autoanim-body-track.json` filename convention is MAMMA's only `[code]`; the
other lanes' file naming was not verified in this pass. That schema pair is the
asset worth protecting through any reorganization.

## Where face and body meet

- `body_binding.py` — attaches a GNM head to a body revision at the neck seam.
- `body_compositor.py` / `mamma_compositor.py` — enforce the ownership split
  (GNM owns face, body lane owns body and fingers).
- `unified_gltf.py` / `body_export.py` — connected 55-joint GLB.
- `body_fidelity.py`, `production_readiness.py` — gates.
- `docs/N5_1_PRODUCTION_CHARACTER_ASSEMBLY.md` — the promoted-character-revision
  contract: identity, proportions, rig, seam, materials, motion ownership and
  evidence must travel together as one immutable revision.

**The forward path is N5.1.** Everything else is a feeder. The named blocker is
hand fidelity: SOMASKEL77 has articulated fingers and the retained take moves
them up to ~26°, MPFB has 15 deforming finger bones per hand, but
`project_soma_to_body_track()` projects only wrists and
`blender_body_worker.py` collapses every finger weight to its wrist. `[doc: N5_1]`

## What "one place" would actually mean

Four separate consolidations, roughly in dependency order:

1. **Body lanes get CLI subcommands and real jobs.** `autoanim body <lane>`
   writing to `artifacts/jobs/<job-id>` like the face lanes do, instead of
   bespoke scripts writing to bespoke directories. Highest structural payoff.
2. **One body provider interface.** `body_provider.py` and
   `nvidia_body_provider.py` already gesture at this; make all four lanes
   implement it so the choice of lane is a parameter, not a different script.
3. **One take object** binding a face job + a body job + a character revision,
   so production-readiness covers the whole shot rather than the face take.
4. **Pick the production body lane.** Clean-room multiview is the only
   commercially clean one but is the weakest. MAMMA is the quality reference
   that cannot ship. GestureLSM needs no camera at all. This is a product
   decision, not an engineering one — it is not made here.

## Decisions taken (2026-08-25)

- **MAMMA stays in the set.** It is an internal capture/test tool, not a
  shipped component. The non-commercial licence constrains *distribution of its
  code, weights and outputs*, not internal use. Keep the clean-room lane as the
  commercial answer; keep MAMMA as the internal quality reference and the best
  available body capture.
- **Modal is the execution surface for now**; the in-house Windows RTX 3090
  machines are a later local test target. Modal rents no consumer GPUs
  (catalog: `T4, L4, A10, L40S, A100-40GB/80GB, RTX-PRO-6000, H100, H200,
  B200, B300`), so the 3090s are a *second, self-hosted* surface rather than a
  Modal option. Feasibility per lane:

  | Lane | Local runner exists? | 24 GB verdict |
  |---|---|---|
  | GestureLSM | yes — `workers/gesturelsm/infer.py`, "pinned end-to-end worker orchestration for an RTX CUDA host" `[code]` | designed for exactly this; the README picks the 3090 over the 3070 for its 24 GB `[doc]` |
  | GEM-X/SOMA | yes — `workers/gem_x/infer_cuda.py` `[code]` | single person, ONNX-tensor boundary; comfortable |
  | MAMMA | no — Modal only, `convert_tracks.py` is a post-download helper `[code]` | doubtful: pins a 48 GB L40S first with A100-40GB fallback, and runs YOLO + SAM2 + MammaNet + multiview SMPL-X fitting. Ampere is supported by its CUDA 12.4 / torch 2.5.1 pin; VRAM is the risk, not the architecture |

  All three workers are built on `nvidia/cuda:*-ubuntu22.04` images and the
  GestureLSM README states it "runs only on a Linux NVIDIA host" `[code/doc]`.
  On the Windows boxes that means **WSL2 with CUDA passthrough**, not native
  Windows — it keeps the pinned Linux wheels, POSIX paths and SAM2/YOLO builds
  intact. Because every worker already takes a sealed hash-bound request
  directory and returns hashed artifacts, adding a local surface is a runner
  swap, not a rewrite.
- **Target is production-quality realism**, body mocap plus face, composited.
  That makes N5.1 character assembly plus a stronger face-from-video observer
  the two live workstreams.

## Open decisions

- Do the body lanes get CLI subcommands and real jobs before or after the N5.1
  hand-fidelity blocker is cleared?
- Which face-from-video upgrade is qualified first (see below)?

## Face fidelity from video — the known upgrade path

The current video lane's weakness is recorded in `COMPLETION_AUDIT.md`: source
contact is an unlabeled landmark heuristic on sparse MediaPipe landmarks, and
41.97% of one-sided neutral residuals are clipped. Sparse landmarks are the
ceiling, not the retarget.

The research references already collected for this exact problem, in
`PRODUCTION_CAPTURE_EXECUTION_PLAN.md` and
`PRODUCTION_RESEARCH_UPDATE_2026-07-20.md`:

| Reference | What it gives the video lane |
|---|---|
| **FlowFace** — *3D Face Tracking from 2D Video through Iterative Dense UV*, CVPR 2024 | The closest match to "ultra-realistic video to face mocap": dense per-pixel UV correspondence instead of sparse landmarks, iteratively refined, temporally tracked |
| **DenseLandmarks** (arXiv 2204.02776) | Hundreds of dense, uncertainty-aware landmarks — the cheaper step up from MediaPipe |
| **SPECTRE** (CVPRW 2023) | Visual-speech-informed reconstruction; directly targets mouth/lip fidelity from video, which is where the current lane clips |
| **TEMPEH** (arXiv 2306.07437) | Direct 3D head inference from calibrated multiview — the multi-camera analogue |
| **MICA** + **DECA** | Metric identity and detail displacement; identity/wrinkle tier rather than motion |
| **AVFace** (CVPR 2023) | Audio-visual 4D reconstruction — fuses the audio and video lanes instead of choosing |

All are research-licensed. Same rule as MAMMA: fine as internal reference,
each needs an explicit disposition before shipping. `[doc]`

## SHELLS — the multiview identity lane's likely target (added 2026-08-25)

*SHELLS: Topologically Consistent Multi-view 3D Head Reconstruction via
Coarse-Guided Layered Surface Sampling.* Bolkart, Wang, Chandran (Google),
SIGGRAPH 2026. <https://arxiv.org/abs/2605.31283> ·
<https://syntec-research.github.io/SHELLS/> · doi:10.1145/3799902.3811201

This belongs to the **`autoanim multiview` identity lane**, not the video
performance lane. It is head *reconstruction*, not motion tracking.

### Why it matters here

The paper's output template has **exactly 17,821 vertices**, from "an internal
dataset with registered 3D head meshes (with 17,821 vertices each) of ≥2500
identities". AutoAnim's GNM adapter declares the identical count
(`gnm_adapter.py:14`, `joint_regressor [4, 17821]`). The paper never names GNM
or FLAME, so **"SHELLS outputs GNM topology" remains inference, not a confirmed
fact** — but the count and the shared Google provenance make it the strong
reading. If true, SHELLS produces meshes already in the topology AutoAnim
animates, removing the cross-topology registration step entirely.

It also directly supersedes TEMPEH, previously listed as our multiview
reference: "SHELLS outperforms TEMPEH (Bolkart et al., 2023) on synthetic data,
achieving a 29% (28%) lower median (mean) V2V error."

### Verified specifications

| Property | Value |
|---|---|
| Input | Calibrated multi-view images with full extrinsics, intrinsics **and lens distortion** |
| Views | 13-camera rig in the main experiments; "results are reasonable with as few as two views" |
| Resolution | rendered 1536×1024, downsampled to **384×256** for the network |
| Temporal | **per-frame only**; no temporal module. Applied frame-by-frame it yields "temporally smooth and expressive performance registrations" — consistency comes from the fixed template, not tracking |
| Inference | 0.08 s/mesh, **~2.4 GB** GPU memory (vs TEMPEH's 0.29 s / ~20 GB) |
| Training | ~2 weeks on one H100 80GB; ~20 GB vs ~65 GB for volumetric baselines |
| Training data | **fully synthetic**: 300,000 pairs, 2,064 identities, Blender Cycles, Wood et al. 2021 procedural style — built from the internal ≥2500-identity registered set. Not public |
| Release | **none announced** — no code, weights, data, or license statement. The arXiv paper is CC-BY |

### Consequences for AutoAnim

1. **Not adoptable as a dependency.** No weights, no code, internal training
   data. Same category as MAMMA but stricter: MAMMA at least ships checkpoints.
2. **It is a blueprint.** The architecture — DinoV2+LoRA features, projective
   sparse feature cloud, coarse mesh, layered sampling shells, shared
   transformer — is described in full, and the training data is *synthetic*
   and procedurally generated. Both are reproducible in principle.
3. **It validates the 17,821-vertex template as the registration target**,
   which is the topology AutoAnim already commits to.
4. **2.4 GB inference is trivially local.** A reimplementation would run on the
   RTX 3090s, or plausibly on the Mac. The cost is training, not inference.
5. **Training needs no cameras; inference needs a calibrated rig.** Keep these
   separate — they are different acquisition problems.

   *Training* used **zero real cameras**: 300,000 synthetic pairs rendered in
   Blender Cycles, "trained exclusively on synthetic data yet generalizes
   effectively to real-world captures". The scarce input is the **registered
   head-mesh corpus** rendered from — ≥2,500 registered identities at 17,821
   vertices, internal to Google. Acquiring registered scans, not cameras, is
   the hard part of reproducing this.

   *Inference* wants calibrated multi-view. 13 views is their rig and their
   synthetic view arrangement, but random camera dropout and mean-variance
   fusion were trained in on purpose, and "results are reasonable with as few
   as two views". Non-negotiable is **calibration quality** — extrinsics,
   intrinsics and lens distortion — not view count, and **not resolution**:
   the network consumes 384×256.

   This means a modest 3–6 camera calibrated rig would serve the `multiview`
   identity lane, MAMMA, and any SHELLS-style reconstructor simultaneously,
   and it also closes the `COMPLETION_AUDIT` gap that no rights-cleared
   calibrated real subject exists. That rig is the single highest-leverage
   physical investment on the roadmap.

### Can we build a SHELLS-equivalent ourselves?

Assessment 2026-08-25. Verdict: **yes, plausibly, and the usual blocker does
not apply to us** — but the hard part is appearance realism, not architecture.

**Why the training-data blocker is softer for AutoAnim than the paper implies.**
SHELLS needed ≥2,500 registered internal identities because that corpus *was*
their generative identity model. AutoAnim already has the released equivalent:

- `vertex_identity_basis [253, 17821, 3]` and `joint_identity_basis [253, 4, 3]`
  — a 253-dimensional identity space in the exact target topology `[code]`
- `expression_basis [383, 17821, 3]`, regionally structured `[code]`
- `gnm/shape/semantic_sampler.py` — a conditional VAE **identity sampler**
  conditioned on gender and four demographic categories, plus a 20-label
  expression sampler `[doc: RESEARCH.md]`
- `triangle_uvs`, `quad_uvs`, `vertex_groups [46, 17821]`, `mirror_indices` `[code]`

So we can sample unlimited identities and expressions and obtain ground-truth
vertices **by construction**, in the exact topology, with no scans and no
registration step. Unverified: whether the CVAE's identity diversity is
comparable to 2,500 real registered scans. That is cheap to measure and belongs
in stage 0.

**The real research risk is appearance, not geometry.** SHELLS's synthetic
pipeline follows Wood et al. 2021, which depended on a large artist asset
library — skin albedo, displacement, hair grooms, eyes, clothing, HDRI
lighting. GNM ships UVs but **no appearance library**. Frozen DINOv2 features
mitigate the synthetic-to-real domain gap (likely why the paper chose a
foundation-model backbone) but do not erase it. This is where the project
succeeds or fails.

**Compute.** Paper: ~2 weeks on one H100 80GB. At Modal's H100 rate of
`$0.001097/sec` ≈ `$3.95/hr`, 336 hours ≈ **$1,330 per full run** `[estimate]`,
and no one gets it right on the first run. But the paper also reports
*training* memory of ~20 GB (vs ~65 GB for volumetric baselines) — so a reduced
configuration may fit the 24 GB RTX 3090s already in-house, slowly. Verify
before relying on it.

**Staged plan, cheap gates first.**

| Stage | Work | Cost | Gate |
|---|---|---|---|
| **−1** | **Oracle-fit GNM to 10–20 real head scans. No training, no rendering.** | scan acquisition | **Is the best-possible GNM surface accurate enough on real heads?** If not, the coefficient-only path is dead and the target must become GNM + free-vertex residual |
| 0 | Sample identities from `semantic_sampler.py`, render, train a small regressor, measure V2V on held-out synthetic | days, ≈$0 | Can the architecture estimate GNM coefficients from multi-view renders? (estimation only — says nothing about the manifold) |
| 1 | Appearance pipeline — procedural or licensed skin/hair/eye/lighting assets | the real research investment | Does a model trained on it transfer to real photos at all? |
| 2 | Scaled training, layered-shell architecture | ~$1.3k/run on Modal H100, or slow local 3090 | Beat our current multiview fitter on held-out synthetic |
| 3 | Real-world evaluation | rig + consented subjects | Beat it on real calibrated capture |

Stage 3 needs the same 3–6 camera calibrated rig that the `multiview` identity
lane and MAMMA need. Third consumer of one investment.

**Licence checks before calling any of this commercially clean:**

- GNM is Apache-2.0 `[doc]`.
- **DINOv2 weights** were originally released CC-BY-NC-4.0 and later
  re-released under Apache-2.0 — confirm exactly which release any
  reimplementation consumes. `[unverified, must check]`
- Wood et al. 2021 is a *method*, not an asset; following it is fine. Any
  purchased or scraped appearance assets carry their own terms.
- SHELLS itself contributes no code or weights, so there is nothing of theirs
  to license — only the paper, which is CC-BY.

**Realistic scale:** on the order of 1–3 months to a working prototype for one
person, dominated by the appearance pipeline. This is a real project, not a
weekend, and not a moonshot.

**Revised after Sol's review** — see `docs/SHELLS_STAGE0_PLAN.md`. Two
corrections matter at this level: SHELLS predicts **free vertex positions**,
explicitly to bypass linear-3DMM expressiveness limits, so a coefficient-only
regressor is a better GNM *fitter*, not a SHELLS reproduction; and the
prerequisite question — whether the best-possible GNM surface is accurate
enough on real heads — needs real scans and no training at all, so it now runs
first as Stage −1.
