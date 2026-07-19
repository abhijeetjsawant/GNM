# Production facial performance and identity execution plan

Status: active implementation plan

Date: 2026-07-19

GNM revision: `3de70dfca5f3244620f44103c24b7cedc0dcb8b6`

AutoAnim baseline: `f16a2656dea7266d68828ed602d14587509f8b57`

## Implementation ledger

- **E0 phone-evidence slice — implemented, reviewed, and regression-green.** The app
  imports bounded long-format Praat/MFA TextGrid phone, word and optional apex
  tiers on the 48 kHz project clock; binds them to exact audio/TextGrid hashes;
  retains deterministic event and bilabial-timing reports; exposes the lane in
  CLI, HTTP and UI; snapshots the annotation as an immutable job attachment;
  and reconstructs the event/timing reports from that TextGrid and the final
  sealed GNM controls during release review.
  The retained LibriSpeech A/B now uses 85 genuine phone intervals from the
  published MFA alignment, not invented labels, and produced byte-identical
  motion arrays with and without evidence. That alignment is automatic, has no
  reviewed apex tier, and is never represented as independent review. The
  current report is deliberately incapable of production approval because
  human-reviewed apex data plus labiodental, tongue and false-contact
  evaluation remain unavailable. Final verification: 467 tests passed, one
  optional released-Claire asset test skipped.
- **E0 Observation-v3/CaptureSession slice — implemented and real-input
  green.** Video jobs now emit a compact `observation-v3.json` plus bounded
  `pixel-observations.npz`. They bind exact source PTS, capture/model/source
  hashes, per-frame decoded RGB hashes, regional absolute/take-relative focus,
  exposure and flow signals, three-signal structural cut candidates, separate
  photometric-discontinuity candidates and explicit observation-epoch
  boundaries. Flat-frame correlation is unavailable rather than fabricated as
  zero or perfect agreement.
  A deterministic, path-free `capture-session.json` binds Capture v1,
  Observation v2, Observation v3 and their artifact hashes under the sealed job
  ledger. Readiness reconstructs all three evidence contracts from sealed bytes.
  The lane is uncalibrated, capped below the strong tier, re-decodes the source
  rather than claiming detector-ingress pixel identity, and is not consumed by
  retargeting. Synthetic regional/blur/cut/tamper tests and the checksum-pinned
  real CREMA-D API-to-GNM path pass.
- **Resource/readiness boundary — fail-closed.** Capture now accepts at most
  7,200 frames and 20 billion aggregate decoded pixels, keeping the mandatory
  Observation-v2 JSON under its 64 MiB verification ceiling with measured
  margin. Capture NPZ/JSONL and Observation-v3 NPZ loaders enforce compressed,
  expanded, member, dtype and schema limits before readiness reconstruction.
  CaptureSession structural verification cannot approve production: its current
  unbound subject, unknown neutrality/identity and `production_validated=false`
  claims remain a separate required failure.
- **Final verification — green.** The hardened slice passed `55` focused
  capture/evidence/readiness/real-CREMA tests and the complete repository passed
  `492 passed, 1 skipped, 1 dependency warning in 483.17s`. The skip is the
  duplicate opt-in released-Claire asset test; retained learned routes and the
  checksum-pinned real video path ran.
- **E0 as a whole — not complete.** Frozen multiview/video evidence baselines
  and the Observation-v3 viewer diagnostics remain before advancing to A1.

## Decision summary

AutoAnim is a working, instrumented research prototype. It is not yet a
production-approved digital-human system. Production quality will not come
from increasing a smoothing window, increasing a texture file dimension, or
calling every detected video frame trustworthy.

The selected architecture has three evidence-first solves:

1. **Audio performance:** use the current learned Audio2Face motion as a data
   prior, add independently reviewable phone events, and optimize the complete
   articulator trajectory with contact, timing, anatomy and region-specific
   dynamics constraints.
2. **Identity and appearance:** retain GNM as the target model, expand from 68
   points to versioned dense GNM surface correspondences, jointly fit shared
   identity/cameras/silhouette/appearance, then add a separately versioned
   neutral corrective and measured PBR material tier.
3. **Video performance:** compute per-region uncertainty from the retained
   pixels before routing motion. Video owns what is visibly reliable; audio may
   repair only weak or missing speech articulation and may propose hidden tongue
   motion with explicit inferred provenance.

The native application becomes the review and correction workspace. Portable
GLB/Three.js remains an interchange preview; the production native viewport
will eventually evaluate raw GNM controls and revisions directly with Metal.

## Current measured baseline

### Audio

Retained real learned job `01kxvby11g6gg7qb978njn87t0`:

- Audio2Face v2.3.1 Claire at 30 fps, dense calibrated Claire-to-GNM retarget.
- 211 final frames; exact media-clock viewer.
- expression effective rank at 95% energy: 5;
- 32/211 frames modified by the emergency mouth continuity projection;
- maximum face-local mouth step: 0.03949 interocular;
- 3/3 inferred contact frames attained after continuity restoration;
- smallest retained articulation range ratio: 0.9715;
- no independent phone events, F/V contacts, tongue contacts, collision
  reference or human approval;
- `production_gate.passed=false`, correctly.

The rigid quality is sequence/context loss. The v2.3 source predicts locally,
then a five-frame Savitzky-Golay conditioner, two geometry projections, broad
Rhubarb gates and an emergency speed cap reduce transient variety without
creating anticipatory or carry-over coarticulation. GNM has no mandible joint;
the recovered jaw observation currently becomes a `jawOpen` floor rather than
rigid lower-teeth and tongue-root motion.

### Multiview identity and texture

The current calibrated synthetic positive control has five accepted views and
approximately 132 degrees of yaw coverage. Its 256 atlas is 53.05% directly
observed overall and 61.89% directly observed on the skin component. It proves
the shared-identity, visibility, texture projection and provenance mechanics;
it does not prove a real person's likeness.

The current real image path reduces MediaPipe's 478 points to 68 and the single
view path fits only 20 identity modes. The multiview solver correctly keeps
GNM identity dimensions 170:253 neutral because they control eyes/teeth and
are unobservable through the sparse face regressor. No retained, consented,
calibrated real subject with an independent metric scan exists.

An 8192-wide packed AutoAnim atlas is not a native 8K skin map: the packed GNM
skin domain receives about 72% of atlas width. Production masters therefore
need separate 8K/16K skin UDIMs and component maps, with bounded derivatives
for GLB.

### Video

Retained real CREMA-D job `01kxve1hnqqa48xyn6g0xyz0zj`:

- 67/67 frames detected at exact source PTS;
- final expression-motion correlation 0.8875;
- 14/14 high-confidence inferred contact frames reach the character seal;
- no strong blink is present to evaluate;
- 41.97% of one-sided neutral residual samples are clipped;
- no finite MediaPipe visibility, presence or face-confidence samples exist;
- the fallback global `tracking_quality` is 1.0 on every frame solely because
  all landmarks are inside a loose image bound.

The last fact is a routing defect. A blurred or occluded mouth can be treated
as fully trusted by retargeting/audio repair even while the diagnostic evidence
correctly caps the same geometry-only observation at 0.5.

## Integrated production data model

### CaptureSession

Every identity or performance capture must retain:

- source media hashes and immutable ordered inputs;
- subject, consent, permitted-use and expiry binding;
- camera `K`, distortion, `R|t`, timebase, exposure, ISO, shutter, focus and
  white balance when available;
- capture role, lighting/polarization state, color/scale references;
- all detector observations, model/runtime hashes and uncertainty;
- accepted/rejected state with deterministic reason codes;
- fit versus held-out camera assignment;
- capture-protocol and schema versions.

Unknown evidence is not zero, neutral, generic, or approved.

### PhoneEvent

The audio lane uses 48 kHz project ticks and stores:

- event ID, normalized phone, word, stress, place, manner, voicing and rounding;
- start, apex and end ticks plus confidence;
- independent-review state and annotator/evidence references;
- source audio, transcript, dictionary, aligner and annotation hashes.

An LLM may author phrase-level intent, affect, gaze and body beats. It never
authors phone timing, lip contact, jaw or tongue coefficients.

### RegionalObservation

Every video frame carries separate mouth, eyes, upper-face and head evidence:

- observed/occluded/blurred/offscreen/missing/unknown state;
- confidence and eventually calibrated covariance;
- face/crop pixel size, blur, exposure and saturation;
- landmark innovation and flow/reprojection residual;
- forward/backward optical-flow consistency;
- mask coverage, cut candidates, observation epochs and identity-continuity
  diagnostics; MediaPipe's private tracker-reinitialization state is not
  claimed;
- motion owner, repair/fill state and provenance.

### CharacterRevision

In addition to the current GNM identity and material artifacts, production
revisions will retain:

- neutral identity corrective plus topology/deformation-transfer version;
- observed/inferred/generic geometry masks;
- master component UDIMs and bounded runtime derivatives;
- scan, held-out geometry, held-out appearance and expression-stress reports;
- capture tier and every claim gate.

## Target algorithms

### Event-aware audio trajectory

For final GNM state `X`, current Audio2Face/retarget motion `X_hat` remains the
data prior:

```text
min_X sum_t ||W_data(t)(X_t - X_hat_t)||²
      + lambda_v(t)||D1 X||²
      + lambda_a(t)||D2 X||²
      + lambda_j(t)||D3 X||²
      + sum_event lambda_event ||G_event(X) - target_event||²
```

Subject to region bounds, lip ordering, character seal reachability, bilabial
and labiodental contacts, silence neutrality, face-normalized step limits and,
when calibrated, tongue/teeth/palate proximity. Fast high-confidence contacts
reduce temporal regularization; silence and uncertain steady vowels permit more.
Contact anchors never pass through a generic low-pass filter.

Initial coarticulation search ranges of 40-120 ms anticipation and 50-150 ms
carry-over are tuning ranges, not universal constants.

### Dense identity fitting

Build a hashed MediaPipe-478 to GNM triangle/barycentric mapping from synthetic
GNM renders over identities, cameras and poses. Reject unstable points and
manually audit lips, eyelids, nostrils, contour and ears. Fit one shared
`beta_head[170]`, per-view camera/expression nuisance, and calibrated scale using:

```text
L = wl * dense_landmarks
  + ws * silhouette_distance
  + wp * robust_photometric
  + wf * dense_features
  + wb * identity_prior
  + we * neutral_expression_prior
  + wc * camera_prior
```

Held-out geometry is authoritative. Photometric improvement cannot excuse a
geometry regression. Eye/teeth modes remain zero without dedicated evidence.

After the parametric fit passes, solve a bounded, Laplacian-regularized neutral
surface corrective with observed/inferred provenance and transfer it through
GNM expressions. It supplements rather than replaces editable GNM controls.

### Measured appearance

Ordinary RGB views produce only baked appearance. A relightable material claim
requires cross-polarized/multilight capture, calibrated color, measured normals
or geometry and held-out lights. Maintain separate high-bit-depth master maps
for skin, eyes, teeth/gums and tongue. AI-completed pores remain inferred.

A pore claim requires native map frequency, measured millimetres per texel and
MTF/power-spectrum retention against reference detail. File size alone is not
evidence.

### Confidence-aware video solve

The generic tracker initializes a direct GNM solve; it is not ground truth.
Pixel evidence determines per-region ownership. High-confidence video owns
visible head/gaze/asymmetry/upper-face/lip contour and contacts. Audio repairs
only low-confidence lower-face intervals and contributes inferred tongue.
Run-level hysteresis prevents ownership flicker. The final offline solve is
bidirectional, exact-PTS and event-aware; high-confidence anchors prevent drift.

Ordinary 25-30 fps RGB supports subtle-expression candidates, not a guaranteed
microexpression claim. A microexpression tier needs roughly 120-200 fps,
short exposure, adequate pixels and FACS onset/apex/offset evidence.

## Native production workspace

The native app will provide:

- synchronized source and 3D panes, plus calibrated overlay when valid;
- source, repaired and final A/B revisions at one exact media time;
- neutral/GNM-only/corrective/textured/animated modes;
- material isolation for base color, normal, displacement, roughness, specular
  and subsurface maps;
- timeline lanes for phones, waveform, articulation, affect, gaze, blink, head,
  tongue, regional confidence, conflicts, repairs and authored edits;
- close-up mouth/tongue/eyes and source residual heatmaps;
- semantic additive correction curves, ownership locks, undo/redo and immutable
  revision promotion;
- jump-to-conflict and rerun-validation workflows.

The current Three.js viewer remains a portable GLB inspector. Native production
review moves to raw GNM evaluation in `MTKView` so morph compression does not
hide or introduce control errors.

## Phased implementation and gates

Every phase uses the same strict loop: build, code/claim review, focused tests,
real-input E2E, visual/numerical inspection, fix, full regression, repeat.

### Phase E0 - evidence contracts and frozen baselines

Build:

- `PhoneEvent`/TextGrid import and provenance;
- Observation v3 regional pixel evidence;
- `CaptureSession` schema and claims taxonomy;
- frozen result/array hashes for the retained audio, multiview and video jobs.

Gate:

- exact timestamps/PTS and existing control arrays remain byte-identical;
- unknown/missing evidence never becomes zero or neutral;
- malformed/tampered annotations or capture bindings fail closed;
- synthetic blur, exposure, regional occlusion and cuts produce deterministic
  regional reason codes;
- the retained real inputs run end-to-end;
- the full suite passes.

### Phase A1 - phone qualification and event metrics

Build onset/apex/release, closure duration, bilabial, labiodental, rounding and
tongue event evaluators. Add MFA only when locally provisioned; manual reviewed
TextGrid is the first supported production evidence.

Gate:

- independent apex median <=1 frame and p95 <=2 frames at 30 fps;
- onset/release median <=40 ms and p95 <=80 ms;
- bilabial F1 >=0.90 and labiodental F1 >=0.85;
- false contact on silence/non-contact phones <1% of frames;
- annotation provenance and a minimum event count are mandatory for approval.

### Phase A2 - event-aware trajectory optimizer

Build the whole-utterance constrained solve around current learned controls.
Keep the mouth-close quarantine, character seal calibration, contact-anchor
restoration and full tongue mapping.

Gate:

- current geometry safety remains green;
- contact peak retention >=0.99;
- articulation range and effective-rank retention >=0.90;
- every onset/apex/release/contact metric is non-regressing;
- no new lip, tooth or tongue penetration;
- real annotated A/B plus blinded animator review prefers or approves the new
  trajectory. A lower jerk score alone cannot pass.

### Phase A3 - mandible/oral anatomy and acting layers

Add an auxiliary mandible pivot, rigid lower teeth/gums, jaw-relative tongue
root and soft-tissue correctives. Compile LLM/user acting beats into additive
upper-face, gaze, head and body revisions while articulation owns the mouth.

Gate:

- lower-teeth rigid residual <=0.25 mm p95;
- tongue target error <=0.5 mm p95 where calibrated;
- no penetration deeper than 0.25 mm;
- applying acting does not materially change phone timing/contact;
- every generated nonverbal layer is seedable, editable and labeled generated.

### Phase I1 - dense GNM correspondences

Build retained 478-point observations, barycentric mapping, confidence and the
68-versus-dense shared-identity benchmark.

Gate:

- deterministic mapping/hash and synthetic recovery;
- held-out dense NME at least 15% below the 68-point baseline;
- scan error does not regress on any locked real fixture;
- mixed-person and duplicate-view rejection remain green;
- dimensions 170:253 remain zero without eye/dental evidence.

### Phase I2 - guided/calibrated capture and joint fit

Add capture QA, full-head roles, calibration verification, masks, silhouettes,
lighting/exposure nuisance and coarse-to-fine photometric fitting.

Gate:

- calibration and scale references independently recompute;
- median facial surface error <=1.5 mm, p95 <=4 mm and median normal error
  <=12 degrees on locked consented scan fixtures;
- held-out aggregate NME <=0.025 and each subject <=0.040;
- multiview improves scan error and held-out NME by >=20% on >=80% of fixtures;
- no photometric win can hide a geometry regression.

### Phase I3 - neutral corrective and material masters

Build versioned corrective transfer and separate component PBR masters with
measured/inferred masks.

Gate:

- corrective improves scan error >=15% on >=80% of subjects and never worsens
  neutral held-out NME by more than 2%;
- no flipped triangles, tearing or oral intersections in the full expression
  stress suite;
- studio measured skin coverage >=90%;
- seam delta-E2000 <=3 and held-out median delta-E2000 <=5;
- normal error <=10 degrees and pore-qualified displacement RMSE <=0.10 mm;
- pore/relightable flags fail without frequency and unseen-light evidence.

### Phase V1 - Observation v3 and viewer diagnostics

Status: pixel-evidence/CaptureSession foundation implemented; calibrated
classification, same-buffer detector ingress, adversarial real capture set and
viewer timeline/overlay remain.

Re-read pixels and emit regional crop resolution, blur/exposure, flow
consistency, landmark innovation, cut candidates, observation epochs and reason
codes. Keep it diagnostic-only during this phase. Add the confidence
timeline/source overlay.

Gate:

- existing PTS and retarget arrays are byte-identical;
- an occluded/blurred mouth cannot reduce a clear brow's confidence;
- no labeled bad region enters the provisional strong tier >=0.75;
- exact source-frame stepping and viewer readouts pass;
- retained CREMA-D and adversarial synthetic videos pass.

### Phase V2 - regional fusion and direct temporal tracking

Make Observation v3 the only confidence authority, add hysteretic channel
ownership, then add dense calibrated screen-space fitting and bidirectional
optimization.

Gate:

- trusted video channels remain bit-identical under repair;
- aperture correlation >=0.95 and p95 amplitude ratio 0.90-1.10;
- bilabial contact precision/recall >=0.95 within one labeled frame;
- long-take drift and screen-space motion error improve over the frozen baseline;
- cuts, occlusion, reappearance and missing spans remain deterministic.

### Phase U1 - native correction workspace

Build raw-track Metal viewport, side-by-side source, diagnostics lanes, revision
A/B, semantic edits and character-library evidence/rights status.

Gate:

- raw viewport samples the exact control frame for the requested rational PTS;
- source/GLB/native geometry comparisons pass stated tolerances;
- 100 job/revision switches do not grow resources;
- keyboard, accessibility, context-loss/recovery and static fallback pass;
- no revision can be promoted while a required claim gate is false.

### Phase Q1 - locked qualification and release

Run the complete real/synthetic matrix, animator and naive-viewer blinded
studies, rights/consent/deletion review, notices, load/recovery and Apple Silicon
device matrix. Artist time-to-approval and structural reanimation rate are the
decisive production metrics.

## External dependencies and constraints

- GNM is Apache-2.0.
- Audio2Face SDK/training code and weights have separate terms; retain model
  notices and complete product legal review.
- RAVDESS, VOCASET, Multiface, MICA, DECA, TEMPEH, FaceSynthetics and many
  related datasets/models are research/noncommercial or separately restricted.
- Research architectures may inform or benchmark AutoAnim but cannot silently
  become shipping dependencies.
- Production identity, performance and evaluation data require commissioned,
  consented capture with explicit likeness, biometric, derivative, training,
  retention and deletion rights.
- NVIDIA/CUDA differentiable renderers are not the native Apple path. Prefer
  analytic GNM Jacobians, Accelerate/Rust solves and Metal visibility/texture
  kernels; qualify Core ML/MPS model conversion per operator and device.

## Primary research references

- [NVIDIA Audio2Face 3D](https://arxiv.org/html/2508.16401) and
  [official repository](https://github.com/NVIDIA/Audio2Face-3D)
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/v3.4.1/user_guide/index.html)
- [Imitator contact-aware personalized speech animation](https://openaccess.thecvf.com/content/ICCV2023/papers/Thambiraja_Imitator_Personalized_Speech-driven_3D_Facial_Animation_ICCV_2023_paper.pdf)
- [Perceptual speech-mesh metrics](https://openaccess.thecvf.com/content/CVPR2025/html/Chae-Yeon_Perceptually_Accurate_3D_Talking_Head_Generation_New_Definitions_Speech-Mesh_Representation_CVPR_2025_paper.html)
- [FlowFace dense video tracking](https://openaccess.thecvf.com/content/CVPR2024/html/Taubner_3D_Face_Tracking_from_2D_Video_through_Iterative_Dense_UV_CVPR_2024_paper.html)
- [AVFace audio-visual 4D reconstruction](https://openaccess.thecvf.com/content/CVPR2023/html/Chatziagapi_AVFace_Towards_Detailed_Audio-Visual_4D_Face_Reconstruction_CVPR_2023_paper.html)
- [SPECTRE visual-speech reconstruction](https://openaccess.thecvf.com/content/CVPR2023W/ABAW/papers/Filntisis_SPECTRE_Visual_Speech-Informed_Perceptual_3D_Facial_Expression_Reconstruction_From_Videos_CVPRW_2023_paper.pdf)
- [DenseLandmarks](https://arxiv.org/abs/2204.02776) and
  [TEMPEH](https://arxiv.org/abs/2306.07437)
- [MICA metric identity](https://arxiv.org/abs/2204.06607),
  [DECA detail](https://arxiv.org/abs/2012.04012), and
  [FitMe](https://openaccess.thecvf.com/content/CVPR2023/html/Lattas_FitMe_Deep_Photorealistic_3D_Morphable_Model_Avatars_CVPR_2023_paper.html)
- [USC polarized spherical-gradient face capture](https://vgl.ict.usc.edu/Research/FaceScanning/EGSR2007_SGI_low.pdf)
- [Apple ARKit face blendshapes](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapes),
  [Vision optical flow](https://developer.apple.com/documentation/vision/vntrackopticalflowrequest), and
  [MetalKit MTKView](https://developer.apple.com/documentation/metalkit/mtkview/)

## Definition of completion

This objective is complete only when all audio, single/multiview identity,
texture, video, viewer and character-library gates above have current evidence;
real inputs and locked references pass; human production review passes; and
rights, consent and dependency approval are complete. Until then AutoAnim must
continue to report the exact achieved tier and unresolved gates.
