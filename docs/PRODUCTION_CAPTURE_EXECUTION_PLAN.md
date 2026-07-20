# Production facial performance and identity execution plan

Status: active implementation plan

Date: 2026-07-19

GNM revision: `3de70dfca5f3244620f44103c24b7cedc0dcb8b6`

AutoAnim baseline: `1b9b8f18b211c9ee7b45af39668daacd32fd70fc`

## Implementation ledger

- **U1a sealed native performance review — implemented and focused-test
  green.** Every eligible video-performance job can now be reconstructed as a
  canonical `autoanim.review-bundle/1.0` document from the HMAC-sealed artifact
  ledger. The bundle binds the retained input, exact rational source-PTS clock,
  display proxy, GNM/Performance versions, identity, seven fail-closed motion
  layers and the sole final renderable revision. It exposes no filesystem paths,
  production approvals or correction writer. A native Swift decoder independently
  checks the self-hash and structural contract; a bounded `autoanimReview` WK
  bridge accepts only typed cursor/layer/selection/revision messages and exact
  job/comparison/revision bindings. The SwiftUI workspace adds sealed layer
  state, exact frame/PTS stepping, compatible cross-job A/B and a B gate that
  remains locked until both viewers acknowledge the same server-decoded frame,
  declared layer state and selection. Bridge commands use bounded lossless
  per-viewer FIFO snapshots; camera-orbit equivalence remains unverified.
  Real candidates `01kxz19xtk9aytx0rfjj0gddzk` (visual only) and
  `01kxyy53rgqamj8y7hddaqf385` (audio repair) share comparison key
  `025363394c08717ce393d62e430317b633d9e098c27c1083b7df96297743500c`.
  The repair changes only 32 dedicated tongue controls on 38/67 frames; native
  GNM evaluation changes only 933 tongue vertices (2.663 mm maximum). It does
  not change lip or lower-face motion and therefore is not evidence that the
  reported rigid/closed-mouth lipsync is fixed. Visible-tongue correctness,
  signed collision freedom, phoneme timing, perceptual preference and artist
  approval remain explicit blockers.

- **A1 multi-articulator diagnostic foundation — implemented, reviewed, and
  regression-green.** `autoanim.phone-articulation-report/1.0` records bilabial inner-lip
  closure, coarse lower-lip/upper-teeth and tongue/upper-teeth proximity, and
  rounded-mouth width against contextual phone spans. It measures un-clipped
  global proxy-run boundaries, uses exact 30/60 fps ticks, binds exact controls,
  identity, GNM, 68-point regressor and decoder assets, and is reconstructed in bounded GNM mesh
  batches from a sealed job. It never changes animation. Independent review found that
  normal phone intervals are not articulatory contact-state labels and that the
  current F/V and tongue full-surface distances do not separate intended GNM
  prototypes. The report therefore exposes only a `phone_span_proxy_gate`; its
  `production_gate` is structurally false. A separate reviewed articulation
  tier, anatomical target surfaces, protrusion measurement, a verified
  character profile, speaker-balanced corpus and perceptual study remain
  mandatory. The full current research and dependency decision is recorded in
  `PRODUCTION_RESEARCH_UPDATE_2026-07-20.md`. Final verification after the last
  source change: `562 passed, 1 skipped` in the complete Python suite and `12
  passed` in the release-mode macOS package. Retained real learned-audio job
  `01kxyj1bydcsm1r8w0sjcwnhcn` processed eight seconds of checksum-pinned
  LibriSpeech with its MFA TextGrid into 240 animation frames, a playable GLB
  and preview, sealed oral reports, and a reconstructable articulation report.
  All 240 oral frames were evaluated with zero reported tongue-collision or
  lip-inversion risk. Its coarse diagnostic F1 values remain poor or unavailable
  (bilabial/labiodental unavailable, rounded `0.1143`, tongue `0.1468`), so this
  run validates the evidence path and simultaneously confirms that the current
  motion is not production-qualified.

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
  Interactive pixel diagnostics are deliberately narrower than capture: at
  most 1,800 frames, with an identity-mapped H.264 display proxy, square pixels,
  zero rotation and no clean-aperture crop. The viewer exposes the reason when
  that lane is unavailable. Exact proxy-frame PNG decode is currently one
  fail-fast application-wide operation; a per-job queue/cache belongs to the
  native resource scheduler before concurrent production review.
- **Prior E0 verification checkpoint — green.** Before the viewer slice below,
  the hardened capture/evidence/readiness implementation passed `55` focused
  tests and the then-complete repository passed `492 passed, 1 skipped, 1
  dependency warning in 483.17s`. Those numbers describe that checkpoint, not
  the current uncommitted tree. The current-tree result is recorded only after
  the final regression below completes.
- **E0 frozen-baseline and viewer slice — implemented and regression-green.**
  The real checksum-pinned 67-frame CREMA-D path now freezes exact
  source PTS, detection, 478-landmark, blendshape, facial-transform,
  Observation-v3 confidence/reason and final GNM expression/rotation array
  hashes. A manual two-run replay reproduced all 50 numeric capture,
  observation and performance arrays exactly; the committed gate freezes the
  named capture, Observation-v3 and final-motion hash groups that would expose
  clock or retargeting drift. The calibrated synthetic multiview oracle
  separately freezes shared identity, nuisance, held-out and fitted-landmark
  hash groups; it remains synthetic evidence, not proof of a real person's
  likeness.
  A derived `autoanim.observation-v3-view/1.0` endpoint reconstructs every
  displayed field from sealed capture/NPZ/summary artifacts. The viewer binds
  it to the older exact-PTS performance evidence, draws regional ROIs over the
  source video, reports tracker and pixel diagnostics separately, and labels
  provisional pixel scores as non-authoritative. Paused steps use a bounded
  manifest-bound PNG endpoint decoded by display-order proxy frame index;
  these are lossy CRF-18 proxy pixels, not retained-source pixel identity.
  Playback uses
  `requestVideoFrameCallback` and maps its media time to the nearest verified
  proxy timestamp. A real 67-frame CREMA-D take passed forward/backward exact
  stepping, paused native-control seek, delayed-request/play recovery,
  callback-unavailable fallback and responsive WebKit inspection. Positive
  first-PTS CFR and VFR fixtures retain all frames after proxy rebasing, and the
  1,800/1,801 interactive boundary is gated through the viewer/API/final-frame
  paths. Final verification on 2026-07-20: `61` focused video/review tests;
  `542 passed, 1 skipped, 1 dependency warning in 529.22s` for the complete
  Python repository; and `12` release Swift tests across four suites. The
  ad-hoc signed native bundle launched its supervised authenticated service,
  reported ready, listed the retained jobs, reconstructed the 67-frame review
  document and served exact proxy frame 16. The remaining WebKit automation
  gap is tracked as a release-workspace test requirement, not represented as a
  completed automated gate.

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

## 2026-07-19 swarm research synthesis

Three independent repo audits covered identity/material capture, facial-video
performance and the native review workflow. Their shared conclusion is that
AutoAnim's strongest assets are exact clocks, immutable provenance, editable
GNM topology and geometry-level safety. The weak evidence is upstream model
fidelity and real qualification. The application must therefore preserve a
preview lane and build a separately gated offline production lane rather than
renaming the current output.

### Smooth audio-driven performance

More generic low-pass filtering cannot create production coarticulation. The
current v2.3 source predicts local poses, while the post-conditioner and
geometry projection can only remove variation. The selected production design
is:

1. Audio2Face-3D v3 as a version-pinned, separately authenticated NVIDIA-worker
   prior. NVIDIA's published architecture adds HuBERT, diffusion and recurrent
   sequence state, and emits face, tongue, jaw and eye controls. The current
   ABI/clock integration is transport-ready, but no genuine v3 worker result or
   paired perceptual qualification exists. See the
   [Audio2Face-3D paper](https://arxiv.org/abs/2508.16401),
   [official SDK](https://github.com/NVIDIA/Audio2Face-3D) and
   [v3 model card](https://huggingface.co/nvidia/Audio2Face-3D-v3.0).
2. Independently reviewed phone onset/apex/release and contact evidence rather
   than LLM-authored phoneme timing. Phrase-level LLM acting remains an editable
   additive upper-face/gaze/head/body proposal.
3. A whole-utterance GNM trajectory solve around the learned prior with
   phone-conditioned anticipation/carry-over, contact, lip ordering, anatomical
   bounds and region-specific velocity/acceleration/jerk terms. Contact anchors
   bypass generic smoothing.
4. Paired blind qualification on the same identity, audio, renderer and
   retarget. Structural smoothness cannot override intelligibility, contact,
   event timing, oral artifacts or animator preference.

Speech does not determine a single correct blink, gaze or emotional acting
track. [EmoTalk](https://openaccess.thecvf.com/content/ICCV2023/html/Peng_EmoTalk_Speech-Driven_Emotional_Disentanglement_for_3D_Face_Animation_ICCV_2023_paper.html)
supports separating content, emotion, identity and intensity, while
[probabilistic speech-driven synthesis](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Probabilistic_Speech-Driven_3D_Facial_Motion_Synthesis_New_Benchmarks_Methods_and_CVPR_2024_paper.html)
shows why the non-speech performance is one-to-many. AutoAnim therefore never
lets a deterministic acoustic estimate masquerade as captured acting.

### Person-specific geometry and appearance

The product must expose capture tiers rather than one "photo to perfect head"
button:

| Tier | Defensible output | Evidence still unavailable |
| --- | --- | --- |
| Single photo | preview GNM initializer with visible/hidden uncertainty | metric depth, rear anatomy, measured pores |
| Guided 5–12 view | shared-identity production candidate with calibrated cameras, scale and untouched held-out views | eye/dental modes without close-ups; scan truth unless separately captured |
| Polarized multiview/multilight | measured diffuse/specular/normal/roughness candidate with per-texel provenance | pores beyond capture MTF; subsurface unless explicitly measured |
| Scan/light stage | reference geometry/material tier plus artist cleanup | automatic production approval |

The next solver retains the sparse 68-point path only as an initializer, then
adds versioned 478-to-GNM correspondences, semantic masks, silhouette, dense
features/flow, robust linear-RGB photometric residuals, per-view expression and
lighting nuisance, covariance and fixed-camera held-out evaluation. Head modes
`0:170` are solved from face/head evidence; eyeball and teeth modes remain zero
without their own captures. A separately versioned neutral corrective and
tangent-space detail preserve identity outside GNM's low-frequency PCA span.

Learned single-image systems are useful priors and research comparators, not a
commercial dependency choice: public DECA and MICA releases carry
non-commercial/model-data constraints. The [NoW metrical benchmark](https://now.is.tue.mpg.de/metricalevaluation.html)
shows why learned priors help while still leaving measurable millimetre-scale
error. [FLAME 2023 Open licensing](https://flame.is.tue.mpg.de/modellicense.html)
is materially different from older model releases, so every provider must bind
its exact model and asset license. A commercially permissive geometric option
is [COLMAP](https://colmap.github.io/license.html), with each optional dependency
reviewed separately.

For physical skin, RGB baking remains `captured_appearance`, not albedo.
[PolFace](https://dazinovic.github.io/polface/) demonstrates a practical
two-polarization smartphone route to high-resolution diffuse, specular and
normal maps; light-stage work such as
[Digital Emily](https://vgl.ict.usc.edu/Research/DigitalEmily/) supports the
same cross/parallel-polarization separation. AutoAnim will tile 4K/8K masters,
preserve measured/inferred/inpainted/generic texel labels and qualify pores by
physical sampling/frequency evidence, never by filename dimensions.

### Video-follow performance

MediaPipe remains the responsive Mac preview/fallback, not the production
truth. Its blendshape model is designed for front-facing mobile-AR conditions
and its own [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf)
warns about lighting, motion, overlap and jitter. Exact transport of those
coefficients proves reproducibility, not agreement with the performer.

The selected production path introduces a provider-neutral visual performance
track with exact PTS, identity/camera/expression/jaw/eye estimates, dense
landmarks or mesh samples, regional covariance, reprojection/flow residuals,
occlusion and shot/identity epochs. An offline provider may use commercially
cleared equivalents of research systems such as
[SMIRK](https://github.com/georgeretsi/smirk),
[MICA](https://github.com/Zielon/MICA) or the
[metrical tracker](https://github.com/Zielon/metrical-tracker), but the ABI is
the stable product boundary; public research weights are not assumed shippable.

Fusion happens in screen/mesh geometry, never by averaging unrelated model
coefficient spaces. High-confidence video owns visible head, gaze, asymmetry,
upper face, lip contour and contact. Audio supplies an uncertain lower-face
prior and hidden tongue candidate only where visual evidence is weak. A
content-sync estimator separately detects dubbed/shifted audio; container-clock
agreement is not lip-sync evidence. Ordinary 24–30 fps video can support subtle
expression candidates, while a microexpression claim requires high-frame-rate,
short-exposure capture and FACS onset/apex/offset truth.

### Native review product

The near-term production workspace keeps the existing Three.js renderer inside
the authenticated WKWebView and builds native SwiftUI workflow, comparison,
evidence, review and export around it. An immediate Metal rewrite would recreate
glTF loading, materials, animation and camera behavior before fixing the larger
artist-workflow gap. Three's
[GLTFLoader](https://threejs.org/docs/pages/GLTFLoader.html) already covers the
required glTF material extensions; a later
[MTKView](https://developer.apple.com/documentation/metalkit/mtkview) raw-GNM
viewport remains appropriate once the review contract passes.

The production workspace will add source/3D/compare panes, reconstruction overlays,
held-out multiview filmstrips, texture confidence/provenance/material channels,
audio/video timelines, warning bookmarks, exact frame/phone navigation,
baseline-versus-revision A/B, scoped immutable reviews and hash-bound export
packages. Phase N2 native/JavaScript messages will use a bounded versioned bridge through
[WKUserContentController](https://developer.apple.com/documentation/webkit/wkusercontentcontroller).
Machine readiness and human approval remain independent; any artifact hash
change invalidates the corresponding review.

### Execution order and stop/go rules

After E0 closes, work proceeds in four parallel evidence tracks with one shared
release review:

1. **A1/A2:** reviewed phone-event qualification, then the event-aware GNM
   trajectory optimizer. Do not call audio production-quality before paired
   perceptual gates pass.
2. **I1/I2:** dense correspondences and a calibrated real same-subject fixture,
   then joint identity/camera/silhouette/photometric fitting. Do not begin
   generative hero skin as a substitute for measured likeness.
3. **V2:** VisualTrack provider ABI, geometry-domain bidirectional sequence
   solve and content-sync evidence. Observation-v3 remains diagnostic-only
   until calibrated regional confidence and authority invariants pass.
4. **N2/U1:** native review-bundle/bridge, reconstruction and performance A/B,
   scoped reviews and export. Keep the WKWebView renderer until exact-clock,
   context-recovery, accessibility and job-switch resource gates pass.

Each track keeps the strict build → independent review → synthetic and real
tests → fix → complete regression loop. No phase advances on a metric whose
ground truth is produced by the same tracker that drives the output.

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

The implemented Rust `surface_secondary_candidate` remains a separately gated,
additive target-relative physics layer. Its CPU/Rayon benchmark passed, while
the SIMD promotion gate did not and no Metal/wgpu backend is claimed. Phase P1
must first bind shared evaluated frames, protect lips/jaw/teeth/tongue contact,
and retain reports; physics may never become lipsync or acting authority. The
execution evidence and blockers are maintained in
`docs/NATIVE_MACOS_PHYSICS_PLAN.md`.

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

Status: pixel-evidence/CaptureSession foundation and exact sealed display-proxy
frame plus source-timed ROI overlay are implemented. The proxy is lossy and is
not represented as source-pixel identity. The confidence timeline, calibrated
classification, same-buffer detector ingress and adversarial real capture set
remain.

Re-read pixels and emit regional crop resolution, blur/exposure, flow
consistency, landmark innovation, cut candidates, observation epochs and reason
codes. Keep it diagnostic-only during this phase. Add the confidence
timeline/source overlay.

Gate:

- existing PTS and retarget arrays are byte-identical;
- an occluded/blurred mouth cannot reduce a clear brow's confidence;
- no labeled bad region enters the provisional strong tier >=0.75;
- exact display-proxy frame stepping bound to source PTS and viewer readouts pass;
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
