# Production facial animation research update

Status: implementation-driving research; production approval is not claimed
Date: 2026-07-20
Repository baseline audited: AutoAnim `64f6ff27286f19aa12544828a762b1478cd9e7c6`, Google GNM
`3de70dfca5f3244620f44103c24b7cedc0dcb8b6`

This memo records the second independent research pass over the three requested
production tracks: audio-driven speech/acting, image or multiview identity and
appearance, and video-follow facial performance. It reconciles current primary
research with the actual AutoAnim code and chooses executable phases without
renaming proxy evidence as production truth.

## Executive decision

The rigid result is primarily an upstream motion-model and articulation problem,
not a shortage of smoothing or physics. The local learned backend is NVIDIA
Audio2Face v2.3.1 through MLX. It produces independently estimated 30 Hz poses;
AutoAnim then applies a bounded retarget, a five-sample conditioner, resampling,
contact repair and a geometry speed limit. Those operations can suppress jitter,
but cannot invent anticipatory/carry-over coarticulation or a coherent phrase-
level performance.

The strongest next motion prior is genuine Audio2Face v3 sequence inference at
60 Hz. The supported NVIDIA SDK deployment remains CUDA/TensorRT, but the pinned
public `network.onnx` is portable. A direct Apple-Silicon CPU probe overturned
the earlier hardware assumption: genuine v3 weights can run locally through
ONNX Runtime even though that path is not NVIDIA's supported SDK runtime.

Phase A3-L now implements that candidate. It verifies the exact public model and
Claire profile hashes, checks the ONNX ABI, executes the official recurrent
one-second/half-second schedule, retains the 15-30-15 center frames, ports the
pinned MIT SDK skin/tongue/jaw/eye postprocess and regularized BVLS solvers, and
retargets the 60 Hz controls into GNM without the v2.3 Savitzky-Golay pass. The
external-worker importer remains separate and still cannot prove inference.
Neither route is production-qualified.

The executable prerequisite is A1 multi-articulator qualification. The first
implementation pass exposed an important evidence error: normal phone intervals
cannot stand in for independently reviewed articulatory onset/contact/release or
negative states. Direct GNM prototype checks also found that the initial unsigned
full-surface F/V and tongue distances do not separate the intended poses. A1 in
this revision is therefore instrumentation only: exact-clock phone-context
statistics, un-clipped proxy runs, bounded mesh-batch reconstruction and explicit
asset bindings. Its production gate is structurally false. Qualification still
requires separate reviewed articulation tiers, anatomical target surfaces,
protrusion, a verified character profile, held-out data and perception evidence.

That diagnostic foundation completed its strict implementation loop. Before
A3-L, the complete Python repository passed `562` tests with one optional
released-Claire asset test skipped, and the release-mode macOS package passed
all `12` tests.
Retained job `01kxyj1bydcsm1r8w0sjcwnhcn` ran eight seconds of real LibriSpeech
plus its MFA TextGrid through the learned backend, GLB/preview export, oral
geometry validation and sealed report reconstruction. Its weak coarse proxy
scores are negative product evidence, not a hidden success: the run proves that
the diagnostic path works while confirming the current articulation quality
does not pass a production claim. A3-L then ran the same eight-second normalized
audio through 18 recurrent executions and emitted exactly 480 frames. On this
Mac, model inference took 2.16 seconds, and the measured raw-consumer wall time
(descriptor verification, session, inference and streamed postprocess) was
6.09 seconds. The complete final-source service job took 86.63 seconds with
peak resident memory 2,919,104,512 bytes (about 2.72 GiB). Retained job
`01kxyptrggtgw7xypj6hp1g5t3` has a verified local HMAC seal and all 14 ledgered
artifact sizes and SHA-256 hashes reconstruct exactly. The final viewer
uses 38 bounded morph targets, passes the full-track reconstruction gate, keeps
zero tongue/teeth collision-risk frames and changes no lip-contact
classification. These are executable engineering results, not perceptual or
SDK-parity approval.

After the final A3-L source changes, the complete Python repository passed
`616` tests with only the two explicitly opt-in released-asset/model tests
skipped. The real cached v3 ONNX smoke test passed separately, the release-mode
Audio2Face runner passed all `7` tests, and the release-mode native macOS package
passed all `12` tests. The signed development bundle also passed token rejection,
authenticated health, supervised-process shutdown and GUI-launch smoke checks.

Identity realism must likewise be benchmark-first. AutoAnim currently offers an
honest sparse initializer and provenance-aware RGB bake, not a measured person-
specific head or 8K PBR skin. Dense fitting should start only with one consented,
calibrated subject and an independent metric scan. Texture resolution is not a
substitute for geometric likeness, lighting separation or captured spatial
frequency.

Video performance should next gain a provider-neutral `VisualTrack v1` with
regional calibrated uncertainty, visibility/occlusion, subject and shot epochs,
shared identity and per-frame expression separation, followed by a bidirectional
offline solve. Replacing MediaPipe immediately with a research tracker would
hide licensing and calibration problems rather than solve them.

## Track A — audio to smooth speech and editable acting

### What the repository actually runs

- `src/autoanim_gnm/audio_pipeline.py` runs v2.3 Claire locally, preserves skin,
  tongue, jaw observations and eyes, performs dense ARKit-to-GNM retargeting,
  and applies contact/continuity safeguards.
- GNM has head, neck and eye joints but no mandible joint. Jaw, teeth and tongue
  coupling are approximated through expression coefficients and geometry
  corrections.
- `src/autoanim_gnm/sequence_provider.py` strictly validates the official v3
  60 Hz clock, window schedule, hashes and Claire control schema.
- `src/autoanim_gnm/a2f_v3_local.py` runs the exact hash-pinned v3 ONNX graph on
  CPU with recurrent state, deterministic audited noise and bounded chunk
  callbacks. `src/autoanim_gnm/a2f_v3_postprocess.py` ports the pinned Claire
  interpolators, geometry composition, BVLS, five-point jaw solve and seeded
  eye animator.
- `a2f-v3-local` is exposed in the CLI, service and native-hosted web UI at
  native 60 fps. It records genuine ONNX inference while explicitly setting
  official SDK runtime, SDK parity, postprocess parity and production approval
  to false. It does not silently fall back.
- Jaw matrices are retained but not applied. Emotion conditioning uses the
  official ten-channel order only for explicit manual affect; dialog/acoustic
  heuristics fail closed to neutral and the later acting layer is disabled to
  avoid double application.
- `src/autoanim_gnm/phone_events.py` imports immutable MFA/Praat evidence, but
  its legacy report scores bilabials only and intentionally fails labiodental,
  tongue and false-contact gates.
- `src/autoanim_gnm/phone_articulation.py` now adds mesh-batched, sealed diagnostic
  telemetry. It deliberately does not promote phone-span F1 or unsigned
  full-surface proximity to production articulation truth.
- The current taxonomy is a small English ARPABET-oriented subset. Unknown
  phones remain context only; language-specific inventories and dental,
  alveolar, lateral and constriction targets must be explicit profile data.

### Why the motion appears rigid

1. The v2.3 source predicts local poses rather than a recurrent/diffusion
   utterance trajectory.
2. The learned solve and post-conditioner both regularize motion; temporal
   detail can be removed twice.
3. The 383-dimensional GNM expression output is driven by a much smaller
   effective motion subspace, especially in lower face and tongue.
4. A whole-lower-face continuity intervention couples articulators that should
   have different timing and dynamics.
5. Bilabial contacts are protected, but labiodental, tongue, teeth, jaw and
   rounding targets lack independent character-specific calibration.
6. Audio does not determine a unique blink, gaze, head gesture or emotional
   acting choice. Procedural secondary motion can look repetitive even when the
   mouth timing is numerically smooth.

### Selected production architecture

```text
audio + transcript + optional director/LLM phrase plan
  -> normalized audio and input QA
  -> genuine v3 sequence inference at 60 Hz
       -> local ONNX candidate for iteration and evidence
       -> authenticated supported-SDK worker for official parity/qualification
     (v2.3 remains labeled preview fallback)
  -> independently reviewed phone/contact evidence
  -> phrase-level affect plan with authored > reviewed > inferred precedence
  -> character-calibrated jaw/lip/teeth/tongue trajectory solve
  -> separate articulation, affect, gaze, blink, head and body layers
  -> GNM controls + complete provenance + correction revision
  -> automatic qualification + blinded animator/naive-viewer review
```

An LLM may propose low-rate intent, emphasis, valence/arousal, gaze targets and
acting beats. It must never author phone-frame timing or overwrite a reviewed
contact. Exact plan bytes, timing, vocabulary, model/provider and approval state
must be bound to the worker request. The external worker ABI has no affect input
and must be versioned rather than silently extended. The local runtime has an
explicit audited ten-value vector; keyframed acting still requires its own
versioned contract.

### A3-L local sequence candidate — implemented gate

- Exact model/profile hashes and ONNX tensor signatures are fail-closed.
- Official padding, warm-up, recurrent latent chaining, generated/retained
  frame hashes and exact 60 Hz target samples are retained as evidence.
- CPU is the default provider. Core ML was slower in the direct probe because
  the graph was split across providers; it is not selected by assumption.
- Raw 88,831-value frames are streamed into stateful postprocessing rather than
  retained for a long clip. The small solved controls, jaw diagnostics, eyes,
  runtime evidence and retarget calibration are retained.
- Application jobs are capped at ten seconds in this candidate because GNM
  render frames and exact low-rank GLB factorization still materialize the full
  track. The inference boundary alone has a 600-second bounded-streaming test;
  that is not an end-to-end duration claim.
- The same real eight-second input produces 480 frames with no timestamp drift.
  Compared with the retained 30 Hz v2.3 job, v3 reduced lower-face acceleration
  p95 from 1.111 to 0.586 and jerk p95 from 1.757 to 0.662, and reduced per-frame
  mouth-step p95 from 0.0304 to 0.0195 interocular units. Mouth-speed p95 rose
  from 0.911 to 1.170 IOD/s and limiter interventions rose from 8/240 to 25/480,
  so this is evidence of smoother temporal derivatives, not yet a perceptual
  win.
- Tongue controls and reconstructed tongue geometry moved on 479/480 frames;
  the full-track oral audit found no tongue/teeth collision risk or lip-order
  inversion. Visibility, phoneme correctness and perceptual tongue quality are
  still unverified.
- The tested clip used 38 of the 40 permitted morph targets. Diverse real clips,
  positive lip-contact examples, boundary-crossing oral cases and interpolated
  between-key validation are required before animated-viewer availability can
  be treated as general rather than clip-specific.

### V3-R1 worker requirements

- Pin SDK, model/network, identity profile, CUDA, TensorRT and container digest.
- Execute the official warm-up/padding/15–30–15 recurrent schedule and official
  skin/tongue solver.
- Add an authenticated request/response ABI with nonce, expiry, key id,
  signature, device/runtime attestation and replay protection. SHA-256 alone
  proves integrity, not worker identity.
- Bind the ten-channel emotion vocabulary and per-frame/keyframed affect plan;
  record neutral-only explicitly when affect is absent.
- Verify jaw matrix convention with an official parity fixture before applying
  it to a GNM mandible corrective.
- Expose a mode-scoped API/native workflow with explicit timeout and fallback;
  never silently downgrade a requested production v3 job to v2.3.
- Keep outputs candidate-only until paired real-output and human qualification
  pass.

### Proposed audio qualification gates

These numerical targets are engineering proposals, not calibrated production
thresholds. They become eligible only after a speaker-balanced held-out corpus
contains independent articulation-state positives and negatives for each
language/profile and the measurements use validated character-bound surfaces.

- Apex error median at most one output frame and p95 at most two.
- Reviewed onset/release median at most 40 ms and p95 at most 80 ms.
- At least 100 independently reviewed events per critical contact family.
- Bilabial F1 at least 0.90, labiodental F1 at least 0.85, and false contact
  below 1% of independently labeled negative articulation states. The current
  phone-span proxy report does not provide this evidence.
- Contact peak retention at least 0.99; articulation range and effective-rank
  retention at least 0.90.
- No new lip ordering, tooth or tongue risk; no timestamp drift or generic v3
  lower-face smoothing.
- Paired coarticulation metrics must improve over v2.3 with a confidence
  interval below zero, not only a lower jerk number.
- At least 12 blinded raters over at least 20 paired clips; v3 naturalness
  preference at least 60% with the 95% interval excluding chance, plus animator
  edit-time and structural-reanimation reporting.

## Track I — image or multiview to identity and skin

### Current capability boundary

- Single view: 68 two-dimensional points, weak perspective and the first 20 GNM
  identity modes. This is a visible-face initializer, not metric recovery.
- Calibrated multiview: 2–12 views and up to 170 sparse-supported identity
  modes. Eye/dental tail modes remain neutral without their own evidence.
- Texture: provenance-aware RGB projection/inpainting at up to 1024 pixels. It
  is captured appearance, not illumination-free albedo or PBR skin.
- Validation: strong synthetic calibration tests, but no real consented subject
  with independent scan truth.

One RGB image cannot determine rear skull, ears, metric depth, eyes, dentition,
tongue, hair or illumination-free skin. Multiple calibrated views reduce that
ambiguity but do not become a scan merely by increasing count. Phone video is a
useful guided input only when metric camera motion or a scale target is present;
ordinary monocular structure from motion remains scale ambiguous.

### Selected dense qualification phase

1. Capture one consented neutral subject with at least five fit views and two
   frozen held-out views, at least 120 degrees of yaw, locked focus/exposure/
   white balance, a metric scale target and a repeat session.
2. Obtain an independent structured-light or equivalent metric scan that is not
   reconstructed from the evaluation photos.
3. Add camera-bundle v2 with raw board observations and recomputable calibration
   evidence instead of trusting summary numbers.
4. Add versioned dense GNM evidence: image point, triangle/barycentric anchor,
   confidence, facial part and provider provenance. Begin with audited 478-point
   anchors, profile silhouettes and reviewed masks.
5. Keep the 68-point solve as initialization; optimize dense reprojection,
   silhouette, part-aware residuals and GNM priors while leaving unsupported
   modes neutral.
6. Add fixed-camera held-out error, scan point-to-surface/normal error,
   capture-subset stability and per-part source-camera overlays.

Initial one-subject vertical-slice gates are calibration RMS at most 0.40 px,
held-out dense median below 2 px and p95 below 5 px per view/part, scan median at
most 1 mm and p95 at most 2.5 mm after rigid alignment without fitted scale,
normal median at most 8 degrees, and subset-stability median at most 0.5 mm.
Production likeness requires a diverse multi-subject set, repeat capture and
independent reviewers; one subject proves only the vertical slice.

### Appearance capture tiers

| Input | Defensible output | Forbidden claim |
| --- | --- | --- |
| Uncontrolled single RGB | preview color projection and inferred hidden areas | measured albedo, pores or rear texture |
| Guided calibrated RGB views | higher-coverage captured appearance with provenance | relightable PBR |
| RAW, color chart, cross/parallel polarization or calibrated multilight | diffuse/specular separation and normal/roughness candidate | pore detail beyond capture MTF |
| Scan/light-stage reference | measured geometry/material candidate plus artist cleanup | automatic production approval |

Later PBR gates include measured skin coverage at least 90%, seam delta-E2000 at
most 3, held-out median delta-E2000 at most 5, normal error at most 10 degrees,
and displacement RMSE at most 0.10 mm on a pore-qualified capture. An 8K file
name or upscaler cannot satisfy these gates. Hair remains a separate asset
system because GNM contains no hair representation.

## Track V — video to source-follow facial acting

### Current capability boundary

The pipeline stores exact PTS, 478 landmarks, 52 MediaPipe blendshapes, a face
transform and deterministic global quality values. It then uses one face-wide
confidence scalar for every region, a short causal filter and fixed mappings for
head translation/gaze. Observation v3 already measures regional pixel focus,
exposure, innovation and flow, but deliberately remains uncalibrated and does
not control retargeting. Audio repair therefore still makes ownership decisions
from the global tracker value.

This is a strong transport/provenance foundation, not production performance
capture. MediaPipe targets front-facing mobile AR and loses subtle dense
geometry; its score is not a calibrated probability of regional reconstruction
accuracy. Ordinary 24–30 fps footage can provide subtle-expression candidates,
but a microexpression claim needs roughly 120–200 fps, short exposure and FACS
onset/apex/offset evidence.

### VisualTrack v1

Create a provider-neutral, shadow-mode artifact beside Capture v1:

- exact source PTS and shot epoch;
- selected/missing/ambiguous/switched/re-entered subject state;
- camera intrinsics/extrinsics or explicit unknown;
- immutable shared identity reference plus per-frame expression/jaw/eyes/head;
- dense landmarks/samples with per-point visibility, residual and covariance;
- regional calibrated confidence for lips/contact, jaw, eyelids, gaze, brows,
  cheeks/nose and silhouette;
- separate tongue-visible state, evidence and motion source;
- provider/model/runtime/license and calibration-profile hashes.

Confidence must mean an estimated probability of meeting a named error bound,
not a renamed model score. Unsupported evidence stays unknown. Preview uses a
causal adaptive filter; final uses a per-shot bidirectional robust solve with
separate dynamics and preserved contact/blink/apex anchors. No temporal state
may cross a shot or subject epoch.

Only after calibration may regional ownership replace the existing global
confidence. High-confidence video owns visible head, gaze, asymmetry and lip
contour; audio repairs low-confidence lower-face intervals and supplies hidden
tongue only with `inferred` provenance.

Qualification includes mouth/eye p95 NME, aperture correlation, contact and
blink precision/recall, calibrated head/gaze error, occlusion AUROC/ECE, cut and
face-switch handling, long-take drift and blinded correction time. Required
adversarial footage includes profile/extreme pose, blur, occlusion, multiple
faces, entry/re-entry, cuts/dissolves, VFR/B-frames, dubbed audio, silence and
visible tongue.

## Native review and correction UX

The present native app is a secure job library and authenticated `WKWebView`
host. Production review needs a versioned `ReviewBundle v1` and one rational-PTS
cursor across:

1. source proxy with landmark/visibility/region overlays;
2. neutral selected-character GNM;
3. source-camera visual solve with residual heatmaps; and
4. final textured animation.

The workspace must expose visual base, audio repair, acting, authored correction
and physics as separate A/B revisions; provide mouth/tongue/eye closeups;
display ownership/confidence timelines; isolate base-color, normal,
displacement, roughness and specular layers; and create immutable correction
revisions with undo/redo. A `WKUserContentController` bridge should carry only
versioned cursor, layer, selection, correction and revision messages. The
portable GLB/Three.js viewer remains useful, but the production viewport should
eventually sample raw GNM controls in Metal so GLB morph compression is not the
measurement authority.

## Ordered execution and stop/go rules

1. **A1 articulation evidence** — diagnostic phone-span metrics and bounded
   mesh-batch sealed reconstruction are implemented. Next define reviewed articulation-
   state tiers and negatives, validated dental/alveolar/labiodental surfaces,
   protrusion, and a sealed character approval profile. Stop at diagnostic
   status until those exist and a real held-out set passes.
2. **A3-L local v3** — implemented and real-input executable. It remains a
   candidate until identical-noise supported-SDK parity, reviewed phone/FACS
   evidence and blinded perception/edit-time studies pass.
3. **V3-R1 supported-SDK parity** — requires a provisioned NVIDIA worker with
   identical window, recurrent state, emotion and exact noise tensors. Do not
   accept adapted v2.3 controls or matching seeds as parity evidence.
4. **V1.0 VisualTrack shadow lane** — add the provider-neutral artifact,
   shot/subject/tracking epochs, per-point uncertainty and regional unknowns.
   It has no motion authority until a later calibration phase passes.
5. **I0 verified identity capture** — one consented subject, calibrated camera
   bundle with at least five fit and two held-out views, repeat capture and an
   independent metric scan. Dense I1 fitting must wait for recomputable camera
   evidence and a frozen two-reviewer qualification contract.
6. **U1 ReviewBundle/native correction** — build around versioned raw evidence
   and corrections, not screenshots.
7. **A2/A3 oral solve and acting** — add mandible/teeth/tongue correctives and
   compose editable performance layers without changing reviewed articulation.
8. **I3 PBR/detail** — only after geometry and controlled capture pass.
9. **Q1 release qualification** — rights, deletion, device/load recovery,
   automatic gates, animator/naive studies and signed approval.

No phase may pass from unit tests alone. Each requires a real input, sealed
artifact reconstruction, adversarial/tamper checks, an independent reference
appropriate to the claim, and a full regression after the last source change.

## Licensing, data and runtime constraints

- GNM is Apache-2.0.
- NVIDIA SDK code and v3 weights have separate MIT/Open Model License terms;
  retain exact notices, use policy and legal approval. Audio2Emotion is a
  separately licensed component.
- MFA software is MIT, but each dictionary/acoustic model needs its own pinned
  license and revision.
- MICA, Metrical Tracker, DECA, SPECTRE, OpenFace, AV-HuBERT and common 4D face
  corpora are research references or separately restricted; none may silently
  become a shipping dependency.
- The official supported v3 SDK deployment is NVIDIA CUDA/TensorRT. The exact
  public ONNX weights run on Apple Silicon CPU, but that local candidate must
  remain labeled separately. Official parity and approval still require an
  authenticated supported-SDK worker and identical causal tensors.
- Faces, gaze, oral closeups, FACS and persistent subject identifiers are
  biometric/sensitive data. Capture needs explicit purpose, derivative/model
  rights, retention, deletion/export and revocation controls.
- The pinned MediaPipe Tasks runtime processes source frames on-device, but its
  official privacy notice says performance/utilization metrics are sent to
  Google and makes the app responsible for required informed consent. Shipping
  builds need an audited telemetry disclosure/disablement decision; on-device
  inference alone is not a complete privacy claim.

## Primary and official sources

- Google GNM: <https://github.com/google/GNM>
- NVIDIA Audio2Face paper: <https://arxiv.org/html/2508.16401>
- NVIDIA Audio2Face SDK: <https://github.com/NVIDIA/Audio2Face-3D-SDK>
- NVIDIA v3 model card: <https://huggingface.co/nvidia/Audio2Face-3D-v3.0>
- Montreal Forced Aligner: <https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner>
- MediaPipe privacy notice: <https://github.com/google-ai-edge/mediapipe#privacy-notice>
- Context-dependent coarticulation: <https://arxiv.org/abs/2507.20568>
- Imitator/contact-aware personalized speech: <https://openaccess.thecvf.com/content/ICCV2023/papers/Thambiraja_Imitator_Personalized_Speech-driven_3D_Facial_Animation_ICCV_2023_paper.pdf>
- Perceptual 3D speech-mesh metrics: <https://openaccess.thecvf.com/content/CVPR2025/html/Chae-Yeon_Perceptually_Accurate_3D_Talking_Head_Generation_New_Definitions_Speech-Mesh_Representation_CVPR_2025_paper.html>
- MediaPipe Face Landmarker: <https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python>
- MediaPipe blendshape model card: <https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf>
- Apple ARKit face tracking: <https://developer.apple.com/documentation/arkit/arfaceanchor>
- SPECTRE visual-speech reconstruction: <https://openaccess.thecvf.com/content/CVPR2023W/ABAW/papers/Filntisis_SPECTRE_Visual_Speech-Informed_Perceptual_3D_Facial_Expression_Reconstruction_From_Videos_CVPRW_2023_paper.pdf>
- MICA: <https://github.com/Zielon/MICA>
- DECA: <https://github.com/yfeng95/DECA>
- FLAME licensing: <https://flame.is.tue.mpg.de/modellicense.html>
- Polarized smartphone appearance capture: <https://openaccess.thecvf.com/content/CVPR2023/html/Azinovic_High-Res_Facial_Appearance_Capture_From_Polarized_Smartphone_Images_CVPR_2023_paper.html>
- One Euro filter: <https://gery.casiez.net/1euro/>
- CASME II high-speed microexpression methodology: <https://pmc.ncbi.nlm.nih.gov/articles/PMC3903513/>
