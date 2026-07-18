# Expanded GNM research, feasibility, product specification, and execution plan

Research and implementation snapshot: 2026-07-18
GNM revision studied: `3de70dfca5f3244620f44103c24b7cedc0dcb8b6`
Upstream: [google/GNM](https://github.com/google/GNM)
Product status: **working local research prototype; production approval withheld**

## Executive decision

Google's GNM Head 3.0 is a strong common geometry and animation endpoint for
this product. It is not, by itself, an audio-to-animation model, image
reconstructor, video tracker, texture estimator, renderer, or animator control
rig. Those interpretation layers must sit around GNM and must be evaluated
independently.

The right product form is a **local-first web application backed by FastAPI and
Python, with an interactive Three.js viewer in the browser**. The server owns
model inference, fitting, validation, provenance, and artifact generation. The
browser owns upload/capture guidance, review, synchronized playback, and
inspection. The CLI calls the same service functions for automation and
reproducibility.

The present repository already proves several important parts:

- learned Audio2Face motion runs locally on Apple Silicon and drives real GNM
  geometry through a dense, geometry-calibrated ARKit/tongue mapping;
- a procedural audio fallback, artifact pipeline, preview renderer, and
  adversarial temporal scorer exist;
- real single photos produce conservative visible-geometry GNM fits;
- static and low-rank animated GNM geometry export as seam-correct GLB;
- the browser can orbit, inspect, and media-synchronize those GLBs;
- frame-accurate MediaPipe VIDEO capture and fixed-identity GNM retargeting are
  implemented;
- shared-identity multiview fitting and multiview UV texture baking are
  implemented as deterministic, synthetic-verified cores.

Those facts do **not** prove the following:

- production-approved lip sync or emotional acting;
- a perfect, biometric, or metrically exact identity from one photograph;
- a verified real multiview portrait-to-textured-head application;
- faithful microexpression capture from ordinary monocular RGB video;
- a production-hardened, offline-bundled, cross-browser viewer.

The recommended route is therefore to retain GNM as the canonical rig, finish
the guided multiview/texture/video workflows around the implemented cores, and
separate engineering completion from quality approval. Production audio
approval additionally requires artist-authored GNM targets, independent
phone/contact annotations, human review, and license approval. No amount of
unit testing can manufacture those external gates.

## Claim vocabulary

This document uses four status labels deliberately:

| Label | Meaning |
|---|---|
| **Implemented** | Code exists in the current repository and has focused automated coverage. |
| **Real-input exercised** | The path has run on a real human audio, photograph, or detector input retained by the project. |
| **Experimental** | It is useful for research/review but lacks the data, calibration, or independent evaluation needed for a product quality claim. |
| **Production-approved** | All numerical, human-review, operational, security, and license gates have passed. Nothing in the current repository has this status. |

## 1. What GNM is

GNM Head 3.0 is a fixed-topology statistical model of a complete animatable
head. The checked-in asset contains:

| Property | Repository-grounded value |
|---|---:|
| Dense vertices | 17,821 |
| Triangles | 35,324 |
| Quads | 17,662 |
| Identity coefficients | 253 |
| Expression coefficients | 383 |
| Joints | 4: neck, head, left eye, right eye |
| Vertex groups | 46 anatomical/semantic masks |
| Sparse landmarks | one barycentric 68-point head definition |
| Geometry backends | NumPy, JAX, PyTorch, TensorFlow |
| Checked-in license | Apache-2.0 for GNM code and assets |

For identity `beta`, expression `phi`, joint rotations `theta`, and root
translation `tau`, the unposed mesh starts with a linear morphable-model sum:

```text
V_bind = V_template + identity_basis @ beta + expression_basis @ phi
```

Identity also moves the bind joint positions. GNM then applies
rotation-dependent pose correctives, evaluates the joint hierarchy, and uses
linear blend skinning. Identity and expression are therefore additive only in
bind space; camera fitting must not misuse head joints to explain a camera, and
animation exporters must preserve pose behavior rather than treating every
posed frame as an unrelated mesh.

### 1.1 Parameter regions

The identity basis is region-concatenated:

- 170 head modes;
- 3 eyeball modes;
- 80 teeth modes.

The expression basis in the actual v3 asset is:

- 100 left-periocular modes;
- 100 right-periocular modes;
- 150 lower-face modes;
- 32 tongue entries, including its mean;
- 1 pupil-dilation entry.

The included formal-definition PDF contains a stale 382-expression count; the
runtime asset and current API contain 383. The application correctly validates
the loaded asset rather than trusting the stale table.

### 1.2 Geometry, topology, and UVs

GNM supplies triangle- and quad-corner UV coordinates, left/right mirror
indices, skinning weights, pose correctives, joint regressors, and anatomical
vertex groups. UVs are stored per face corner while glTF stores UVs per render
vertex. A correct GLB exporter must duplicate a source vertex at every UV seam
and retain a mapping to the original GNM vertex. The implemented exporter does
this deterministically; the current v3 head becomes 18,437 render vertices
from 17,821 source vertices.

The UV layout has logical regions for skin, teeth/gums, tongue, eye interiors,
and eye exteriors. The bundled edge-flow image is a topology diagnostic, not a
person's albedo.

### 1.3 Landmarks and observability

Each GNM sparse landmark is a weighted combination of three source vertices.
This is differentiable and stable, but sparse landmarks do not observe the
entire model:

- identity modes 170:253 have zero motion at the 68 landmarks;
- expression modes 350:383, including tongue and pupil, have zero motion at
  those landmarks;
- depth, global scale, rear cranium, ears under occlusion, and surface detail
  remain weak or absent in one frontal image.

A sparse fitter must therefore lock eye/teeth identity and other unobservable
directions rather than inventing values. A low 2D landmark error is evidence of
2D correspondence, not proof of a correct 3D person.

### 1.4 Statistical parameters are not animator controls

The native names are PCA components such as `head_000` and
`lower_face_region_000`. GNM does not expose `jawOpen`, phonemes, visemes, FACS
action units, ARKit shapes, lip contact, or collision. It also has no physical
jaw joint. Its semantic expression decoder samples twenty broad labels such as
happy, pucker, stretch, wink, snarl, and tongue-center; it is a generator, not
an inverse audio/image/video model.

### 1.5 What GNM contributes and what it does not

GNM contributes a consistent topology, expressive regional bases, pose, eyes,
teeth, tongue, UVs, differentiable execution, and multiple numerical backends.
It does not contribute:

- audio, image, video, text, or emotion encoders;
- temporal coarticulation, blinking, gaze behavior, contact, or collision;
- learned texture, albedo, reflectance, lighting, hair, facial hair, or wrinkles;
- a production renderer or browser asset format;
- a photo-to-coefficients inverse model;
- a production facial-animation quality benchmark.

That boundary explains the application architecture:

```text
audio / photos / video
        -> perception and confidence
        -> calibrated semantic or geometric controls
        -> GNM identity + expression + pose
        -> validated mesh/GLB/video artifacts
        -> synchronized browser review
```

## 2. Feasibility summary

| Capability | Technical feasibility | Current state | Honest current claim |
|---|---|---|---|
| Clean audio to continuous mouth motion | High | Implemented and real-input exercised | Learned GNM speech animation prototype |
| Audio to production emotional performance | Medium | Manual A2F emotion exists; automatic heuristic is unvalidated | User-directed expressive intent, not validated emotion recognition |
| Single photo to GNM | High for visible coarse geometry; low for hidden/metric identity | Implemented and real-photo exercised | Single-view visible-geometry estimate |
| Guided multiview identity | High for observable GNM geometry with calibrated, neutral coverage | Shared sparse core implemented; synthetic only | Research fitter, not integrated capture product |
| Person-specific texture | Medium with sufficient calibrated views and controlled lighting | Provenance-aware baker implemented; synthetic only | Texture-baking core, not a verified person texture workflow |
| Monocular video to facial performance | High for coarse pose and 52 blendshape-like controls | Pipeline/API and licensed moving-human E2E implemented; perceptual/FACS validation pending | Experimental video retarget |
| Production microexpressions | Low-to-medium from commodity RGB; higher with TrueDepth/multiview/HMC | Not demonstrated | Requires a higher-quality capture tier |
| Interactive 3D review | High | Basic GLB/Three.js path implemented | Local inspection viewer, not release-hardened workstation |
| A perfect 3D clone from one image | Not identifiable | Not implemented and must not be promised | Impossible without priors/hallucination for unseen content |

## 3. Audio to lip sync and expressions

### 3.1 Why the original motion was rigid

The initial fallback held one of nine semantic mouth poses between Rhubarb cue
boundaries and used short boundary blends. It had no learned phonetic context,
true coarticulation, jaw mechanics, contact solve, dynamic acting, gaze, or head
motion. Smoothing that track more would reduce jumps but also erase the P/B/M
closures and F/V contacts that make speech readable.

Research systems support a contextual learned path:

- [Audio2Face-3D](https://arxiv.org/abs/2508.16401) predicts separate skin,
  tongue, jaw, and eye motion from acoustic context;
- [FaceFormer](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_FaceFormer_Speech-Driven_3D_Facial_Animation_With_Transformers_CVPR_2022_paper.pdf)
  demonstrates the value of contextual speech representations and temporal
  attention;
- [CodeTalker](https://openaccess.thecvf.com/content/CVPR2023/html/Xing_CodeTalker_Speech-Driven_3D_Facial_Animation_With_Discrete_Motion_Prior_CVPR_2023_paper.html)
  uses a learned discrete motion prior to reduce regression-to-the-mean motion;
- [EmoTalk](https://openaccess.thecvf.com/content/ICCV2023/html/Peng_EmoTalk_Speech-Driven_Emotional_Disentanglement_for_3D_Face_Animation_ICCV_2023_paper.html)
  separates speech content, emotion, identity, and intensity and explicitly
  penalizes velocity mismatch;
- [DiffPoseTalk](https://diffposetalk.github.io/) treats pose/style as a
  separate many-to-many signal rather than deriving it from mouth aperture.

### 3.2 Selected and implemented architecture

```text
audio
  -> FFmpeg mono 16 kHz normalization
  -> Audio2Face-3D v2.3.1 Claire through Swift/MLX
  -> 140 skin + 10 tongue + 15 jaw + 4 eye neural values at 30 fps
  -> official Claire geometry reconstruction
  -> bounded 52 ARKit + 16 tongue solve
  -> contact-aware temporal conditioning
  -> dense geometry-calibrated Claire/ARKit-to-GNM mapping
  -> [T,383] GNM expression + pose/nonverbal tracks
  -> finite GNM meshes, synchronized MP4, GLB, controls, and timeline
```

The current dense retarget is materially stronger than the earlier hand-written
semantic map. It robustly aligns Claire and GNM neutral point clouds, builds
confidence-weighted surface correspondence, and solves every released source
control against each bounded GNM expression region. It preserves independent
left/right controls and maps controls previously discarded by the semantic
fallback. It is deterministic and cache/fingerprint checked.

This calibration is still an automatic surface transfer, not an artist's
approved facial rig. Neutral correspondence can be locally wrong, GNM has no
physical jaw, and there is no lip/teeth/tongue collision model. The result must
remain `production_validated: false`.

### 3.3 Smoothness without mushiness

The correct temporal policy is control-aware:

- preserve or minimally filter blink, closure, press, roll, and jaw-contact
  controls;
- apply short zero-phase or velocity-adaptive filtering to slower controls;
- use contextual acoustic output for coarticulation instead of interpolating
  between isolated viseme labels;
- use region-preserving magnitude limits before an elementwise safety clip;
- evaluate velocity, acceleration, jerk, stationary transitions, and closure
  depth together;
- reject smoothing that improves jerk by weakening contact more than the
  allowed tolerance.

The learned real-input runs reduced frozen lower-face transitions to zero and
substantially reduced mouth-step, velocity, acceleration, and jerk compared with
the original fallback. These are engineering improvements, not independent
perceptual approval.

### 3.4 Emotion, LLMs, and nonverbal behavior

An LLM such as GPT or Claude can plan phrase-level acting: intended emotion,
intensity envelope, emphasis, pauses, and user-editable beats. It must not emit
phoneme or frame timing. Acoustic inference or a forced aligner must own lip
microtiming. A transcript-aware diagnostic can use [Montreal Forced
Aligner](https://montreal-forced-aligner.readthedocs.io/) or
[WhisperX](https://arxiv.org/abs/2303.00747), while recognizing that word
timestamps are not phone-contact ground truth.

Automatic speech-emotion recognition is ambiguous and domain-sensitive. The
current deterministic heuristic is explicitly unvalidated. Manual emotion is
safe as an authoring input; automatic emotion needs a separately licensed
model, a balanced evaluation set, confidence gating, and a neutral fallback.
Gaze, head motion, and blinks are only partly determined by audio and must
remain editable or capture-driven.

### 3.5 Production acceptance gates

Production audio approval requires, at minimum:

- at least ten varied real utterances and at least 100 independently annotated
  phone/contact events;
- median boundary error no greater than 45 ms and p90 no greater than 100 ms;
- P/B/M closure recall at least 90%;
- no material loss of closure/contact after temporal conditioning;
- all GNM arrays finite, bounded, deterministic where configured, and within
  one output frame of media duration;
- artist-reviewed GNM ARKit and speech-contact targets;
- blinded three-person lip-sync mean-opinion score at least 4/5;
- expression evaluation separated from lip timing;
- legal approval for model distribution and notices.

The repository has deterministic, geometry, temporal, activation-family, real
audio, and adversarial-shift/smoothing/static tests. It does not contain the
independent annotations, artist approval, or human MOS required by the last
gates.

## 4. Photo to 3D identity

### 4.1 Single-photo tier

The integrated single-photo path currently:

1. detects exactly one face with MediaPipe;
2. maps a versioned subset of its 478 points into the tested GNM-68 convention;
3. fits a robust weak-perspective camera, 10 then 20 head identity modes, and a
   small bounded nuisance expression;
4. reports normalized landmark error, stability under pixel perturbation,
   saturation, face size, pose, confidence, and caveats;
5. keeps unobservable eye/teeth identity, tongue, pupil, and final expression
   neutral;
6. exports OBJ, parameter NPZ, overlay, preview PNG, and seam-correct GLB.

This path has run on real photographs. Its output is a coarse estimate of the
visible face under a statistical GNM prior. It cannot recover the rear skull,
ears hidden by pose/hair, exact facial depth, metric scale, pores, albedo, hair,
or teeth/eye identity from a frontal image.

[DECA](https://arxiv.org/abs/2012.04012) demonstrates learned single-image
shape, detail, albedo, expression, and lighting inference; [EMOCA](https://emoca.is.tuebingen.mpg.de/)
shows why perceptual/emotion preservation needs more than landmarks and generic
photometric losses; [MICA](https://github.com/Zielon/MICA) is a useful metric
identity reference. Their released models and FLAME dependencies require
license and topology-transfer review before product use. They should be
benchmarks or optional initializers, not silently bundled dependencies.

### 4.2 Guided multiview tier

For best GNM identity accuracy, use a guided neutral capture rather than
unstructured uploads. A practical minimum is front, left/right three-quarter,
and left/right profile; preferred coverage adds shallow three-quarter views and
rear three-quarter/back views. Capture guidance should require:

- neutral expression, relaxed closed lips, eyes open, and no speech;
- fixed focal length/zoom, locked exposure/white balance, and saved intrinsics
  where possible;
- diffuse, stable lighting without moving cast shadows or specular hotspots;
- full head in frame, adequate resolution, sharpness, and no beauty filter;
- hair moved away from ears and face when geometry/texture coverage matters;
- capture roles recorded explicitly rather than inferred from filenames.

The implemented shared-identity core accepts per-view GNM-68 points, image
size, optional calibrated intrinsics, view role, confidence, and visibility. It
optimizes:

- one shared 253-value output with only the 170 landmark-observable head modes
  eligible to move;
- an independent perspective or weak-perspective camera per view;
- a small nuisance expression per view;
- robust landmark and view weights.

It unlocks identity deterministically in 20/40/80/120/170-mode stages, solves
in an SVD-observable subspace, reports condition/rank/saturation, and uses
leave-one-view-out prediction to reject a detectably mixed identity. Synthetic
front/three-quarter/profile tests cover near-exact intrinsic shape recovery
after the unavoidable similarity gauge, noise, occlusion, gross landmarks,
mixed people, determinism, and invalid input. The core materially outperforms
the existing single-view 20-mode fit on the tested synthetic identities.

It is not yet wired into a real guided-capture API/UI and has not been scored on
a rights-cleared multiview human dataset. Rear views usually lack facial
landmarks; using them for rear skull/ear geometry requires calibrated cameras,
silhouettes, dense correspondence, or photometric refinement beyond the sparse
core.

### 4.3 Accuracy path beyond sparse landmarks

The production multiview optimizer should keep the implemented sparse solve as
initialization and add, in stages:

- calibrated perspective cameras and distortion correction;
- dense facial landmarks and per-view visibility from a z-buffer;
- silhouettes for cheeks, cranium, and ears;
- shared identity with independent expression/lighting/camera nuisance;
- symmetry and PCA priors that weaken as observations improve;
- multiscale photometric consistency over skin-only masks;
- robust exclusion of hair, brows, eyelashes, specular highlights, background,
  and occluders;
- held-out-view validation and identity-consistency reporting.

Even this does not justify the word “perfect.” GNM's finite statistical basis
may not contain the exact person's geometry, and hair/facial hair are separate
assets. The product should report coverage, uncertainty, and model residual,
not a binary “clone succeeded” badge.

## 5. Person-specific texture

### 5.1 Correct objective

Texture capture should estimate a UV-space appearance map only where source
images directly observe the fitted surface. It must distinguish measurement
from completion. The implemented baker therefore produces mutually exclusive
provenance masks:

- `observed`: directly projected from a visible calibrated view;
- `mirrored`: copied through a declared symmetry rule;
- `inpainted`: filled locally without direct evidence;
- `generic`: default material where no person evidence exists.

The UI must never present mirrored, inpainted, or generic texels as if they were
photographed.

### 5.2 Implemented texture core

The deterministic CPU baker already supports:

- triangle-corner UV rasterization;
- calibrated pinhole cameras;
- z-buffer visibility and back-face rejection;
- view-angle and confidence weighting;
- multiple masked views;
- overlap-based gain/bias harmonization;
- seam-bounded local filling;
- confidence, source-view, overlap, and provenance maps;
- exhaustive input validation and deterministic output.

Synthetic tests cover gradient/checker fidelity, occlusion, backfaces, increased
coverage from multiple views, exposure mismatch, deterministic harmonization,
provenance exhaustiveness, seam-safe inpainting, and invalid inputs. GLB export
can embed a supplied UV texture without changing topology.

The missing work is material: the photo workflow does not yet convert fitted
multiview camera estimates into a real GNM texture bake, and no real person's
front/sides/back image set has passed a retained end-to-end test.

### 5.3 Production texture workflow

The integrated workflow should:

1. fit shared identity and cameras;
2. build skin/eye/teeth/tongue/hair masks separately;
3. undistort, linearize color, and estimate per-view exposure/white balance;
4. project only z-buffer-visible, front-facing texels;
5. downweight grazing angles, blur, highlights, shadow edges, occlusion, and
   inconsistent expressions;
6. harmonize overlap without erasing real local skin color;
7. blend seams in UV space while retaining observed provenance;
8. show coverage and confidence before allowing export;
9. embed the texture and provenance metadata in the GLB/job manifest.

Controlled cross-polarized or diffuse capture is the best path to albedo-like
skin appearance. Ordinary photographs combine albedo, illumination, camera
response, makeup, and specular reflection; a baked result is appearance under
those conditions, not a physically pure material. Eyes, teeth, tongue, hair,
eyebrows, eyelashes, and facial hair need dedicated materials or geometry.

## 6. Video-driven facial performance

### 6.1 Implemented experimental pipeline

The video path uses the official [MediaPipe Face Landmarker](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarker)
in VIDEO mode. It preserves exact source presentation timestamps, decodes one
RGB frame for each probed timestamp, and records:

- 478 3D-relative facial landmarks;
- 52 named blendshape scores;
- one facial transform matrix per detected frame;
- detector presence/quality and missing-frame state;
- source/model hashes, FFmpeg/MediaPipe versions, and commands.

Retargeting holds one identity fixed. When the configured Claire geometry
assets are available, 51 of MediaPipe's 52 named outputs (all except its
synthetic `_neutral` channel) enter the same dense geometry-calibrated
Claire-to-GNM map used by learned audio. `mouthClose` is now quarantined from
that expression map because the released Claire row opens this character;
tracked inner-lip geometry instead drives the same bounded character contact
solve used by audio. The older low-dimensional semantic
prototype map remains an explicitly labeled fallback. Head pose/translation
and eye directions remain separate GNM joint controls. A short,
confidence-aware causal filter preserves all high-confidence blink and
mouth-contact samples exactly; missing frames are held briefly and then decay
rather than being fabricated observations. The pipeline saves the exact
calibration/hash, raw capture, retargeted controls, PTS-checked H.264/AAC proxy,
animated or fail-closed static GLB, metrics, and caveats through
`POST /api/video`.

Tests cover exact FFmpeg timestamps, schema immutability, blink/contact
preservation, pose/translation/eye motion, fixed identity, provenance,
serialization, and both static and genuinely moving real-face inputs. The
opt-in, checksum-pinned CREMA-D actor fixture passed the full HTTP path with
67/67 detected frames, 93.9% retained non-contact filter variation, exact
fast-control filter passthrough, 1.23 ms maximum source/proxy timestamp difference, and a
GNM animated GLB with 0 validator errors and 0 warnings. Its all-frame
landmark reconstruction p95 was 0.116 mm. The single geometry-derived closure
event is retained, and all 14 high-confidence contact frames reach the
character's calibrated 0.00307 interocular seal instead of remaining at its
0.04183 neutral gap. Review found subject-specific
neutral coefficient bias had activated a region safety bound on 41/67 frames;
the fixed pipeline records a timestamp-based 0.2 s neutral baseline, routes
gaze only to baseline-relative eye joints, preserves 93.9% of source temporal variation,
and activates no region bound on the retained clip. Inputs without a genuine
neutral lead-in are explicitly caveated. The current one-sided correction also
clips 41.97% of non-gaze residual samples below the selected reference and now
reports that loss explicitly. These measurements prove transport,
tracking, retarget execution, export, and synchronization—not perceptual or
microexpression accuracy. Subjective animator review and a frame-labeled FACS,
gaze, and lip-contact benchmark remain pending.

### 6.2 What monocular RGB can and cannot capture

Commodity RGB video can provide useful timing for lip motion, blinks, broad
brow/cheek motion, gaze proxies, and rigid head motion. It remains weak for:

- metric translation and depth;
- occluded lip/tongue/teeth contact;
- subtle asymmetric muscle motion outside the 52-control schema;
- wrinkles, blood-flow/color changes, and high-frequency skin deformation;
- gaze under eyelid/iris ambiguity;
- fast motion blur, rolling shutter, low light, profile self-occlusion, and
  hand/prop occlusion.

Calling 52 coarse controls “microexpression capture” would overclaim. A higher
quality tier should use a depth-capable front camera, synchronized multiview or
head-mounted cameras, higher frame rate and shutter speed, controlled lighting,
an identity-specific calibration range, and artist-reviewed solve targets.

### 6.3 Audio/video fusion

For a performance video with sound, source video remains the authority for
head pose, gaze, blinks, and visible acting. The acoustic model can supply a
confidence-gated mouth/tongue/contact prior when the mouth is blurred or
occluded. Fusion must occur in named control/contact space and preserve source
timestamps. It must not average two tracks blindly: visible video closures
should win at high tracking confidence, acoustic contact should assist at low
visual confidence, and disagreements must be exposed in diagnostics.

## 7. Interactive 3D viewer and asset format

### 7.1 Why GLB and Three.js

Binary glTF is the correct delivery format because it embeds geometry,
materials, textures, morph targets, and animation in one allowlisted artifact.
The [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
defines the interoperable contract, and the official [Khronos
Validator](https://github.com/KhronosGroup/glTF-Validator) is the conformance
gate. OBJ remains useful as a legacy geometry download but does not carry the
required animation contract.

Three.js is selected over a generic `<model-viewer>` wrapper because the app
needs exact media-clock control, wireframe/topology inspection, diagnostic
overlays, component visibility, and error recovery. Its license is
[MIT](https://github.com/mrdoob/three.js/blob/dev/LICENSE). `OrbitControls`,
`GLTFLoader`, and `AnimationMixer` provide the needed primitives without the
larger editor surface of a full scene engine.

### 7.2 Implemented viewer foundation

The repository currently has:

- deterministic seam-correct static GLB export with source-vertex mapping;
- embedded supplied textures or anatomical debug colors;
- deterministic low-rank vertex-animation factorization;
- automatic selection of the smallest morph rank that passes mesh and landmark
  reconstruction limits;
- standard animated GLB output and a static fallback when a target cap cannot
  pass;
- audio and video media-clock synchronization by sampling a paused
  `AnimationAction.time` directly, including backward seeks after end-of-clip;
- an allowlisted FastAPI viewer endpoint;
- orbit, zoom, camera reset, exposure, surface/wireframe modes, and analytic
  lighting.

The viewer now serves a checksum-pinned, versioned Three.js `0.183.2` bundle
locally, retains the MIT license, applies a restrictive CSP, bounds navigation,
exposes live/fallback status, makes its canvas keyboard-focusable, and disposes
GPU resources on page exit. Focused exporter/API tests cover those contracts.
It is not yet fully cross-browser, context-loss-tested, or memory-profiled.

### 7.3 Shipping viewer requirements

The remaining production workspace must add:

- source/fitted/textured/current-pose selection;
- component toggles for skin, eyes, teeth/gums, and tongue;
- shaded, flat, normals, UV-debug, texture, and wireframe modes;
- front/side/three-quarter presets, orthographic mode, fullscreen, fit/reset;
- semantic timeline slider, waveform, mouth/contact, expression, video quality,
  and warning lanes;
- keyboard/touch equivalents, fallback text, live status, reduced-motion
  behavior, and WCAG AA contrast;
- PNG/MP4 fallback for unavailable WebGL or invalid animation;
- abort/disposal on job switches and context-loss recovery;
- validator results, reconstruction error, texture coverage, calibration hash,
  backend, and warning display.

glTF only recommends that generic clients support at least eight morphed
attributes. The app can test and support up to its 32-target cap in the pinned
Three.js version, but third-party viewer compatibility above eight targets must
remain best effort.

## 8. Application product specification

### 8.1 Form-factor justification

A local web application is the best first product because:

- Python/Swift own the current inference stack and should not be ported into
  browser JavaScript merely for packaging;
- Three.js gives a portable GPU viewer on macOS, Windows, and Linux;
- local execution protects face/audio/video inputs by default and avoids an
  immediate account, upload, storage, and consent backend;
- the same FastAPI service supports browser, CLI, tests, and later desktop-shell
  packaging;
- immutable job directories make every result reproducible and inspectable;
- expensive work can later move behind a queue without changing artifact or
  viewer contracts.

A native DCC plugin is a later integration target, not the primary application:
it would improve artist authoring but fragment capture and review across host
versions. A cloud service is also later; it introduces significant privacy,
consent, abuse, storage, isolation, and GPU-operations requirements.

### 8.2 User journeys

#### Audio

1. Upload or record audio.
2. Choose `learned`, `auto`, or `fallback`; show exact model/backend and license
   caveat.
3. Select manual emotion/intensity or leave automatic acting neutral unless a
   validated model has sufficient confidence.
4. Review synchronized media/3D, waveform, mouth/contact activity, warnings,
   and reconstruction metrics.
5. Export GLB, MP4, raw neural controls, ARKit/tongue controls, GNM controls,
   timeline, and calibration metadata.

#### Single photo

1. Upload a clear neutral image.
2. Run quality checks and display mapped/fitted landmarks.
3. Reject multiple faces, severe pose/rotation, tiny faces, cropping, or poor
   fit with typed recovery guidance.
4. Review the untextured fitted GNM with the single-view caveat visible.
5. Export GLB, OBJ, parameters, overlay, and preview.

#### Guided multiview and texture

1. Choose geometry-only or geometry-plus-texture.
2. Capture required roles with a progress ring around the head.
3. Gate each view on identity consistency, sharpness, face size, pose,
   expression, exposure, and coverage; offer a specific retake instruction.
4. Fit shared identity/cameras and show held-out residuals and rejected views.
5. Bake texture, show observed/mirrored/inpainted/generic coverage, and allow the
   user to disable non-observed completion.
6. Review neutral/textured/wireframe/UV modes and export the full audit package.

#### Video performance

1. Upload a face performance video and optionally select an existing fitted
   identity.
2. Show source timing, face-presence, longest gap, and quality diagnostics.
3. Review synchronized source video and GNM animation with head/eyes/mouth lanes.
4. Permit correction of baseline pose and identity, then reprocess
   deterministically.
5. Export source proxy, capture schema, performance controls, GLB, and report.

### 8.3 Result contract

Every terminal job should expose:

- immutable job ID, source hash, configuration, dependency/model versions, and
  UTC timestamps;
- status and typed warnings/errors;
- explicit `production_validated` state;
- model dimensions and coordinate system;
- backend, calibration hash, coverage/confidence, quality metrics, and caveats;
- allowlisted artifacts with media type, bytes, and SHA-256;
- a viewer contract naming GLB, media clock, duration/fps, animation status, and
  reconstruction error;
- enough raw observations and parameters to reproduce or audit the result.

Artifacts must be written atomically. User uploads need encoded-byte, decoded
pixel, duration/frame, and path controls. Job artifacts are server-generated;
the viewer must never resolve arbitrary filesystem paths or accept an uploaded
GLB as trusted executable content.

### 8.4 Nonfunctional requirements

- Local-first and offline after bootstrap; production must bundle viewer assets.
- Deterministic output for identical inputs/configuration where stochastic
  acting is disabled or seeded.
- Fail closed on nonfinite geometry, timestamp mismatch, invalid topology,
  reconstruction-limit failure, mixed identity, and unallowlisted artifacts.
- Preserve source media timing; media is the viewer master clock.
- Provide static PNG/MP4 and downloadable controls when WebGL fails.
- Never call a low landmark residual proof of identity, texture, or lip-sync
  quality.
- Treat biometric face data, audio, and performance video as sensitive; do not
  upload or retain externally without explicit consent and policy.
- Attach provenance and require consent/usage notices for identity cloning and
  generated performance; add watermark/signing policy before public sharing.

## 9. Phased shipping and verification plan

Every phase follows the same strict loop:

1. build only the declared scope;
2. review correctness, architecture, limits, licenses, error paths, and claims;
3. run focused unit/integration tests plus real-input end-to-end tests;
4. inspect visual artifacts and numerical reports;
5. fix every failure and rerun the focused and full regression suites;
6. advance only when the exit gate passes without a waiver.

### Phase 0 — pin upstream truth and claims

**Status:** implemented and historically verified.

**Milestone:** reproducibly load the pinned GNM asset and expose its real
dimensions, regions, topology, landmarks, UVs, and limitations.

**Dependencies:** pinned GNM repository and model assets; Python environment.

**Tests/gate:** upstream suite plus separately discovered fitting/visualization
tests; application asset assertions; compile/diff hygiene. Record test-discovery
gaps and macOS OSMesa limitations rather than hiding them.

### Phase 1 — common GNM, artifacts, and conservative single-photo foundation

**Status:** implemented; real-photo exercised.

**Milestone:** one validated adapter/control rig and one job/artifact contract
serve CLI, API, and browser. A real photo yields neutral fitted geometry and
confidence, never a “clone” claim.

**Dependencies:** GNM NumPy backend, MediaPipe task asset, OpenCV, SciPy,
FastAPI, FFmpeg.

**Tests/gate:** official-versus-compact landmark parity; region isolation;
finite/topology/export tests; seeded 20-mode recovery; real portrait and
astronaut runs; blank, multiple, rotated, tiny, and cropped failure cases;
API/CLI parity.

### Phase 2 — learned audio and dense retarget foundation

**Status:** engineering implementation complete for the research prototype;
real audio exercised; production quality gate pending.

**Milestone:** learned Claire inference is the preferred local backend, with a
typed fallback, full 52+16 control audit, dense geometry-based GNM calibration,
contact-aware temporal conditioning, synchronized MP4/GLB, and honest quality
state.

**Dependencies:** FFmpeg; Swift/MLX `speech-swift`; official Claire package;
NVIDIA Open Model License review; Rhubarb fallback.

**Tests/gate:** real human and emotional speech; deterministic raw output;
silence/rest; coefficient dimensions/timestamps; all named control families;
synthetic and released-asset calibration; asymmetric/unmapped controls;
finite meshes; A/V frame parity; adversarial ±2/±4-frame shifts, excessive
smoothing, static, constant-open, cue permutation, and silence motion.

**Production exit still missing:** artist-authored GNM targets, independent
phone/contact annotations, blinded MOS, emotion validation, collision/contact
review, and legal approval.

### Phase 3 — conformant GLB and basic interactive viewer

**Status:** core implemented and integrated for image/audio; release hardening
pending.

**Milestone:** every successful geometry/animation job has an allowlisted GLB
or an explicit static fallback, with exact source-vertex mapping and measured
reconstruction error.

**Dependencies:** triangle-corner UVs, trimesh/Pillow, glTF 2.0, pinned Three.js,
Khronos validator.

**Tests/gate:** UV seam round trip; texture embedding; static and real-GNM
animated export; minimal passing rank; fail-closed rank cap; timestamp and
accessor validation; allowlisting and media-clock API tests; real image/audio
browser orbit/play/pause/seek; zero validator errors. Before production, bundle
assets locally and pass Chromium/WebKit, mobile, keyboard, context-loss,
WebGL-unavailable, cleanup, and memory tests.

### Phase 4 — moving-human video performance

**Status:** capture, dense/fallback retarget, API, video-clock viewer, and one
retained licensed moving-human E2E are implemented. The product phase remains
experimental pending perceptual, occlusion, and labeled performance tests.

**Milestone:** a real performance video drives fixed-identity GNM mouth, eyes,
head pose, and translation at exact source times and remains seek-synchronized.

**Dependencies:** MediaPipe VIDEO model, FFmpeg/FFprobe, calibrated retargeter,
licensed moving-face fixture, Phase 3 viewer.

**Tests/gate:** variable/exact PTS; rotation metadata; real moving face with
speech, blinks, asymmetry, pose, and short occlusion; missing-frame hold/decay;
contact latency; identity invariance; head/eye orientation; source/3D drift no
greater than one source frame; seek/start/middle/end; actual browser playback;
manual visual review. A static image encoded as video does not satisfy this
gate.

### Phase 5 — guided multiview identity application

**Status:** robust sparse core implemented and synthetic-verified; capture,
service/API, real data, and dense refinement remain.

**Milestone:** guided front/three-quarter/profile inputs produce one shared GNM
identity with rejected-view handling, observable-rank and metric caveats, and a
held-out validation report.

**Dependencies:** capture protocol/UI, per-view landmark confidence/visibility,
intrinsics when available, rights-cleared multiview face set, Phase 1 artifacts.

**Tests/gate:** exact and noisy synthetic recovery; weak/perspective cameras;
occlusion and gross landmark robustness; unobservable modes locked; mixed-ID
rejection; determinism; real same-person view consistency; deliberate different
person; held-out-view error; comparison against single-view 20-mode and neutral
mean. Then add silhouettes/dense refinement and rerun every gate.

### Phase 6 — multiview person texture application

**Status:** deterministic baker implemented and synthetic-verified; not
integrated or real-input validated.

**Milestone:** accepted multiview fits yield an embedded person-specific skin
texture plus observed/mirrored/inpainted/generic confidence maps.

**Dependencies:** Phase 5 cameras/mesh, segmentation masks, color pipeline,
real controlled multiview capture, GLB exporter/viewer.

**Tests/gate:** synthetic gradient/checker, z-buffer/backface, overlapping
views, exposure harmonization, seam isolation, provenance exhaustiveness, and
determinism; then real front/sides/rear coverage, seam and color review,
held-out-view rendering, no projection onto eyes/teeth/hair, embedded-GLB round
trip, and coverage thresholds. Do not fill unseen areas silently to make the
texture look “complete.”

### Phase 7 — unified production-candidate workspace

**Status:** next integration phase.

**Milestone:** audio, single/multiview photo, texture, and video share guided
workflows, one review workspace, one job schema, provenance, accessible
diagnostics, and exports.

**Dependencies:** Phases 2–6, local viewer bundle, job queue design, security
review, user/consent policy.

**Tests/gate:** complete real-input matrix through CLI/API/browser; restart/job
recovery; concurrent queue isolation; upload bombs and malformed media;
artifact traversal/allowlist; mobile/keyboard/reduced motion; WebGL/media errors;
cross-job resource cleanup; full dependency/bootstrap run on clean macOS and
Linux; performance budgets based on measured devices.

### Phase 8 — production approval and release

**Status:** blocked on external data, artistic, legal, and human-review inputs;
not simulated by current code.

**Milestone:** a release candidate passes every pipeline's declared numerical
and perceptual quality bar, security/privacy review, notices, and operational
load/recovery tests.

**Dependencies:** rights-cleared annotated speech and multiview/video data;
facial animator; representative human review panel; product counsel; deployment
and incident owners.

**Tests/gate:** phone/contact corpus and MOS; multiview benchmark and held-out
identity review; real texture coverage/seam study; moving-video performance and
microexpression tier evaluation; fairness slices; consent/deletion/audit flow;
license/notice bundle; load, crash recovery, backup/retention, and canary. Only
after all of these pass may `production_validated` become true for the specific
validated configuration.

## 10. Implemented status versus remaining claims

| Area | Present evidence | Missing evidence or work |
|---|---|---|
| GNM model integration | Real v3 asset, official and application tests, exact dimensions | Upstream packaging/render defects remain external |
| Audio inference | Real LibriSpeech/RAVDESS learned runs; compiler-v9 real artifact with exact face-local mouth-step and silence-hygiene checks passing; bounded contact anchors preserve 3/3 inferred contacts; deterministic finite exports | 32/211 limiter interventions, independent phone labels, MOS, artist rig approval, legal release review |
| Dense audio retarget | Official Claire asset calibration, independent control preservation, raw-jaw/eye retention, spatial contact correction | Ground-truth GNM performance, character jaw/dental calibration, collision/contact approval |
| Single photo | Two real-photo fits and typed negative cases | Hidden geometry, metric accuracy, texture, perceptual identity benchmark |
| Multiview identity | Unified UI/API plus shared bounded solver, observability/mixed-ID gates, and retained calibrated five-view synthetic positive control | Rights-cleared calibrated real-person capture, held-out real geometry, and independent likeness review |
| Texture | Retained calibrated synthetic multiview bake with 53.05% direct coverage, exhaustive per-component provenance, and valid textured GLB | Real-person capture, segmentation/de-lighting, intrinsic albedo/BRDF separation, and artist seam/likeness review |
| Video | Exact timing, synthetic motion, dense retarget, CREMA-D moving actor HTTP/viewer E2E, 51/52 structural control coverage with inverted `mouthClose` quarantined, landmark-derived character seal attainment, proxy PTS check | Independent perceptual/FACS/lip-contact evaluation, labeled neutral/bidirectional calibration, occlusion fixture, subject-specific high-quality tier |
| GLB/viewer | Static, textured, audio-animated, and video-animated live browser paths; checksum-pinned local Three.js including transitive core; embedded-texture CSP fix; four final GLBs with 0 validator errors/warnings | Mobile matrix, accessibility audit, WebGL context-loss recovery, memory/load testing |
| Full expanded regression | 148 application tests, 7 Swift tests, 278 official GNM tests, 60 nested fitting tests, 21 camera/color passes plus 3 skips, final GLB ledger, and live browser checks all rerun after integration | Production approval datasets, human panels, legal/consent, deployment/load/recovery evidence remain external release gates |

The older 61-test release statement has now been superseded by the final
148-test integration run. This proves the implemented research-prototype paths,
not the independent production-quality or legal gates listed in the same row.

## 11. Dependencies, licenses, and adoption decisions

| Dependency/research | Role | License/constraint decision |
|---|---|---|
| [Google GNM](https://github.com/google/GNM) | Canonical head geometry/rig | Apache-2.0 checked-in code/assets; retain notices and pinned source install |
| [NVIDIA Audio2Face-3D repository](https://github.com/NVIDIA/Audio2Face-3D) | Learned audio architecture/runtime reference | SDK MIT and training code Apache-2.0; platform-specific requirements |
| [Claire model package](https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire) | Current learned weights, geometry, named targets | NVIDIA Open Model License; the checksum-pinned model card and an explicit provenance notice are retained beside the assets, but redistribution/notice compliance still requires product counsel |
| [speech-swift](https://github.com/soniqo/speech-swift) | Apple-Silicon Swift/MLX execution | Apache-licensed third-party port; pin exact version and retain parity/real-input tests |
| Rhubarb | Deterministic offline fallback cues | MIT per the inspected dependency; keep as fallback/diagnostic, not ground truth |
| [MediaPipe Face Landmarker](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarker) | Photo/video landmarks, transforms, blendshapes | Pin package/model hashes; review package and model-asset redistribution terms separately |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Opt-in real moving speech-performance test | Pinned official revision; ODbL database/DbCL contents; retain attribution and review performer/publicity/biometric use before redistribution |
| [Three.js](https://github.com/mrdoob/three.js/blob/dev/LICENSE) | Browser viewer | MIT; official npm `0.183.2` archive and six required runtime modules (including the transitive `three.core.js`) are checksum-pinned, served locally under a versioned allowlist, and the exact license is retained; npm reported `0.185.1` on 2026-07-18, so upgrading remains an explicit compatibility task |
| [glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) | Asset/animation contract | Validate generated artifacts with the official Khronos validator |
| DECA/EMOCA/MICA/FLAME ecosystem | Research benchmarks or possible initializer | Do not bundle until code, checkpoint, training-data, and FLAME terms are reviewed for the intended use |
| RAVDESS test subset | Optional emotional test evidence | CC BY-NC-SA in the existing workflow; not a commercial application asset |

Repository code licenses do not automatically license model weights or training
data. A commercial training set must be rights-cleared independently.

## 12. Additional use cases unlocked

Once the common identity, animation, texture, viewer, and provenance contracts
are complete, the same stack supports:

- animator previsualization and contact/retarget diagnostics;
- identity-preserving dubbing and localization with user-approved acting edits;
- game/avatar GLB export and cross-rig ARKit/MediaPipe/GNM conversion;
- telepresence and asynchronous avatar messages;
- facial-performance review with source/3D time-locked diagnostics;
- synthetic GNM data generation with exact geometry, landmarks, pose, and
  coefficient ground truth;
- accessibility avatars and privacy-preserving stylized representation;
- dataset/algorithm benchmarking across audio, single-view, multiview, and
  video methods;
- supervised authoring of custom expression, viseme, and corrective libraries;
- quality-control tools that find dropped frames, timing drift, saturation,
  topology inversion, texture holes, or view inconsistency.

Public identity cloning or performance transfer also creates impersonation and
consent risk. Productization must include authorization, provenance, retention
controls, export disclosure, and abuse-response policy alongside technical
quality work.

## 13. Final definition of done

The expanded objective is complete only when all of the following are proven by
current artifacts and rerun evidence:

1. Real audio produces synchronized, finite, inspectable GNM animation through
   the learned and fallback paths.
2. Production audio claims are withheld until artist calibration, independent
   phone/contact timing, MOS, expression, legal, and collision/contact gates
   pass.
3. A real single photo produces a conservative visible-geometry fit with clear
   uncertainty and no hidden-geometry claim.
4. A real guided multiview session produces one shared identity, rejects a
   deliberately mixed person, improves held-out geometry over single view, and
   reports observability/scale caveats.
5. Real multiview photos produce a textured GLB with measured coverage and
   exhaustive observed/mirrored/inpainted/generic provenance.
6. A real moving-human video drives lips, expressions, blinks, eyes, head pose,
   and translation at source timestamps, with missing-frame and confidence
   behavior tested.
7. The viewer shows fitted, textured, audio-animated, and video-animated GLBs,
   remains media-synchronized, and passes desktop/mobile/accessibility/error/
   cleanup tests with static fallbacks.
8. CLI, API, browser, artifacts, and warnings agree for the same inputs.
9. The complete expanded application, native runner, upstream GNM, validator,
   real-input E2E, and browser suites pass after the final change.
10. Every release dependency, model weight, fixture, notice, consent flow, and
    retained asset is approved for its actual use.

Until items 4–10 are satisfied, the project remains an unusually capable and
well-instrumented research prototype—not a flawless or production-approved
face-cloning system.
