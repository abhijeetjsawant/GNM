# Production audio-driven GNM facial animation

Status: implementation decision and phased execution specification
Date: 2026-07-18
GNM revision studied: `3de70dfca5f3244620f44103c24b7cedc0dcb8b6`

## Executive decision

The current Rhubarb-to-nine-pose driver cannot become production quality by adding a stronger low-pass filter. It has no learned coarticulation, no phonetic context, no dynamic emotional phrasing, no jaw joint, no contact constraints, and no nonverbal motion model. More smoothing would hide some jumps while also weakening the closures that make `/p/`, `/b/`, and `/m/` readable.

The recommended architecture has two tiers:

1. **Learned local path:** NVIDIA Audio2Face-3D v2.3.1 Claire, running through the Swift/MLX Apple Silicon port, produces continuous actor-specific face, tongue, jaw, and eye motion. Reconstruct the Claire face motion with NVIDIA's released 140-shape geometry basis, solve 52 bounded ARKit weights using the released Claire targets and solver configuration, then retarget those semantic weights into GNM.
2. **Deterministic fallback:** replace hold-and-jump cues with dominance-based coarticulation, contact-aware temporal filtering, phrase-level prosody, region-aware emotion composition, blinks, and small head beats. This is a robust offline fallback and a useful diagnostic baseline, but it must not be marketed as equivalent to a facial-performance model.

For a commercial-quality product, the learned path should be the default. The final GNM retarget remains an approximation until the project has artist-authored GNM ARKit/viseme targets or rights-cleared paired audio and GNM motion for fine-tuning.

## Why the current result looks rigid

The diagnosis was visible directly in the pre-change implementation retained
as the baseline for this execution:

- `animation.py` chooses exactly one of A-H/X at every frame and holds it until a boundary.
- Only a raised-cosine blend of at most 70 ms is applied at adjacent boundaries.
- One emotion vector is held over the entire clip, apart from a 300 ms fade at its ends.
- Every head/neck/eye joint rotation and the root translation are identically zero.
- `rig.py` builds nine speech poses from a 20-label expression decoder. These are semantic expression samples such as `stretch_face` and `pucker`, not speech-captured visemes.
- GNM v3 exposes 150 unnamed lower-face statistical components and 32 tongue components, but no jaw joint, phoneme controls, ARKit controls, collision model, or temporal animation model.

The real verified controls quantify the perceptual problem:

| Real clip | Frames | Cues | Exactly stationary frame transitions | Head motion |
|---|---:|---:|---:|---|
| LibriSpeech, 8 s | 240 | 50 | 30.5% | none |
| RAVDESS anger, 4.1 s | 124 | 14 | 49.6% | none |

The pattern is therefore **hold, jump, briefly blend, hold**. The anger clip is especially rigid because a constant emotion vector dominates nearly half of its transitions.

## What production quality means

Production quality is not a single smoothness score. It requires all of the following:

- **Articulation:** closures, lip rounding, aperture, lower-lip/teeth contact, and visible tongue events agree with speech.
- **Timing:** motion anticipates the acoustic event naturally, does not drift over long clips, and remains within an audiovisual tolerance of one rendered frame.
- **Coarticulation:** the pose depends on surrounding sounds; consonants do not look like isolated cards.
- **Temporal behavior:** no mechanical holds, spikes, tremor, or over-smoothed mush.
- **Expression:** affect changes over phrases and emphasis beats without corrupting lip readability.
- **Whole performance:** blinks, eyes, and subtle head motion prevent a mask-like result.
- **Retarget fidelity:** motion survives the Claire/ARKit-to-GNM conversion without saturation, identity drift, inverted triangles, or mouth/tongue artifacts.
- **Control and auditability:** an artist can override emotion/intensity and inspect every generated track.

## Repository constraints

GNM is a strong endpoint rig, not an audio animation system:

- 17,821 vertices and 35,324 triangles.
- 253 identity coefficients.
- 383 expression coefficients: 100 left eye, 100 right eye, 150 lower face, 32 tongue, and 1 pupil component.
- Four joints: neck, head, left eye, and right eye. There is no jaw joint.
- The expression basis is additive and differentiable, followed by pose correctives and linear blend skinning.
- The semantic decoder has 20 coarse labels. It has no jaw-open, lip-contact, phoneme, anger, sadness, or fear class.
- The fitting utility projects a same-topology 3D target into GNM. It does not create correspondence between an unrelated actor topology and GNM.

Consequently, the correct boundary is:

```text
audio understanding / learned motion
        -> semantic or calibrated motion controls
        -> constrained GNM retarget
        -> GNM geometry and rendering
```

## Evidence from current systems and research

### NVIDIA Audio2Face-3D

[NVIDIA's Audio2Face-3D paper](https://arxiv.org/abs/2508.16401) describes a production-oriented system trained from multi-camera 4D capture, with separate skin, tongue, jaw, and eye outputs, real-time inference, blendshape solving, and optional Audio2Emotion. Its v2.3 regression model consumes approximately 0.52 seconds of audio for a frame; v3 uses a one-second context and diffusion to emit a 30-frame block. NVIDIA reports direct mesh, joint, and blendshape workflows and time-keyed emotion control.

The [official repository](https://github.com/NVIDIA/Audio2Face-3D) publishes the SDK under MIT, the training framework under Apache-2.0, and v2.3/v3 model weights under the NVIDIA Open Model License. The official CUDA SDK currently requires Windows or Linux, CUDA 12.8+, TensorRT 10.13+, and an NVIDIA GPU, so it cannot run natively in this macOS application.

The Apache-licensed [speech-swift Audio2Face module](https://github.com/soniqo/speech-swift) supplies a hand-written Swift/MLX forward pass for Apple Silicon. It parity-tests against NVIDIA ONNX fixtures and emits timestamped model coefficients. Claire v2.3.1 emits 140 skin, 10 tongue, 15 jaw, and 4 eye values at 30 fps. The port is new and third-party, so parity and real-input tests are required locally; its availability does not remove NVIDIA model-license obligations.

The official [Claire model package](https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire) is ready for commercial and noncommercial use under the NVIDIA Open Model License. It includes:

- the learned 140-dimensional skin and 10-dimensional tongue geometry bases;
- a 52-pose named ARKit skin target library and 16 tongue targets;
- solver masks and NVIDIA's L1, L2, temporal, and symmetry parameters;
- a 40M-parameter Wav2Vec2/CNN regression network;
- explicit conditioning labels for amazement, anger, cheekiness, disgust, fear, grief, joy, out-of-breath, pain, and sadness.

This is the shortest credible route from learned audio motion to GNM on the current machine.

### UniTalker and contextual research models

[UniTalker (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05747.pdf) combines heterogeneous vertex, FLAME, and ARKit datasets through separate heads, PCA balancing, model warm-up, and pivot identity embeddings. A2F-Bench contains 18.53 hours, 934 speakers, and 8,654 sequences. The paper reports 9.2% and 13.7% lip-vertex-error reductions on BIWI and VOCASET, and a 10-second inference time of 0.024-0.054 seconds on a V100. The code is Apache-2.0, but its reference environment is Linux, Python 3.10, CUDA 12.1, and PyTorch 2.2; checkpoints and dependent datasets/assets have separate terms. It is a useful future GNM-head fine-tuning base, not the least-risk production integration here.

[FaceFormer (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_FaceFormer_Speech-Driven_3D_Facial_Animation_With_Transformers_CVPR_2022_paper.pdf) demonstrates why contextual speech encoders matter: Wav2Vec2 features and autoregressive temporal attention improve closed-lip events such as `/b/`, `/m/`, and `/p/`. It is research evidence for context, not the selected runtime.

[CodeTalker (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Xing_CodeTalker_Speech-Driven_3D_Facial_Animation_With_Discrete_Motion_Prior_CVPR_2023_paper.html) addresses regression-to-mean with a learned discrete motion prior. That is valuable when training a future GNM-native model, but its autoregressive path is slower and tied to its training topologies.

### Emotion and nonverbal motion

[EmoTalk (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Peng_EmoTalk_Speech-Driven_Emotional_Disentanglement_for_3D_Face_Animation_ICCV_2023_paper.html) separates content, emotion, identity, and intensity, then predicts 52 blendshape coefficients. Its velocity loss explicitly matches predicted and ground-truth first differences to suppress jitter. This supports separate content and emotion channels rather than adding one full-face expression to every viseme.

[DiffPoseTalk (SIGGRAPH 2024)](https://diffposetalk.github.io/) models style and head pose as a stochastic, reference-conditioned process. The practical conclusion is not that this application needs diffusion immediately; it is that head pose and audio-uncorrelated motion are separate many-to-many signals and should not be derived by scaling mouth openness.

An LLM may plan phrase-level acting—emotion label, intensity, emphasis, and intended beats—but must never produce phoneme or frame timing. Lip microtiming must come from the acoustic model (or, for the fallback, a forced aligner).

### Alignment

When a transcript is available, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) remains the preferred diagnostic phone tier. [WhisperX](https://arxiv.org/abs/2303.00747) provides VAD, transcription, and forced word alignment, but word timestamps are not a replacement for phones. Alignment is useful for evaluation and fallback cues; the selected Audio2Face inference path does not require a transcript.

## Selected runtime architecture

```text
input audio
  -> ffmpeg: mono 16 kHz PCM
  -> learned motion backend
       -> Audio2Face v2.3.1 Claire, Swift/MLX, 30 fps
       -> [140 skin | 10 tongue | 15 jaw | 4 eye]
  -> NVIDIA-space post-process
       -> reconstruct Claire skin/tongue deltas
       -> bounded, temporally regularized ARKit solve
       -> 52 face + 16 tongue weights
  -> GNM retarget
       -> semantic ARKit-to-GNM target matrix
       -> lower-face/tongue contact preservation
       -> region-aware emotion/intensity overlay
       -> blink/eye/head nonverbal tracks
       -> direction-preserving coefficient bounds
  -> [T,383] expression + [T,4,3] joints
  -> GNM mesh -> preview and exported controls
```

### NVIDIA-space blendshape solve

For every frame, Claire's raw skin target is:

```text
y_t = mean_skin + c_t @ PCA_skin
```

Let `A` contain NVIDIA's active ARKit delta poses over `frontalMask`, `n` be the released ARKit neutral, and `w_(t-1)` the preceding solution. Solve `0 <= w_t <= 1`:

```text
min ||A w_t - (y_t - n)||²
    + lambda_L2 ||w_t||²
    + lambda_temporal ||w_t - w_(t-1)||²
    + lambda_symmetry ||S w_t||²
    + NVIDIA's coupled L1 approximation
```

Use the exact active-pose and regularization settings from `bs_skin_config.json`. Precompute `A^T A`, `A^T(mean-n)`, and `A^T PCA^T`, so per-frame solving operates on roughly 43 variables instead of 30,000 masked coordinates. Apply the same process to the 16 tongue targets.

### ARKit-to-GNM retarget

The initial retarget matrix is deterministic and auditable:

- `jawOpen` -> calibrated GNM lower-face aperture proxy;
- `mouthFunnel`, `mouthPucker`, `mouthStretch*`, `mouthSmile*`, `mouthFrown*`, `mouthRoll*`, `mouthPress*`, `mouthLeft/Right` -> corresponding region-masked semantic decoder directions;
- blink, squint, wide, brows, cheeks, and sneer -> region-masked GNM semantic directions;
- tongue targets -> the available 32-dimensional GNM tongue subspace, initially dominated by `tongue_center`.

Speech and emotion cannot simply be added. For each region:

```text
eyes   = learned_eyes + emotion_eyes + blink
mouth  = learned_mouth + emotion_mouth * (1 - 0.75 * speech_activity)
tongue = learned_tongue
pupil  = restrained arousal
```

If a region exceeds the supported coefficient magnitude, scale the entire region proportionally. Elementwise clipping changes the motion direction and is only the final safety guard.

The first retarget is semantic, not a topology-corresponded production solve. The production calibration upgrade is to sculpt GNM ARKit targets on the exact GNM topology and project them with GNM's included regularized PCA fitting utility.

## Expression phrasing

Emotion is a slow, time-varying performance track:

- infer or select a broad category;
- compute robust RMS, voiced probability, pitch range, and onset/emphasis tracks;
- smooth intensity over roughly 250-500 ms;
- attack an emotional phrase over 200-350 ms and release over 350-700 ms;
- reduce emotional lower-face contribution during high speech activity;
- create restrained head pitch beats at emphasis peaks and deterministic 100-160 ms blinks every 3-6 seconds.

Automatic emotion classification remains confidence-gated. A broad acoustic heuristic is not production emotion recognition. NVIDIA Audio2Emotion is a better supported companion but has its own license and usage restrictions; it can be added only after a separate legal and quality review.

## Quality benchmark

### Deterministic and geometry gates

- output arrays are finite and shaped `[ceil(duration*fps),383]`, `[T,4,3]`, and `[T,3]`;
- all coefficients remain in `[-3,3]`, with no silent saturation;
- identical audio, configuration, and seed produce byte-identical control arrays;
- speech does not leak into eye/pupil regions; emotion does not create tongue motion;
- all sampled GNM meshes are finite;
- at least 99.9% of sampled triangle normals retain their neutral orientation;
- audiovisual duration differs by at most one frame.

### Temporal gates

- stationary transition fraction during active speech < 8%; target < 3% for learned mode;
- no single-frame coefficient jump > 1.25;
- robust acceleration and jerk remain below thresholds established from approved reference clips;
- smoothing must not reduce P/B/M closure depth by more than 10% versus the unsmoothed contact target;
- silence returns within 5% of neutral mouth aperture within 150 ms.

These gates intentionally distinguish **smooth** from **mushy**: low jerk alone cannot pass if closures disappear.

### Timing and speech corpus

Use at least ten real human utterances, including:

- “Buy Bobby a puppy” for P/B/M closure;
- “Five vivid violets” for F/V contact;
- “She sells shiny shells” for fricatives;
- “Lily likes blue balloons” for tongue and rounded vowels;
- “Father saw a tall dark dog” for open/rounded vowels;
- fast, slow, whispered, accented, and modest-noise variants.

Manually annotate at least 100 closure/phone boundaries in Praat and require:

- median absolute boundary error <= 45 ms;
- 90th percentile <= 100 ms;
- P/B/M closure recall >= 90%;
- three-person lip-sync mean-opinion score >= 4/5.

### Expression corpus

Use a balanced real RAVDESS subset and require:

- manual emotion selections are visibly distinct and never corrupt lip timing;
- automatic valence sign accuracy >= 85% before automatic emotion is described as validated;
- arousal Spearman correlation >= 0.6;
- peak rendered emotion is recognized by raters >= 70% for happy/surprise/disgust and >= 50% for anger/sad/fear;
- neutral speech produces no persistent emotional pose.

### Learned-backend gates

- Swift/MLX emits the documented coefficient count and monotonically increasing timestamps;
- output varies on real speech and returns toward rest in silence;
- the ARKit solve has bounded weights and normalized reconstruction residual below a calibration threshold;
- at least jaw-open, pucker/funnel, smile/stretch, and press/closure control families activate on the speech corpus;
- learned mode improves stationary fraction and blinded preference over fallback mode on the same files.

## Phased implementation and stop/go rules

## Executed results on this machine

Phases 1-4 were implemented and exercised on the retained LibriSpeech and
RAVDESS files. The learned runtime is the exact Claire identity used by the
released geometry assets. Its 8-second output is byte-deterministic across two
runs (SHA-256 `82b2c2f56f733c3cafd7317e6bb6bc1551cc38d206c22c03ba72ca18eab19ddd`).
The Apple-Silicon inference pass took 1.3-1.5 seconds for 8 seconds of audio;
the 4.1-second emotional clip took 0.62 seconds. A 1.5-second silence input
produced 46 identical neural frames with zero temporal motion.

The end-to-end comparison below uses GNM mouth landmarks normalized by
interocular distance and the same output clock. `Fallback v2` is the new
procedural compiler, not the original hold/jump implementation.

| Clip/backend | Frozen lower-face transitions | Mouth step p95 | Velocity p95 | Acceleration p95 | Jerk p95 | Emergency-limited frames |
|---|---:|---:|---:|---:|---:|---:|
| Libri, original | 30.5% | 0.085 | 1.793 | 2.195 | not recorded | none |
| Libri, fallback v2 | 3.8% | 0.040 | 0.986 | 0.786 | 1.329 | 85 |
| Libri, learned Claire | **0.0%** | **0.025** | **0.464** | **0.282** | **0.455** | **4** |
| RAVDESS, original | 49.6% | 0.057 | 1.286 | 1.419 | not recorded | none |
| RAVDESS anger, fallback v2 | 5.7% | 0.040 | 0.868 | 0.642 | 1.030 | 15 |
| RAVDESS anger, learned Claire | **0.0%** | **0.031** | **0.613** | **0.336** | **0.423** | **2** |

Both learned runs rendered complete finite GNM meshes, muxed audio, preserved
all control frames, returned the lower face to rest, and exported raw neural,
52-channel ARKit, 16-channel tongue, and 383-channel GNM controls. The learned
solver activated jaw-open, funnel/pucker, close/press/roll, stretch/smile, and
tongue families on real speech. Manual emotion is sent through Audio2Face's
native ten-channel explicit emotion input; unvalidated automatic heuristic
labels are not applied to learned motion.

The automated quality scorer was tested against deliberately shifted tracks
(plus/minus two and four frames), heavy smoothing, static neutral, constant
open, cue permutation, and emotion-only silence motion. All adversaries fail.
The scorer intentionally refuses production approval without independently
authored phonetic-event annotations and matching geometry prototypes, so the
current application still reports `production_validated: false`.

Remaining production blockers are substantive rather than software failures:

- the semantic ARKit-to-GNM map needs artist-authored same-topology targets;
- GNM still has no physical jaw joint or collision/contact rig;
- tongue direction is compressed into GNM's single semantic tongue sample;
- no rights-cleared, independently annotated phone/contact corpus or human MOS
  panel was supplied for this execution;
- NVIDIA model redistribution and notices require product legal review.

The result is therefore a working learned prototype and a materially improved
review tool, not an unqualified production release.

### Phase 1 — temporal fallback and measurement

Build:

- continuous dominance/coarticulation weights;
- contact-aware filtering;
- per-frame prosody, emotion intensity, blinks, and head beats;
- temporal/contact metrics in `result.json`.

Tests:

- unit tests for dominance, silence, closure preservation, deterministic blinks, and region isolation;
- real LibriSpeech and RAVDESS end-to-end runs;
- before/after metric report and visual inspection.

Stop/go: do not proceed until all existing tests and new temporal/contact tests pass. This phase remains labeled `procedural_fallback`.

### Phase 2 — learned Audio2Face inference on Apple Silicon

Build:

- a minimal Swift executable that depends only on the `Audio2Face3D` product;
- exact-version dependency pinning;
- typed Python subprocess adapter and health reporting;
- cached, checksummed Claire model/assets setup.

Tests:

- Swift unit/E2E test with real weights;
- real LibriSpeech and RAVDESS inference;
- coefficient count, timestamps, finiteness, determinism, silence, and motion tests.

Stop/go: learned mode cannot be selected unless real model inference passes. A missing model produces a typed fallback warning, never silent substitution.

### Phase 3 — NVIDIA solve and GNM retarget

Build:

- PCA-to-ARKit precomputation and bounded temporal solver;
- 52-channel ARKit and 16-channel tongue artifact export;
- semantic ARKit-to-GNM matrix with speech/emotion region composition;
- contact-aware limits and mesh validation.

Tests:

- synthetic recovery from known ARKit weights;
- NVIDIA solver configuration parity tests;
- real-audio activation-family checks;
- geometry, contact, temporal, and saturation tests;
- comparison against Phase 1 on identical inputs.

Stop/go: learned mode is not exposed in the UI until synthetic solve recovery and real-audio GNM rendering both pass.

### Phase 4 — application integration and review loop

Build:

- `auto`, `learned`, and `fallback` backend options;
- visible backend, model, confidence, quality metrics, and warnings;
- downloadable raw learned motion, ARKit weights, GNM controls, and preview.

Tests:

- API/CLI parity;
- browser upload, playback, artifact links, backend/error messaging;
- image-pipeline regression suite;
- full real-input test matrix.

Stop/go: review correctness, architecture, error paths, licenses, and claims. Fix every failure and rerun the full suite plus browser QA.

### Phase 5 — production calibration and model improvement

This phase requires new data/art authority and is not faked by code:

- sculpt 52 GNM ARKit targets plus 12-20 speech/contact targets on GNM topology;
- project targets into GNM and have a facial animator approve them;
- acquire rights-cleared paired audio/4D or HMC facial performance;
- fine-tune an Audio2Face or UniTalker-style GNM head with vertex, lip-contact, velocity, acceleration, coefficient-prior, and emotion-disentanglement losses;
- run the annotated corpus and human MOS study.

Stop/go: only after those gates pass should the system be called production-quality without qualification.

## Risks and limitations

1. The Swift/MLX runtime is third-party and new; local parity tests are mandatory.
2. NVIDIA weights use the NVIDIA Open Model License, not Apache/MIT; product counsel must review distribution and notices.
3. Audio2Emotion has separate restrictions and is not implicitly cleared by using Audio2Face.
4. Claire motion is actor-specific. Solving to ARKit removes topology dependence but not every aspect of Claire's performance style.
5. GNM has no jaw joint. `jawOpen` is a deformation proxy, so lower teeth and jaw mechanics cannot exactly match Claire.
6. The initial GNM ARKit map uses coarse semantic samples. Artist-authored same-topology targets are required for final calibration.
7. GNM has no collision/contact system; lips, teeth, and tongue need explicit QA and possibly corrective shapes.
8. Public academic datasets are generally too small, too neutral, or subject to separate/research-only terms. Do not assemble a commercial training set by assuming repository licenses cover the data.
9. Audio alone does not uniquely determine gaze, blinks, head motion, or acting intent. Those signals must remain controllable and, where stochastic, seedable.
10. Low-quality audio, overlap, singing, whispering, dialect mismatch, and extreme performance require separate test coverage.

## Definition of completion for this execution pass

This pass is complete only when:

- the research and phase plan are committed to the workspace;
- the fallback composer passes new temporal/contact tests and improves real metrics;
- the learned model runs locally on real audio or is documented with the exact reproduced blocker;
- learned outputs are retargeted through named ARKit controls into finite GNM animation;
- real LibriSpeech and emotional speech produce muxed previews and inspectable artifacts;
- all existing image and app regressions pass;
- browser QA succeeds;
- measured limitations are stated without describing unvalidated output as production quality.
