# AI-Driven GNM Pipeline Feasibility

## Decision summary

| Pipeline | MVP feasibility | Production feasibility | Honest output claim |
|---|---|---|---|
| Audio to lip sync | High for clean audio with 6-9 visemes | Medium; needs calibrated targets and timing benchmarks | Timed GNM mouth animation |
| Audio to emotion | Medium for broad arousal/limited labels | Medium-low without licensed/trained speech-emotion data | Confidence-gated expressive intent |
| Single photo to GNM | High for coarse visible landmark geometry | Medium-low for likeness from one view | Single-photo GNM shape estimate |
| Photorealistic clone | Not provided by GNM | Requires another appearance/reconstruction stack | Out of scope |

## Pipeline A: audio to lip sync and expressions

### Recommended MVP

```text
audio file
  -> ffmpeg 16 kHz mono normalization
  -> Rhubarb offline phonetic recognition (A-H/X timed cues)
  -> deterministic GNM semantic prototypes
  -> region masking + coarticulation + coefficient bounds
  -> slow emotion envelope from transcript/prosody/manual override
  -> [frames, 383] coefficients
  -> official GNM mesh generation
  -> JSON/NPZ/OBJ and muxed preview video
```

Rhubarb is the best MVP dependency because it accepts audio directly, runs offline, emits timed mouth cues, can use optional dialog text, and is MIT licensed. Version 1.14's x86-64 macOS binary was verified on this Apple Silicon host through Rosetta. It produced 50 timed cues from an eight-second real LibriSpeech recording without a transcript.

The application evaluates GNM's checked-in H5 decoder with NumPy to obtain deterministic label means. Each viseme is region-masked so speech cannot move the eye blocks or pupil. Suggested bootstrap targets:

| Cue | Initial target |
|---|---|
| X | neutral zero vector |
| A, P/B/M | restrained `compress_face`, calibrated toward closed aperture |
| B, most consonants/EE | small `stretch_face + smile_wide` |
| C, EH/AE | medium `stretch_face` |
| D, AA | strong `stretch_face` |
| E, AO/ER | `funneler` |
| F, UW/OW/W | `pucker + funneler` |
| G, F/V | `lips_roll_in` |
| H, L | open lower face plus `tongue_center` tongue block |

Emotion is composed separately:

- joy: `happy + smile_wide`;
- surprise: `surprise`;
- disgust: `disgust`;
- sadness proxy: `corners_down + compress_face`;
- anger proxy: `snarl + platysma + compress_face`;
- fear proxy: `surprise + compress_face`;
- neutral/low confidence: zero.

An LLM may emit a validated segment-level emotion plan, but must never invent phoneme timings. For predictable local operation, the MVP accepts a manual emotion and uses a transparent acoustic arousal/valence heuristic when no override is supplied. A later text model can consume an ASR transcript.

### Accuracy path

For known dialog or serious timing requirements:

1. transcribe locally with whisper.cpp or accept a supplied transcript;
2. force-align phones with Montreal Forced Aligner;
3. map ARPABET phones to the cue library;
4. preserve diphthong trajectories and bilabial closures;
5. calibrate 12-20 artist-authored GNM viseme/coarticulation meshes;
6. use GNM's regularized PCA projection to recover coefficients;
7. benchmark on manually annotated real speech.

This is the point where "accurate lip sync" becomes a measured claim rather than a demo impression.

### Audio acceptance tests

Mechanical gates:

- cue intervals are monotonic, non-overlapping, and cover the audio duration;
- output shape is `[ceil(duration * fps), runtime_expression_dim]`, sampled at
  `arange(frame_count) / fps` so the last frame lies inside the clip;
- every coefficient and generated vertex is finite;
- coefficients stay within the configured safety range;
- speech cues have exactly zero eye/pupil coefficients;
- identical input and seed produce byte-identical controls;
- preview audio/video offset is at most one frame.

Geometry gates:

- mouth aperture satisfies `D > C > B > X`;
- P/B/M closure remains near neutral aperture;
- rounded/puckered cues narrow the mouth relative to wide-open cues;
- L visibly moves tongue vertices;
- silence returns to near-neutral within 150 ms;
- mouth landmark motion remains below both the absolute 4% interocular safety
  bound and the 1.20-interocular-units/s cadence-independent gate.

Real-input gates for a production claim:

- at least ten human utterances covering P/B/M, F/V, L, open vowels, rounded vowels, fast/slow/accented/noisy speech;
- at least 100 manually annotated phone or closure boundaries;
- median boundary error <=45 ms and 90th percentile <=100 ms;
- P/B/M closure recall >=90%;
- three-person lip-sync mean opinion score >=4/5.

Emotion gates must use real emotional speech, such as a license-reviewed RAVDESS subset. If top-2 emotion or valence/arousal targets are missed, the feature must remain labeled experimental or manual-only.

### Audio limitations

- GNM has no jaw joint, speech controls, coarticulation model, or contact constraints.
- Rhubarb provides a coarse mouth-shape timeline, not phone-accurate 3D performance.
- Sparse landmarks cannot fully validate tooth/lip or tongue/palate contact.
- Semantic emotion labels are incomplete and decoder samples can leak between regions.
- Audio emotion is ambiguous, culturally/domain dependent, and often license-sensitive.
- ASR and alignment failures propagate into animation.

## Pipeline B: single image to 3D face matching

### Recommended MVP

The product claim is: **"single-photo GNM shape estimate matched to visible facial geometry, with confidence and a neutral canonical mesh."** It must not say "3D clone" or imply metric likeness.

```text
JPEG/PNG
  -> face/quality preflight
  -> 68-point detector or versioned MediaPipe-to-68 mapping
  -> explicit GNM landmark permutation
  -> camera fit on mean GNM
  -> strongly regularized first 10 identity modes
  -> refine first 20 identity modes
  -> optional tiny nuisance expression fit
  -> full 253-vector with unobservable blocks held at zero
  -> official GNM mesh + overlay + metrics + confidence
```

The fit uses a compact landmark model derived exactly from GNM's barycentric definition:

```text
L(beta, phi) = L_template + B_identity * beta + B_expression * phi
```

This avoids evaluating all 17,821 vertices during every optimizer step. The final mesh is still produced with the official GNM API.

Fit stages:

1. reject zero/multiple faces, too-small faces, severe occlusion, yaw above about 35 degrees, or pitch above about 20 degrees;
2. fit weak-perspective camera using inner eyes and nose;
3. fit the first 10 head identity coefficients with a Gaussian prior and `[-3, 3]` bounds;
4. refine the first 20 using all visible landmarks, a robust loss, and lower jaw weights;
5. use MediaPipe blendshapes to reject strong expressions or fit only a few strongly regularized nuisance expression modes;
6. set eye, teeth, tongue, and pupil coefficients to zero because 68 landmarks cannot observe them;
7. return normalized mean error, pose, bound fraction, stability, overlay, and explicit confidence reasons.

Direct synthetic calibration against the checked-in asset supports this scope. With 20 modes, moderate pose, and 0.5-pixel landmark noise, a research probe achieved mean normalized landmark error around 0.007, coefficient cosine around 0.85, and mean visible-mask vertex error around 0.80 mm. Expanding to 40 modes reduced coefficient stability despite similarly low reprojection error. Low reprojection error alone is therefore not proof of recovered identity.

### Image acceptance tests

Unit gates:

- barycentric landmark weights sum to one;
- the explicit GNM-to-standard permutation produces a spatially valid canonical jaw;
- compact landmarks match `vertices_and_landmarks` within `1e-6`;
- unobservable identity/expression blocks have zero landmark basis and remain zero;
- invalid, zero-face, and multi-face inputs produce typed errors;
- all outputs have runtime-derived shapes and finite values.

Synthetic gates:

- zero identity recovery;
- random first-10/20 identity coefficients across camera pose/scale and 0, 0.5, and 1.5-pixel noise;
- noise-free normalized error <0.001 and coefficient cosine >0.98;
- at 0.5-pixel noise, normalized error <0.012, coefficient cosine >0.80, and visible facial vertex mean error <1.5 mm;
- compact fitted coefficients reproduce the official full-GNM mesh.

Real-photo gates:

- at least three consented identities with frontal and approximately +/-20-degree images;
- actual smile, glasses, facial hair, low-light, and >40-degree negative cases;
- accepted neutral views: inner normalized error <=0.025, all-landmark error <=0.04, and <=10% of fitted coefficients at bounds;
- same-person cross-view neutral meshes: pairwise visible-face mean distance <2 mm and first-10 coefficient cosine >0.9;
- explicit low-confidence/rejection on hard cases;
- human review of landmark overlays and a neutral turntable.

The included astronaut photograph is a valid public real-photo smoke test and MediaPipe 0.10.35 successfully returned one face, 478 landmarks, 52 blendshapes, and a transformation matrix. It is not enough to validate identity consistency by itself.

### Image limitations

- A single view is ambiguous in focal length, scale, depth, ears, and rear skull.
- GNM has no albedo or person-specific texture model.
- Landmarks miss cheeks, forehead, cranium detail, wrinkles, skin, hair, facial hair, and dental layout.
- Expression and identity can explain the same 2D error.
- Hair, glasses, occlusion, and detector contour conventions bias the fit.
- Unusual identities are pulled toward the PCA mean.
- ArcFace similarity against an untextured render is not a valid geometry metric.

### Higher-quality alternatives

- Ask for three photos or a five-to-ten-second turntable and bundle-adjust one identity across views. This is the highest-value product upgrade.
- Use MICA/FLAME as a metric initializer and retarget into GNM only after resolving the model-license and topology-correspondence work.
- Use DECA/EMOCA as research benchmarks, not silently as commercial dependencies.
- Move photometric/silhouette inverse rendering to Linux/CUDA with a differentiable rasterizer and an explicit appearance/lighting model.
- Evaluate research-grade accuracy against the [NoW benchmark](https://now.is.tue.mpg.de/) rather than selfie reprojection.

## Additional use cases unlocked

1. **Synthetic perception data.** Generate exact identity, expression, pose, gaze, landmark, segmentation, depth, normal, and UV ground truth.
2. **Cross-rig retargeting.** Calibrate ARKit/MediaPipe/FACS-like controls to GNM regional coefficients.
3. **Multi-view avatar fitting.** Shared identity with per-frame pose/expression produces a stronger avatar from a short capture.
4. **Expression editing.** Neutralize, amplify, or transfer expression while retaining identity.
5. **Talking NPC authoring.** Batch dialog-to-animation with deterministic controls and exportable geometry.
6. **Gaze and eye research.** Parametric cornea, iris, pupil, and eye joints enable controlled gaze/glint datasets.
7. **Dental and tongue visualization.** Internal anatomy supports speech/dental teaching and stylized visualization, subject to validation limits.
8. **Avatar QA.** Landmark, region, symmetry, coefficient, and mesh-validity diagnostics for artist-authored targets.
9. **Accessibility avatars.** Local, privacy-preserving speech avatars for communication, with clear non-clinical positioning.
10. **Animation compression.** Store identity once and transmit compact expression/pose tracks instead of dense mesh sequences.
