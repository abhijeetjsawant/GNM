# Production-grade audio-driven facial animation for GNM

Status: repository- and artifact-grounded technical audit
Date: 2026-07-18
Audited artifact: `artifacts/jobs/01kxtt8w2rnpk3n5575nxqhaj7`
Scope: audio to lip sync, expression, jaw, tongue, eyes, gaze, blink, and head performance

## Executive finding

The originally audited result was rigid because it preserved only a narrow,
heavily conditioned projection of the learned performance. Audio2Face emitted
meaningful jaw and eye tracks, but that revision discarded both and invented
head/gaze behavior procedurally. The current compiler now retains those streams,
uses a raw-jaw observation in the dense retarget, and maps measured eyes into GNM
eye joints. The deeper structural limit remains: GNM has no mandible joint or
contact/collision rig, so jaw and tissue motion are still approximated through
lower-face expression components.

The audited clip compounds that loss with a broad temporal conditioner and a whole-lower-face speed limiter. This makes the result numerically smooth but attenuates plosive closure, rounding, blink, and jaw attacks. Its emotional and secondary motion is also low-dimensional: the clip has a held anger performance, negligible learned blink, no learned gaze direction, sub-degree head motion, and no translation. The exported 383-dimensional GNM expression track uses only four principal motion directions for 95% of its variance; its 32-dimensional tongue region uses one.

This is fixable, but production quality requires an articulation system rather than more generic smoothing. The recommended order is:

1. Preserve and drive the released Audio2Face jaw and eye tracks. **Implemented
   in the current prototype; artist axis/gain calibration remains.**
2. Upgrade the default backend from Audio2Face v2.3.1 at 30 fps to the v3 diffusion model at 60 fps where supported, retaining v2.3 only as a tested fallback.
3. Replace fixed filtering and the scalar lower-face limiter with a phoneme-posterior-informed, contact-aware constrained solve over lips, jaw, lower teeth, and tongue.
4. Add an explicit GNM-compatible mandible layer or, as an interim measure, a
   jaw-observation constraint in the blendshape solve. **The interim constraint
   is implemented; the physical mandible layer is not.**
5. Treat affect, gaze, blinks, head gestures, and idle motion as editable layers with explicit provenance. Audio can suggest them; it cannot recover a unique source performance.
6. Calibrate the mapping for each character/speaker and validate on independently annotated, rights-cleared recordings plus blinded animator review.

The audited artifact is correctly marked `production_validated: false`. Its score of `34.737` is not a production score: no independently annotated events were provided, so all content and timing checks failed closed.

## What the current system actually does

### GNM is the endpoint rig, not the speech model

Google's GNM v3 is a dense parametric head model with 17,821 vertices, 35,324 triangles, 253 identity coefficients, 383 expression coefficients, and four joints. The four joints in the loaded asset are `neck`, `head`, `left_eye`, and `right_eye`; there is no jaw joint. The expression vector is partitioned into left eye, right eye, lower face, tongue, and pupil regions. GNM supplies skin, eyes, teeth, and tongue geometry, semantic sampling, pose correctives, and skinning, but it does not supply phoneme timing, coarticulation, contacts, collision, or a speech-motion model.

Local evidence:

- `gnm/shape/README.md:8-28` describes identity, expression, neck/eyeball pose, translation, and the Apache-2.0 license.
- `src/autoanim_gnm/gnm_adapter.py:13-40` asserts the exact topology and parameter counts.
- The audited model asset reports the four joint names above.

### Learned audio path

The learned path is Audio2Face-3D v2.3.1 Claire through the Swift/MLX runner. Every source frame contains exactly 169 values:

| Partition | Dimensions | Intended meaning |
|---|---:|---|
| Skin | 140 | Claire facial PCA coefficients |
| Tongue | 10 | Claire tongue PCA coefficients |
| Jaw | 15 | Five tracked jaw points, each with XYZ displacement |
| Eyes | 4 | Two Euler controls per eye |

At the audited revision, the code parsed all four partitions but the main
pipeline solved only skin and tongue; jaw/eye arrays were neither persisted nor
composed. That architecture-boundary data-loss bug has since been fixed. Current
`arkit_controls.npz` files retain jaw rotation vectors and eye rotations, the
jaw observation softly restores opening in the dense solve, and measured eyes
are composed into GNM's left/right eye joints.

The skin and tongue solvers reconstruct Claire geometry and solve bounded named blendshape weights. The skin solve already contains L1/L2, symmetry, cancellation, and previous-frame temporal regularization (`src/autoanim_gnm/a2f.py:595-750`). A second Savitzky-Golay pass is then applied to the named controls (`src/autoanim_gnm/audio_pipeline.py:86-153`). The current working tree has shortened this to a five-frame, region-dependent detail-preserving conditioner, which is an improvement over the audited artifact's `contact-aware-savgol-v1`; it still does not enforce physical contacts or preserve articulator phase.

The learned expressions are linearly interpolated to the export clock, centered
by acoustically quiet frames, composed with an emotion delta, contact-corrected,
and passed through a mouth-step guard. Compiler v9 measures the guard in the
production evaluator's exact face-local geometry, then repairs any continuity-
lost contact by holding the proven seal frame and redistributing only a bounded
four-frame approach/release neighborhood. It is still an emergency local
projection rather than the windowed jaw/lip/tongue trajectory solve proposed
below, but the retained real job now keeps all three inferred contact targets.

Rhubarb's nine coarse mouth cues are retained as diagnostic timeline metadata and do not drive the learned mouth. This is the correct separation, but the cues are not accurate enough to validate phone contacts. The pipeline's A/V check compares stream duration and frame count, not phonetic synchronization (`src/autoanim_gnm/audio_pipeline.py:489-502`).

### Current artifact audit

The audited input is `03-01-05-02-01-01-01.wav`, a 4.104125-second, strong-intensity anger utterance from RAVDESS. The filename encodes the first statement, “Kids are talking by the door,” spoken by actor 01. The job used the learned Claire v2.3.1 path, manual anger at 0.65 strength, 30 fps, and 124 output frames.

| Observation | Measured value | Production implication |
|---|---:|---|
| Backend | Audio2Face v2.3.1 Claire regression | 30 fps, older and more averaged than v3 diffusion |
| Raw A2F layout | 140 skin + 10 tongue + 15 jaw + 4 eye | Full jaw/eye signals exist at input |
| Raw jaw norm range | 2.6395 model units | Material jaw performance is discarded |
| Raw eye norm range | 0.2625 radians/control units | Material eye performance is discarded |
| Raw ARKit skin rank, 95% energy | 6 | Already compact, but has useful independent modes |
| Conditioned ARKit skin rank, 95% energy | 5 | Conditioning removes one dominant motion direction |
| Final GNM expression rank, 95% energy | 4 | Retarget/composition further collapses motion |
| Final lower-face rank, 95% energy | 3 | Limited articulation variety |
| Final tongue rank, 95% energy | 1 | Tongue is effectively a single-axis gesture |
| `jawOpen` peak retention | 89.46% | Jaw attacks materially softened |
| `mouthFunnel` / `mouthPucker` peak retention | 89.03% / 86.32% | Rounded vowels lose shape and contrast |
| `mouthClose` peak retention | 97.40% | Close to useful, still below the recommended contact gate |
| Blink peak retention | 79.43% left / 79.76% right | Already tiny blinks become weaker |
| Source named eye-look controls | all exactly zero | No gaze direction survives this path |
| Mouth speed limiter | 7 frames | Consecutive interventions at 1.667-1.867 s plus 2.233 s alter real articulation |
| Head rotation maximum | 0.9369 degrees | Visually close to a locked head |
| Root translation | exactly zero | No body/camera-relative performance layer |
| Output stream offset | 0.876 frame | Container durations agree; phonetic timing remains untested |
| Independent timing events | 0 | No claim about sync accuracy is supported |
| Quality production gate | failed | Correct result; seven essential checks lack evidence |

The artifact's non-contact jerk p95 is reduced to 20.52% of the pre-conditioned result. That number looks attractive but is not a quality proof. Human speech contains high-frequency, high-acceleration contacts. Minimizing jerk without a reference distribution rewards a smooth, mushy mouth. The simultaneous loss in jaw, rounding, and blink peaks demonstrates that failure mode.

The older artifact conditioner used a nine-frame default support. At 30 fps, nine samples span 266.7 ms, with roughly 133 ms on either side of the target frame. That is long relative to labial closures. The current five-frame implementation reduces this support, but does not resolve the underlying double-regularization or absence of contact constraints.

### Post-fix retained-job audit (2026-07-18)

Retained job `artifacts/jobs/01kxv8041maktkjfcd5z9ftjkg` recompiles the same input audio (SHA-256 `2990a29cc750ed8ba32498cbe89596a411d19c189dc7aeafa4f1dfec8b97b323`) with compiler v8. The v7 baseline was neutral while the v8 job applies anger at 0.65, so this is an exact artifact ledger, not a controlled affect A/B.

| Measure | Pre-fix v7 | Post-fix v8 |
|---|---:|---:|
| Quality mouth-step maximum, interocular | 0.060836509183490144 | 0.039000047687167985 |
| Quality mouth-step p95, interocular | 0.049467948826571406 | 0.03899984544461134 |
| False-silence motion ratio p95 | 0.23354096493052207 | 0.06559222112146124 |
| Speech-active stationary fraction | 0.0 | 0.0 |
| Neutral return | 1 frame | 1 frame |
| Independently annotated/scored events | 0 / 0 | 0 / 0 |
| Mouth-step / false-silence checks | fail / fail | pass / pass |
| Limiter interventions | 7 / 211 frames | 32 / 211 frames |
| Post-limiter contact loss | 0 / 3 candidates | 1 / 3 candidates |

Compiler v8 replaces the activity-dependent 0.047-0.060 raw-landmark limiter with a bidirectional 0.039 cap measured in the evaluator's exact per-frame face-local geometry. It also adds a two-frame symmetric VAD hangover only to the silence-quality mask; it does not rewrite the animation's activity track. The conditioner, mask policy, and compiler version are recorded in the result and timeline for auditability.

The retained animated GLB passes the Khronos glTF Validator with 0 errors and 0 warnings; the sole informational message is an unused UV accessor on an untextured asset. That proves structural portability, not facial-performance quality. The result is smoother because abrupt frame-to-frame mouth displacements now remain below the production continuity threshold. It is still not production quality: 32 of 211 frames are limiter interventions, one of three candidate contacts is lost after limiting, and zero independent annotations leave seven content/timing checks unscored and failed closed. GNM still has no physical jaw or lip/teeth/tongue collision layer, and neither a valid GLB nor a low velocity bound proves phonetic timing, contact accuracy, or animator usability [1, 9, 10].

Imitator supports this separation of concerns: it trains a vertex-velocity loss to match frame-to-frame motion and a distinct bilabial-contact loss for `/m b p/`, noting that reconstruction-only training averages expressions and produces improper closures. Its ablation reports better lip-sync and closure realism from the contact term. Continuity is therefore a safety constraint, while production articulation still needs explicit, independently evaluated contact objectives [27].

### Compiler-v9 contact-anchor audit (2026-07-18)

Retained job `artifacts/jobs/01kxvby11g6gg7qb978njn87t0` rebuilds the exact
same input, affect, transcript hint, clock, and learned controls as v8. This is
therefore a controlled compiler comparison rather than a cross-affect result.

| Measure | Compiler v8 | Compiler v9 |
|---|---:|---:|
| Quality mouth-step maximum, interocular | 0.0390000477 | 0.0390000477 |
| Quality mouth-step p95, interocular | 0.0389998454 | 0.0389998454 |
| Limiter-adjusted frames | 32/211 | 32/211 |
| Contact candidates attained | 2/3 | **3/3** |
| Post-limiter contact loss | 1 | **0** |
| Locally restored contacts | not recorded | **1** |

The v8 contact solve at frame 103 reached its target (`0.0382913` versus
`0.0382938` interocular), but the one-sided continuity projection reopened it
to `0.0500935`. Holding that contact as an anchor and moving the approach into
frame 102 preserves the exact velocity ceiling. Only frames 102 and 103 differ
between the retained v8/v9 tracks; the intervention mask swaps frame 103 for
102 and remains 32 frames total. An incompatible contact is still rejected:
the repair is bounded to four frames, exact-evaluated, and accepted only when
every affected transition remains within the geometry contract.

This removes the observed contact-loss defect, not the production evidence
gap. The three targets are inferred from Claire plus Rhubarb rather than an
independent phone corpus, and a local anchor repair does not model a jaw hinge,
lip tissue, teeth, tongue, or collision. The longer-term production path is a
windowed constrained trajectory solve that minimizes deviation from reference
positions, velocities, and accelerations while enforcing character contact and
collision constraints.

## Why it reads as rigid

The perceptual causal chain of the originally audited artifact was:

```text
v2.3 regression at 30 fps
  -> skin/tongue blendshape solve with temporal regularization
  -> explicit jaw and eye tracks discarded
  -> second fixed temporal conditioner
  -> dense ARKit-to-GNM approximation without a jaw joint
  -> quiet-rest subtraction and linear resampling
  -> whole lower-face/tongue step limiter
  -> mostly held affect plus procedural secondary motion
  -> low-rank, anatomically underconstrained result
```

The current compiler removes the explicit jaw/eye data-loss step, aligns the
continuity contract, and preserves locally feasible contacts. The 30 fps source
prior, low-rank retarget, missing physical mandible/collision model, generated
secondary motion, and hard continuity interventions remain. That is why v9 is
smoother and more articulate without yet being production-natural.

This creates several visible symptoms:

- **Rubbery rather than articulate mouth:** the jaw, lips, and tongue are not modeled as linked articulators with distinct contacts. The limiter scales them together.
- **Weak consonant punctuation:** `/p b m/` need a sharp, complete lip seal; `/f v/` need lower-lip/upper-incisor contact. Generic smoothing attenuates both.
- **Rounded vowels look similar:** funnel and pucker peaks lose 11-14% in the audited clip.
- **Mannequin upper face:** anger is held while blink and gaze have little independent variation.
- **Mechanical motion rhythm:** secondary motion is deterministic and coupled to a small set of prosody features, so it repeats rather than reflecting scene intent.
- **Character mismatch:** the Claire source anatomy and dynamics are projected into a GNM head without a character-specific jaw hinge, lip seal, teeth/tongue offset, or expression-gain calibration.

## What audio can and cannot determine

### Reasonably inferable from audio

- phoneme/posterior probabilities and approximate contact timing;
- syllable, word, phrase, speech-rate, emphasis, pitch, energy, pause, and voice-quality features;
- lower-face motion correlated with articulation, including plausible jaw opening, lip closure, rounding, and some tongue events;
- broad affective evidence such as arousal and valence, with uncertainty;
- a distribution of plausible head, blink, brow, and gaze behavior conditioned on prosody.

### Not recoverable from audio alone

- the speaker's actual gaze target, eye contact, or saccade endpoints;
- the exact blink times, head gestures, listening reactions, and idle motion performed in the recording;
- a unique upper-face performance or microexpression sequence;
- semantic intent, sarcasm, concealment, cultural acting choices, or who in the scene is being addressed;
- exact hidden tongue/teeth geometry and contact when multiple articulations produce similar acoustics;
- the character's jaw hinge, dental occlusion, lip-seal shape, tissue behavior, facial asymmetry, and control gains;
- “ground truth emotion.” Prosody is ambiguous and labels such as anger are coarse acting directions;
- non-speaking partner reactions and turn-taking behavior from a monologue track.

Speech-driven face animation is therefore one-to-many. MeshTalk separates audio-correlated and audio-uncorrelated motion, and DiffPoseTalk explicitly evaluates probabilistic output; both are better conceptual models than pretending every generated blink or nod was recovered. If scene metadata supplies a look-at target, dialogue addressee, shot framing, or director beats, those signals should drive the corresponding layers. If absent, generated secondary motion must be labeled `generated`, seedable, editable, and reproducible.

An LLM can help convert a transcript plus scene metadata into phrase-level acting instructions such as `{start, end, valence, arousal, intent, emphasis, gaze_target}`. It must not generate phoneme-frame timing or overwrite high-confidence acoustic contacts. Director keys always take precedence.

## Recommended production architecture

```text
audio + optional transcript + optional scene/director controls
  |
  +-> normalization, clipping/SNR/reverb QA, VAD
  +-> acoustic model: A2F v3 raw skin/tongue/jaw/eyes at 60 fps
  +-> independent phone posteriors/forced alignment with confidence
  +-> prosody and phrase features
  +-> optional LLM phrase/intent plan (never phoneme timing)
          |
          v
  layered performance graph
    articulation reference | affect | gaze | blink | head/idle
          |
          v
  per-character calibrated constrained solve
    jaw hinge + lip/teeth/tongue contacts + collision + temporal prior
          |
          v
  GNM expressions + neck/head/eye joints + provenance/edit tracks
          |
          v
  geometry QA + independent timing QA + human review + export
```

### 1. Acoustic inference and alignment

Normalize to the exact model format and report input quality before inference. Reject or flag clipping, severe SNR, incorrect channel layouts, long dropouts, and unsupported language/voice styles. Preserve original audio and transformation provenance.

Use Audio2Face v3 diffusion at 60 fps for the quality tier. NVIDIA's current documentation attributes higher quality, better emotion and nonverbal behavior, and less averaging to the diffusion variants; the v2.3 regressors remain 30 fps and better characterized. v3 is a new runtime integration, not a model swap inside the existing Swift/MLX runner. Keep v2.3 as a lower-memory fallback until parity, latency, and licensing tests pass.

When a transcript is available, prefer the supplied script over ASR. Produce word/phone intervals and posterior confidence at 10-20 ms resolution with a language-appropriate aligner. Montreal Forced Aligner supports context-dependent triphones and speaker adaptation. Alignment is a contact prior and diagnostic layer; it must not quantize the continuous learned motion into nine viseme cards. Evaluation annotations must be made independently of the production aligner to avoid circular scoring.

### 2. Preserve raw jaw and eyes

This is the first implementation fix because the data already exists.

For each A2F frame, reshape the 15 jaw values into five 3D displacements. The released asset contains `neutral_jaw` with shape `[5,3]`. Form observed points:

```text
J_t = neutral_jaw + reshape(raw_jaw_t, 5, 3)
```

Fit a weighted rigid transform from `neutral_jaw` to `J_t` with Procrustes/Kabsch: subtract weighted centroids, compute the 3x3 covariance SVD, correct a negative determinant, then recover rotation and translation. Reject frames with large residual or non-finite values. This produces a physically interpretable jaw observation instead of inferring jaw only from `jawOpen`.

GNM has no mandible joint, so use two stages:

- **Interim:** add jaw displacement and lip-distance residuals to the bounded skin-control solve, following NVIDIA's documented jaw-driven soft-constraint approach. The raw observation should constrain `jawOpen`, `jawForward`, `jawLeft`, and `jawRight`, while leaving lip controls free to form contacts.
- **Production:** add a mandible articulation layer around GNM: lower teeth and tongue root attach to a calibrated jaw hinge, lower-face skin uses jaw correctives, and the final expression solve compensates tissue deformation. Version this layer separately from Google's model asset.

Map the four raw eye controls directly to GNM's left/right eye joints after calibrating axis order, sign, neutral offset, and angular scale on a small look grid. Do not substitute sinusoidal gaze for measured eye rotation. Procedural or target-driven gaze can be composed as a separate delta when the measured track is absent or intentionally overridden.

### 3. Contact-aware coarticulation solve

Use the learned retarget as a reference, not an inviolable final curve. At each offline clip, solve a windowed trajectory over GNM expression `x_t`, jaw transform `q_t`, and eye joints `g_t`:

```text
min sum_t
    w_ref(t) ||W (x_t - x_hat_t)||^2
  + w_jaw(t) ||jaw_observation(q_t) - raw_jaw_t||^2
  + lambda_v ||D1 x_t||^2
  + lambda_a ||D2 x_t||^2
  + lambda_j ||D3 x_t||^2
  + lambda_sparse ||x_t - x_hat_t||_1
  + contact and collision penalties
```

Subject to calibrated control bounds, jaw range/hinge limits, non-penetration, and phoneme-posterior contact constraints:

- `/p b m/`: upper/lower lip signed distance reaches the calibrated seal band at the contact apex;
- `/f v/`: lower lip reaches the upper-incisor contact band while the lips do not fully seal;
- `/t d n l s z/`: use tongue-to-alveolar/palatal constraints only where the rig contains a calibrated hidden contact target;
- open vowels: enforce a minimum aperture range conditional on the character and speaking style, not a universal constant;
- all events: preserve left/right asymmetry where observed and prohibit lip, tooth, and tongue mesh penetration.

Use asymmetric dominance kernels or phone-posterior windows so anticipatory coarticulation can begin before the acoustic boundary and release can differ by context. Contact apexes should be soft constraints under low confidence and near-hard constraints under high confidence. Because GNM expression is linear before pose correctives but signed-distance collision is nonlinear, a sequential quadratic or sequential convex solve is appropriate.

Do not smooth all controls identically. Dynamic weights should preserve fast, high-confidence contacts and permit stronger smoothing during steady vowels, silence, and low-confidence/noisy regions. For live mode, use a region-specific adaptive filter such as One Euro plus a short lookahead and a contact state machine with hysteresis. One Euro raises cutoff during fast motion and lowers it during slow motion; it is useful for live jitter control, not a replacement for the offline contact solver.

### 4. Emotion and secondary performance

Separate five output layers:

1. acoustic articulation;
2. affect/upper face;
3. gaze and eye rotation;
4. blink/saccade;
5. head/neck/idle.

Affect should be continuous and phrase-keyed rather than a single label held across a clip. Store valence, arousal, category mixture, intensity, confidence, and provenance. Blend at semantic beats, suppress only the affect channels that conflict with a speech contact, and preserve upper-face independence.

Gaze should be solved from explicit look-at or dialogue targets where available. Without targets, sample a reproducible plausible trajectory from a learned prior and expose the seed. Blinks need event timing and asymmetric eyelid curves, not exact periodic spacing. Head gestures should depend on phrase structure, speaker style, turn-taking, and scene constraints rather than mouth openness. Silence needs a listening/idle state; Audio2Face's own paper identifies semantically meaningful upper-face behavior and idle/listening behavior as limitations of audio-only short-context inference.

### 5. Speaker and character calibration

The Claire model is one actor prior; GNM is a statistical character space. A production mapping needs a versioned profile for each target character:

- neutral/rest pose and lip-seal target;
- jaw hinge location, rotation axes, translation coupling, and safe angular range;
- upper/lower teeth positions and dental occlusion;
- tongue root/height/depth offsets and safe contact surfaces;
- lip aperture, width, funnel, pucker, press, roll, and asymmetric gains;
- eye neutral, angular axes/range, eyelid coupling, and blink shape;
- expression-control bounds and correctives;
- source model, asset hashes, character identity hash, language/accent coverage, and calibration version.

For a lightweight profile, capture neutral plus calibrated extreme controls and phonetically balanced phrases with bilabials, labiodentals, alveolars, rounded vowels, fast transitions, and silence. For model adaptation, follow the scale of the A2F capture recipe: many phonetically varied 3-15 second sentences across styles/emotions, synchronized high-frame-rate multi-view or 4D capture, and held-out speakers/phrases. Fit the mapping with geometry, contact, temporal, and sparsity losses; never tune and report on the same clips.

## Production evaluation protocol

The thresholds below are proposed release gates for this application, not values guaranteed by the cited papers or ITU. Establish final values from approved capture distributions and renderer scale. ITU-R BT.1359's perceptual A/V windows are delivery-level detectability/acceptability guidance; they are too loose to validate phoneme animation.

### Independent automatic gates

| Area | Initial production gate | Notes |
|---|---|---|
| Content timing at 60 fps | median absolute apex error <= 1 frame (16.7 ms), p95 <= 2 frames (33.3 ms) | Independent manual/contact annotations; report milliseconds too |
| Stream sync | absolute audio/video clock skew <= 1 rendered frame; no cumulative drift | Separate from content timing |
| `/p b m/` lip seal | precision >= 98%, recall >= 98% | At least 100 held-out events per contact family |
| `/f v/` lip/incisor contact | precision >= 95%, recall >= 95% | Character-calibrated signed-distance band |
| Contact peak retention | bilabial >= 98%; jaw/funnel/pucker/tongue >= 95% | Compare before/after postprocessing; apex shift <= 1 frame |
| Collision | no sustained penetration > 0.25 mm; any > 0.50 mm fails | Proposed initial geometry tolerance, validate against mesh scale |
| Jaw observation fit | median <= 1 mm and 1 degree; p95 <= 2 mm and 2 degrees | Against rights-cleared capture, not the model's own proxy |
| Dynamics | acceleration spectrum and Fréchet mouth-motion distance within approved capture distribution | Do not optimize raw jerk downward without a reference |
| Neutral/silence | no speech-correlated mouth motion in clean silence; calibrated return without terminal snap | Test breathing, room tone, music, and nonverbal audio separately |
| Robustness | no NaN, saturation, inverted/degenerate geometry, clock drift, or nondeterminism for a fixed seed | Report fallback and confidence |
| Affect classifier | macro UAR/F1, valence/arousal CCC, calibration error, and slice results | Does not prove “correct acting” |
| Secondary motion | seed diversity without mode collapse; look-at error when target supplied | Plausibility requires human review |

The existing evaluator has a good fail-closed principle: timing/content require independent annotations (`src/autoanim_gnm/lipsync_quality.py:1-6,204-365`). It is not yet a production evaluator because it uses only a 68-point mouth proxy, accepts as few as three events, and has no explicit lip seal, lower-lip/teeth contact, tongue contact, jaw fit, collision, spectral dynamics, confidence slice, or capture-reference metric.

### Required test slices

Cover at minimum:

- slow, conversational, fast, whispered, shouted, breathy, and sung/unsupported speech;
- bilabials, labiodentals, alveolars, sibilants, rounded vowels, diphthongs, repeated contacts, and rapid cross-family transitions;
- sentence-initial/final contacts and contacts adjacent to silence;
- clean studio, clipping, compression, music, stationary noise, transient noise, reverberation, and microphone distance;
- short clips, long-form clips, pauses, false starts, laughter, breaths, coughs, and non-speech vocalization;
- each supported language, accent, age/voice range, speaking rate, and character calibration;
- neutral and each supported affect at multiple intensities;
- scene target present/absent, direct-to-camera, and dialogue turn-taking.

Every critical contact family needs at least 100 independently annotated held-out events before its gate is meaningful. Product claims across languages require representative speakers and phone inventories per language, not an English-only aggregate.

### Human and production-workflow gates

Run randomized, double-blind A/B tests against the previous build and, where rights allow, capture reference. Include both experienced facial animators/TDs and naive viewers. Use audio-on and audio-off trials to separate synchronization from motion plausibility. Follow ITU-T P.910 principles for controlled subjective audiovisual testing.

Measure:

- perceived sync, articulation clarity, naturalness, affect appropriateness, and overall preference;
- animator time-to-approval, number/type of keys changed, contact corrections, and shot rejection rate;
- P0 failures: collision, missed critical closure, severe timing error, broken geometry, or wrong gaze target;
- inter-rater agreement and confidence intervals, not only mean scores.

Set the final release threshold after a pilot establishes variance. A practical production criterion is that a new build must have no statistically significant regression on any critical slice, no P0 geometry/contact defects, and a lower confidence bound on “usable without structural reanimation” agreed with the animation lead. Edit time is more meaningful than a single automated lip-sync score.

## Dataset and licensing reality

Public research corpora are useful for benchmarking, but most do not provide commercially clean paired 4D face/audio training data.

| Asset | Useful for | Terms / constraint | Recommendation |
|---|---|---|---|
| Google GNM | Target head model and tools | Apache-2.0 | Suitable, retain notices |
| NVIDIA A2F SDK / training framework | Runtime/API and reference training code | SDK MIT; framework Apache-2.0 | Suitable subject to component notices |
| NVIDIA A2F v2.3/v3 weights | Learned motion | NVIDIA Open Model License; model cards describe commercial and noncommercial use | Legal review and attribution/use-policy manifest required |
| NVIDIA Audio2Emotion | Affect conditioning | Custom terms restricting use to Audio2Face | Treat as a separate licensed component |
| NVIDIA Claire sample dataset | Demonstration/evaluation | Custom terms limit use to the A2F project; not a general production corpus | Do not use as unrestricted commercial training data |
| RAVDESS | Emotion/audio robustness research | CC BY-NC-SA 4.0; commercial license offered separately | Current audited clip is research-only unless separately licensed |
| VOCASET | 4D speech-face research benchmark | Research/noncommercial license; commercial use and commercial model training prohibited | Benchmark only under its agreement |
| BIWI 3D audiovisual corpus | Research benchmark | Corpus EULA limits use to research and prohibits commercial use/redistribution | Benchmark only |
| IEMOCAP | Emotion research | Signed institutional data-release agreement | Verify institution and intended-use rights; do not assume commercial rights |
| CREMA-D | Acted 2D audiovisual emotion | ODbL database and DbCL contents terms | Potential research/stress set; legal review for derived artifacts and distribution |

The production-clean route is commissioned, talent-released capture with explicit consent for model training, evaluation, derivative animation, commercial deployment, retention, and deletion. Maintain a rights manifest with source hashes, performer consent, permitted uses, territory/term, redistribution rules, biometric/privacy status, and any demographic metadata restrictions. Public audio-only corpora can expand noise/language stress testing but cannot validate 3D contact geometry.

## Priority gap table

| Priority | Gap | Evidence | Remediation | Exit evidence |
|---|---|---|---|---|
| P0, prototype fixed | Raw jaw and eye tracks were discarded | Current artifacts retain 15+4 source values, jaw observation, and measured eye joints | Artist-calibrate jaw/eye axes, offsets, gains, and confidence behavior | Held-out jaw/eye capture plus animator approval |
| P0 | No GNM mandible/contact rig | GNM has neck/head/eye joints only | Add jaw layer or jaw-constrained corrective solve; calibrate teeth/tongue | No contact/collision P0s on held-out capture |
| P0 | No independent content validation | Artifact has zero annotated events; gate fails | Build rights-cleared annotated corpus and contact prototypes | All contact/timing gates pass by slice |
| P0 | Generic conditioner attenuates articulation | Jaw 89.46%, funnel 89.03%, pucker 86.32% peaks | Contact-aware constrained trajectory; remove duplicate generic smoothing | Peak retention and timing gates pass |
| P0 | Whole-lower-face scalar limiter changes phase | Seven clustered limiter frames in audited clip | Per-articulator constraints and reference-distribution outlier handling | No limiter-induced apex shift/collision |
| P1 | 30 fps v2.3 regression averages performance | Audited backend and NVIDIA comparison | Integrate v3 diffusion 60 fps quality tier | Model parity, latency, quality, and license gates |
| P1 | Claire-to-GNM mapping lacks character anatomy | No hinge/seal/teeth/tongue profile | Versioned per-character calibration | Held-out jaw/contact/collision thresholds |
| P1 | Affect is coarse and weakly editable | One manual anger label/intensity in artifact | Phrase-level multi-label affect track plus director keys | Human affect ratings and no lip-sync regression |
| P1 | Secondary performance is not recovered | Tiny blink, zero eye-look, sub-degree head in artifact | Target-driven gaze; stochastic editable blink/head/idle layers | Look-at accuracy and blinded plausibility tests |
| P1 | Quality score overweights generic geometry | Three-event minimum; no contact/collision/jaw metrics | Expand evaluator and require adequate event counts/slices | Fail-closed release report with provenance |
| P2 | Live temporal behavior unspecified | Offline zero-phase filtering hides latency issue | Region-specific One Euro/lookahead/contact state machine | Measured p50/p95 latency and live QA |
| P2 | Unsupported audio behavior not surfaced | Nonverbals/noise are A2F limitations | Input QA, confidence, explicit fallback/unsupported states | Robustness matrix with honest status |

## Phased execution plan

### Phase A0 — Baseline and evidence lock

- Freeze the audited artifact, exact model/asset/code hashes, source audio rights, and numeric metrics.
- Add a reproducible analysis command for rank, control retention, jaw/eye magnitude, contacts, collisions, and clock sync.
- Create independent Praat/TextGrid annotations for at least 100 events in each critical contact family.
- Define character-relative contact surfaces and neutral geometry.

Exit: baseline reproduces bit-for-bit; evaluation fails closed when annotations or rights metadata are missing.

### Phase A1 — Restore released performance channels

- Persist raw jaw/eye arrays and their provenance.
- Implement and unit-test the five-point rigid jaw observation.
- Calibrate/map raw eye controls to GNM eye joints.
- Use jaw observations as soft constraints in the existing blendshape solve.
- Remove redundant filtering for channels already stable; keep explicit before/after curves.

Tests: synthetic rigid-transform recovery, determinant/sign cases, missing/corrupt frame handling, eye-axis grid, real audited clip, contact peak retention, GLB reconstruction, and visual A/B.
Exit: raw-to-output jaw/eye correlations are documented; jaw/funnel/pucker peak retention >=95%; no regression in independently annotated timing.

### Phase A2 — Jaw/contact/coarticulation solver

- Implement the calibrated mandible/teeth/tongue articulation layer.
- Add phone posterior confidence and asymmetric dominance windows.
- Add lip-seal, labiodental, tongue-target, jaw-range, and collision constraints.
- Replace the whole-vector limiter with constraint/outlier diagnostics.

Tests: isolated and contextual phone suites, rapid transitions, all contact surfaces, extreme jaw, corrupted alignment, noise, long clips, and no-transcript path.
Exit: automatic P0 contact, collision, jaw, and timing gates pass on held-out rights-cleared capture.

### Phase A3 — v3 60 fps quality backend

- Integrate the official supported v3 runtime behind a capability boundary.
- Retarget raw v3 outputs through the same calibrated solver.
- Keep v2.3 as an explicit fallback with a visible quality label.
- Record model version, license, device, latency, and random seed.

Tests: official fixture parity, long/short input, model/device failure, deterministic seeded output, v2.3 fallback, memory, p50/p95 latency, and blinded v2.3/v3 A/B.
Exit: v3 improves production metrics/human preference without new P0 regressions and satisfies deployment constraints.

### Phase A4 — Affect and secondary performance layers

- Add phrase-level affect controls and optional LLM scene-plan JSON schema.
- Add look-at/director targets, generated gaze/blink/head/idle priors, seeds, and per-layer overrides.
- Export each layer separately with `measured`, `inferred`, `generated`, or `authored` provenance.

Tests: conflicting director/LLM/acoustic inputs, sarcasm/ambiguous prosody, silence/listening, gaze target changes, deterministic seed, diversity across seeds, and animator edit round-trip.
Exit: target-driven gaze is accurate; generated performance is plausible in blinded review; articulation gates do not regress.

### Phase A5 — Character/speaker calibration

- Build calibration UI and versioned profile schema.
- Capture or author the neutral, extremes, contacts, and phonetically balanced set.
- Fit regularized mapping on train material and report only held-out performance.
- Add profile compatibility/hash checks and safe fallback.

Tests: multiple identities/anatomies, teeth/tongue offsets, asymmetry, invalid profile, asset update, and cross-speaker stress.
Exit: every shippable character passes held-out contact, collision, jaw, and human review gates.

### Phase A6 — Production validation and release

- Run all language/style/noise/nonverbal slices, expert and naive studies, and animator edit-time trials.
- Produce a signed model/data/license/provenance card and known-limitations matrix.
- Add canary metrics for failure rate, fallback rate, latency, saturation, collision, and user overrides without retaining biometric/audio data beyond policy.

Exit: all automatic gates pass; animation lead signs off; no unresolved P0/P1; rights and privacy review complete; rollback/fallback verified.

## Authoritative and primary sources

1. Google GNM repository and local model documentation: <https://github.com/google/GNM> and `gnm/shape/README.md`.
2. NVIDIA, *Audio2Face-3D: Audio-driven Realistic Facial Animation for Digital Avatars* (model architecture, output layout, capture, postprocessing, metrics, limitations): <https://arxiv.org/abs/2508.16401>.
3. NVIDIA Audio2Face-3D official repository (SDK, framework, models, component licenses): <https://github.com/NVIDIA/Audio2Face-3D>.
4. NVIDIA ACE Unreal Audio2Face documentation (v3 diffusion at 60 fps versus v2.3 regression at 30 fps): <https://docs.nvidia.com/ace/ace-unreal-plugin/latest/ace-unreal-plugin-audio2face.html>.
5. NVIDIA Audio2Face release notes: <https://docs.nvidia.com/ace/latest/modules/a2f-docs/text/changelog.html>.
6. NVIDIA Claire v2.3.1 model card: <https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire>.
7. NVIDIA Audio2Face v3 model card: <https://huggingface.co/nvidia/Audio2Face-3D-v3.0>.
8. NVIDIA Claire sample dataset card and use terms: <https://huggingface.co/datasets/nvidia/Audio2Face-3D-Dataset-v1.0.0-claire>.
9. Zhou et al., *VisemeNet* (animator-centric visemes, style controls, coarticulation, temporal discontinuity/contact issues): <https://people.umass.edu/~yangzhou/visemenet/visemenet.pdf>.
10. Edwards et al., *JALI* (jaw/lip integration from audio and transcript): <https://doi.org/10.1145/2897824.2925984>.
11. Casiez et al., *The 1€ Filter* official project (speed-adaptive real-time filtering): <https://gery.casiez.net/1euro/>.
12. Richard et al., *MeshTalk* (audio-correlated versus audio-uncorrelated facial motion): <https://arxiv.org/abs/2104.08223>.
13. Ng et al., *Learning to Listen* (non-deterministic dyadic facial motion): <https://openaccess.thecvf.com/content/CVPR2022/html/Ng_Learning_To_Listen_Modeling_Non-Deterministic_Dyadic_Facial_Motion_CVPR_2022_paper.html>.
14. Yang et al., *DiffPoseTalk* (probabilistic speech-driven facial motion and benchmark limitations): <https://arxiv.org/abs/2311.18168>.
15. Fan et al., *FaceFormer* (long-context speech features and speaker style): <https://openaccess.thecvf.com/content/CVPR2022/html/Fan_FaceFormer_Speech-Driven_3D_Facial_Animation_With_Transformers_CVPR_2022_paper.html>.
16. Peng et al., *EmoTalk* (disentangled speech content, emotion, identity, and intensity): <https://arxiv.org/abs/2303.11089>.
17. McAuliffe et al., *Montreal Forced Aligner* (triphone alignment and speaker adaptation): <https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html>.
18. Chung and Zisserman, *Out of Time: Automated Lip Sync in the Wild* / SyncNet (diagnostic A/V synchronization): <https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/>.
19. ITU-R BT.1359-1 (relative timing of sound and vision): <https://www.itu.int/rec/R-REC-BT.1359-1-199811-I/en>.
20. ITU-T P.910 (subjective video/audiovisual quality methods): <https://www.itu.int/rec/t-rec-p.910/en>.
21. VOCA project and license: <https://voca.is.tue.mpg.de/index.html> and <https://voca.is.tue.mpg.de/license.html>.
22. BIWI 3D audiovisual corpus EULA: <https://data.vision.ee.ethz.ch/cvl/datasets/B3DAC2/CorpusEULA.pdf>.
23. RAVDESS dataset and license: <https://zenodo.org/records/1188976>.
24. CREMA-D repository and license: <https://github.com/CheyneyComputerScience/CREMA-D> and <https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/LICENSE.txt>.
25. IEMOCAP access page and release agreement: <https://sail.usc.edu/iemocap/> and <https://sail.usc.edu/iemocap/Data_Release_Form_IEMOCAP.pdf>.
26. Karras et al., *The Wild West of Speech-Driven Facial Animation* (lack of evaluation consensus): <https://diglib.eg.org/items/98ecd1e1-ed35-48a2-9541-abd09ef733aa>.
27. Thambiraja et al., *Imitator: Personalized Speech-driven 3D Facial Animation* (ICCV 2023; vertex-velocity consistency and bilabial `/m b p/` lip-contact supervision): <https://openaccess.thecvf.com/content/ICCV2023/html/Thambiraja_Imitator_Personalized_Speech-driven_3D_Facial_Animation_ICCV_2023_paper.html>.

## Bottom line

GNM has enough expression capacity for a materially better result. The current
prototype now preserves A2F jaw and eye evidence, uses a dense calibrated map,
adds spatial contact correction, and enforces the evaluator's continuity
contract while preserving all three inferred contacts on the retained job. Its
remaining rigidity comes from the 30 fps low-rank source prior, hard interventions
on 32/211 frames, generated rather than recovered secondary acting, and the
absence of a physical mandible/contact system. A character-specific windowed
coarticulation/contact solve, independently
annotated evaluation, and blinded animator review are the shortest defensible
route from the improved prototype to production quality.
