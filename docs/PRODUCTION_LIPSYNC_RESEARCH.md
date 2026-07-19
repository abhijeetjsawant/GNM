# Production audio-driven GNM facial animation

Status: working learned prototype; production temporal-model upgrade specified, not yet validated
Date: 2026-07-19
GNM revision studied: `3de70dfca5f3244620f44103c24b7cedc0dcb8b6`

## Executive decision

The original Rhubarb-to-nine-pose driver could not become production quality by adding a stronger low-pass filter. The application now has a working learned Audio2Face v2.3.1 path, dense GNM retarget, and contact-preserving geometry correction, but that path still has a production ceiling: its source network estimates one pose at a time, its downstream temporal conditioning is a fixed per-channel filter rather than a phonetic sequence model, and its expression condition has limited motion dynamics. More smoothing would hide some variation while weakening the transitions and closures that make speech readable.

The recommended architecture has two tiers:

1. **Learned local path:** NVIDIA Audio2Face-3D v2.3.1 Claire, running through the Swift/MLX Apple Silicon port, produces continuous actor-specific face, tongue, jaw, and eye motion. Reconstruct the Claire face motion with NVIDIA's released 140-shape geometry basis, solve 52 bounded ARKit weights using the released Claire targets and solver configuration, then retarget those semantic weights into GNM.
2. **Deterministic fallback:** replace hold-and-jump cues with dominance-based coarticulation, contact-aware temporal filtering, phrase-level prosody, region-aware emotion composition, blinks, and small head beats. This is a robust offline fallback and a useful diagnostic baseline, but it must not be marketed as equivalent to a facial-performance model.

For a commercial-quality product, a learned path should remain the default, but the current v2.3 path is a verified prototype rather than the final source model. The highest-impact implementable next phase is a source-model upgrade to Audio2Face-3D v3.0 on a version-pinned NVIDIA worker, followed by the existing auditable ARKit-to-GNM retarget and the existing phone-gated contact safeguard. This replaces framewise regression with sequence generation while preserving the P/B/M closure work already proven in GNM geometry. The final GNM retarget remains an approximation until the project has artist-authored GNM ARKit/viseme targets or rights-cleared paired audio and GNM motion for fine-tuning.

## 2026-07-19 production audit: why the learned result still feels rigid

### Short answer

The result is continuous, but continuity is not coarticulation. The current learned path removes obvious holds and caps large mouth steps, yet it does not generate a temporally coherent facial performance in GNM space. It starts with Claire v2.3, which NVIDIA documents as taking a 0.52-second audio window and producing one pose whose output is independent of previous and future facial poses. It then solves each frame into ARKit controls with only a previous-solution regularizer, applies a fixed five-sample Savitzky-Golay filter, linearly resamples the controls, and finally imposes a geometry-space emergency speed limit. Those downstream operations can suppress spikes, but they cannot add sequence-level anticipatory/carry-over behavior or actor-like expression dynamics beyond whatever the source's per-frame acoustic window already encoded.

The preserved P/B/M closures are not the source of the global rigidity. They are sparse, phone-gated anchors, and the composer already redistributes the approach/release when its continuity limit would otherwise open a seal. Weakening or smoothing those anchors would make the animation less intelligible without solving the motion-model problem.

### Code-grounded signal path

| Layer | What the repository currently does | Production implication |
|---|---|---|
| Neural source | `audio_pipeline.py:606-633` runs neutral Claire v2.3 and solves its 140 skin and 10 tongue PCA coefficients into named controls. NVIDIA describes v2.3 as 0.52 seconds of audio to one independently generated frame. | Local acoustic context exists, but there is no generated pose history or sequence-level motion state. Regression also tends toward an average performance. |
| ARKit solve | `a2f.py:805-827` solves every frame by bounded least squares and biases the right-hand side toward the previous solution. | This is useful retarget regularization, not learned coarticulation; the solver itself cannot add or refine the upcoming-/m/ influence beyond what the source coefficient already contains. |
| Post-conditioner | `audio_pipeline.py:107-174` uses the same five-sample, order-2 zero-phase polynomial filter by control family, with fixed residual gains. Contact-critical controls are left raw. | At 30 fps this is a five-sample support (roughly 167 ms), independent of phone identity, speaking rate, stress, or direction of transition. It reduces jitter but can make attacks/releases uniformly damped. |
| Cue contribution | `animation.py:84-117` uses fixed 32-55 ms attacks, 45-75 ms releases, static cue dominance, and retains at most two cue influences. In the learned composer these weights gate contact; they do not replace the learned mouth. | The gate prevents false seals, but Rhubarb's coarse A-H labels are not a production phone-alignment or coarticulation model. |
| GNM clock | `animation.py:1113-1129` linearly interpolates every GNM and affect channel onto the export clock. | Linear interpolation preserves duration and removes sampling gaps; it does not add biomechanical in-betweens. |
| Emotion | `audio_pipeline.py:713-773` runs a second, constant-label emotional v2.3 pass and subtracts neutral motion. `rig.py:119-139` retains the full upper-face delta but limits emotional lower-face motion during speech. | The content/affect separation is architecturally sound, but the source condition and envelope remain broad. They do not model phrase-specific acting, spontaneous upper-face variation, or identity-specific speaking style. |
| Contact | `audio_pipeline.py:177-278` derives closure evidence, quarantines an inverted `mouthClose` retarget row, and `animation.py:1130-1145` requires agreement with the closed-mouth cue. `animation.py:1187-1262` applies contact before the step limit and restores a seal lost by that limit. | Keep this. It is a sparse intelligibility constraint, not the cause of full-track stiffness. It still needs independent phone/contact truth before it can be called production-accurate. |
| Quality | `lipsync_quality.py:204-365` gates both the absolute `0.04`-IOD mouth step and `1.20` IOD/s mouth speed, stationary motion, return to neutral, false silence motion, target contrast, and apex timing, and correctly refuses self-authored annotations. | It detects several bad tracks, but does not yet score phone-context transition shape, approach/release timing, closure duration, speech-mesh perceptual alignment, or human preference. |
| Oral geometry | `oral_validation.py:1-11` explicitly limits its claims to lip/tongue/teeth geometry and structural proxies. | Passing oral validation proves finite, reconstructable geometry; it does not prove phoneme correctness, natural coarticulation, or believable acting. |

### What the present tests establish—and what they do not

The real LibriSpeech learned test requires less than 1% stationary lower-face motion, a quality-space mouth step no greater than 0.04 interocular units and speed no greater than 1.20 IOD/s, at least 99% source contact-peak retention, and articulation-range/rank retention (`tests/test_phase2_audio.py:726-800`). Separate contact tests prove that correction happens before continuity limiting, that non-P/B/M cues do not create seals, and that a lost reachable anchor is restored by moving its approach rather than opening the contact (`tests/test_phase2_audio.py:455-641`). The quality scorer also rejects synthetic time shifts, heavy smoothing, static/open mouths, cue permutation, and emotion-only silence motion (`tests/test_lipsync_quality.py`).

Those are meaningful correctness tests, but they cannot support a production perceptual claim. The retained real clips have no independently authored phone/contact tier, no reference 3D facial performance, no identity-matched speaking-style target, and no blinded human ratings. Consequently `production_validated` is correctly false. The present noncontact jerk ratio also deliberately excludes contact-critical controls, and no formal gate covers event-local transition jerk or closure hold duration. A smoother-looking test number can therefore coexist with a rigid-looking performance.

Focused re-verification on 2026-07-19 ran the quality adversaries plus the three contact/continuity tests from the live tree: `13 passed, 21 deselected in 7.36s`. No audio motion code was changed in this audit.

A later production-correction pass adds an explicitly authored neutral-relative
mouth-aperture solve rather than increasing jaw/mouth coefficients globally.
It hard-vetoes contact evidence, changes only GNM lower-face modes `200:350`,
measures PCA leakage onto the tongue mesh, prevents new lip-order inversion and
locally attenuates only edit deltas that would cross either the exact face-local
`0.03995`-IOD absolute step ceiling or the `1.1985` IOD/s per-edge limit computed
from exact timestamp deltas. On the retained eight-second learned clip, gain `1.08` changes
146/240 frames; two are locally continuity-limited, 98.63% reach the full
target, and the maximum final step is `0.03995`. Tongue controls and isolated
tongue geometry remain active on 238 frames, the maximum aperture-edit PCA
tail is `7.54e-7` interocular, and the structural oral audit finds zero final
lip-order or tongue/teeth proximity risk. These results fix a concrete
under-opening/geometry defect. They do not change the conclusion above: the
source is still a framewise Claire v2.3 model without independent phone timing
or perceptual production approval.

The animated GLB factorization now treats oral meaning as a hard export
invariant instead of relying only on aggregate vertex error. Inner-lip/contact
landmark supports are weighted during factorization; an accepted morph rank
must preserve every source contact classification and introduce zero signed
lip-order risks. The retained 67-frame video exposed why this is necessary:
the former rank-13 export passed its 0.410 mm global maximum-error gate but
created five viewer-only lip-order risks. The oral-semantic export selected
rank 29, preserved all twelve contact frames, introduced zero risks, and reduced
the measured oral maximum reconstruction error to 0.0933 mm. Result summaries
now aggregate control-track and reconstructed-viewer risk instead of hiding a
viewer-only defect.

The first stop/go evidence foundation is now implemented separately from
animation generation. `autoanim.lipsync-qualification/1.1` binds one existing
controls track to the exact source audio, character manifest, identity
artifact and evaluated identity array, runtime GNM/decoder rig, rational
timebase, manually independent annotation artifact, and artist-approved
68-point GNM target artifacts. Strict parsing rejects duplicate/unknown keys,
NaN, substituted hashes, self-scoring declarations, incomplete prototypes,
weakened thresholds, unsafe control archives, and clock mismatch. It then runs
the existing apex/prototype and motion-hygiene evaluator deterministically.

That version deliberately reports only
`independent_apex_pose_and_motion_hygiene_only`. Even when its core quality gate
passes, `production_validated` remains false because event start/release,
closure duration, context-dependent transition shape, and blinded perceptual
approval are not yet scored. Hashes establish exact artifacts and declarations;
they do not prove the human annotator's identity or process. Real qualification
still requires the independent corpus in Milestone A.

The terminal entry point is `autoanim-gnm qualify-lipsync`. It takes the sealed
profile, retained controls, exact source/character/identity artifacts, and one
`--evidence ARTIFACT_ID=PATH` argument for every declared annotation/prototype
artifact. The resulting scoped report keeps its canonical report digest and is
also HMAC-sealed by the selected local job-store trust root.

## Research convergence after the implementation audit

The primary literature points to changing the temporal generator and its training/evaluation objectives, not repeatedly tuning a global smoother:

- [NVIDIA Audio2Face-3D](https://arxiv.org/html/2508.16401) states that v2.3 emits one frame independently from a 0.52-second window, while v3 uses HuBERT, diffusion, GRU sequence state, a one-second audio window, and the central 0.5-second animation segment. NVIDIA says v3 generally produces higher-quality, more expressive animation. Both versions include explicit lip-distance objectives; v2.3 additionally uses phoneme prediction, phoneme-transition motion, velocity, quiet-audio, and lip-thickness losses. The official [v3 model card](https://huggingface.co/nvidia/Audio2Face-3D-v3.0) confirms skin, tongue, jaw, and eye output, HuBERT/transformer/diffusion architecture, and the need for use-case-specific evaluation. This is the closest drop-in production source upgrade for the current stack.
- [Learning Phonetic Context-Dependent Viseme](https://arxiv.org/abs/2507.20568) directly studies the failure mode here: framewise reconstruction does not explicitly model anticipatory and carry-over coarticulation. A five-frame phonetic-context weighting improved FVE, LVE, lip dynamic time warping, and maximum lip error across FaceFormer, CodeTalker, SelfTalk, and ScanTalk. Its lesson is to put the context into the learned objective, not to assume a five-frame output filter is equivalent.
- [Imitator](https://arxiv.org/abs/2301.00023) combines temporal prediction with a loss specifically weighted at detected /m/, /b/, and /p/ events. It is direct evidence that temporal smoothness and bilabial contact are complementary objectives rather than a tradeoff that should be resolved by erasing closures.
- [FaceFormer](https://openaccess.thecvf.com/content/CVPR2022/html/Fan_FaceFormer_Speech-Driven_3D_Facial_Animation_With_Transformers_CVPR_2022_paper.html) reports that short phoneme windows with limited context can yield inaccurate lips and uses long-term audio context plus autoregressive mesh prediction. [CodeTalker](https://openaccess.thecvf.com/content/CVPR2023/html/Xing_CodeTalker_Speech-Driven_3D_Facial_Animation_With_Discrete_Motion_Prior_CVPR_2023_paper.html) identifies regression-to-the-mean as a cause of over-smoothed motion and adds a learned motion prior plus temporal autoregression.
- [EmoTalk](https://openaccess.thecvf.com/content/ICCV2023/html/Peng_EmoTalk_Speech-Driven_Emotional_Disentanglement_for_3D_Face_Animation_ICCV_2023_paper.html) separates content, emotion, identity, and intensity. [Probabilistic Speech-Driven 3D Facial Motion Synthesis](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Probabilistic_Speech-Driven_3D_Facial_Motion_Synthesis_New_Benchmarks_Methods_and_CVPR_2024_paper.html) explains why the full performance is one-to-many: speech constrains articulation strongly, but not a single correct blink, brow, gaze, or acting trajectory. A deterministic mouth track plus generic head beats cannot represent that distribution.
- [Perceptually Accurate 3D Talking Head Generation](https://openaccess.thecvf.com/content/CVPR2025/papers/Chae-Yeon_Perceptually_Accurate_3D_Talking_Head_Generation_New_Definitions_Speech-Mesh_Representation_CVPR_2025_paper.pdf) shows why lip vertex error alone is insufficient and proposes mean temporal misalignment via derivative DTW, perceptual lip readability, and speech/lip intensity correlation. These complement this repository's geometry gates.
- The May 2026 [AudioFace preprint](https://arxiv.org/abs/2605.07478) is relevant to the optional LLM question: transcript and phoneme structure may improve interpretable mouth control. It is emerging evidence, not a reason to let an LLM invent frame timing. The safe product boundary remains: acoustic/phone model for articulation and timing; an LLM may provide a low-rate, editable phrase plan for intent, emphasis, and emotion.

## Selected next phase: sequence source with closure-constrained GNM retarget

### Decision

Integrate Audio2Face-3D v3.0 behind a source-agnostic motion interface and run the official, version-pinned model on a Linux/Windows NVIDIA worker. Use the official v3 identity geometry and blendshape solve to produce named semantic controls, then feed those controls through the existing calibrated GNM retarget. Keep the current GNM-space P/B/M contact calibration and post-limit anchor restoration until a better phone-conditioned contact head proves it can replace them.

This is more implementable and higher leverage than training a new GNM model immediately: released v3 weights and runtime already exist, while GNM-native training first requires rights-cleared paired facial performance and artist targets that the project does not have. It is also a cleaner experiment than adding another filter because the current v2.3 path remains available as an exact A/B baseline.

The NVIDIA worker is a real dependency, not an implementation detail to hide. The official v3 model card lists NVIDIA-accelerated Windows/Linux deployment; the current Apple-Silicon MLX runner only implements Claire v2.3. A local-only product can keep v2.3 as `preview` quality, but it cannot honestly claim that post-filter tuning reproduces v3 sequence generation.

The source boundary for that worker is now implemented, without pretending the
worker exists locally. The control vocabulary remains
`autoanim.sequence-control-schema/1.0`; the wire envelopes are explicitly
`autoanim.a2f-v3-worker-request/1.1` and
`autoanim.a2f-v3-worker-response/1.1`. Those sealed validators bind the exact
mono-16-kHz PCM sample clock, model/runtime/identity/control-schema hashes, the
model-native 60 Hz output clock, inference plan, named finite
skin/tongue/jaw/eye arrays, and a zero-rooted cascading execution-order
provenance chain. Version 1.1 corrects an earlier contract bug
that mislabeled the 30 retained frames in each half-second diffusion step as a
30 fps stream. The official SDK computes 60 frames from a one-second window,
discards 15 on each side, retains 30, and advances by half a second. Legacy
30 Hz v3 envelopes now fail closed; a 60 Hz source may still be resampled onto
an explicitly chosen 30 or 60 fps delivery clock. The generic immutable track
preserves the rational timebase and uses the explicit quality labels
`a2f_v3_sequence_candidate_unqualified` and
`a2f_v2_3_framewise_preview`, so a framewise v2.3 result cannot be relabeled as
v3. Content hashes are not authentication signatures; deployment still needs
an authenticated transport and worker identity. Local preflight on this
Darwin/arm64 machine fails closed with `NVIDIA_V3_EXTERNAL_WORKER_REQUIRED`.

The 1.1 chunk records are SDK inference records rather than arbitrary transport
partitions. Validation reconstructs the exact signed one-second windows from a
`-16000`-sample start, 8000-sample stride, left/right zero padding, source
intersection, 60 generated frames, 15/15 discarded margins, and 0–30 emitted
center frames. The fields are named `execution_chain_in_sha256` and
`execution_chain_out_sha256`, not state hashes. The execution-order hash chain
includes the zero-output warm-up inference and the final padded partial
inference. It binds ordering and payload integrity; it does not hash or prove
the SDK's private GRU tensor state. Exact integer callback target samples are
retained as evidence; `timestamps_seconds`
is deliberately canonicalized to `output_frame_index / 60` for downstream
interpolation. This prevents the SDK's alternating 266/267-sample timestamp
quantization from being mistaken for clock drift while preserving the original
sample ticks for audit.

ABI validity is intentionally broader than application importability. A sealed
one-frame response is valid transport and archive evidence, including its
zero-output warm-up execution and execution-order chain. The animation
application nevertheless rejects a v3 source track with fewer than two emitted
frames using `DURATION_TOO_SHORT`, before writing partial animation artifacts.
Retarget interpolation and motion-quality checks require a trajectory, not one
pose. Preserve a valid one-frame envelope for audit or supply longer audio; do
not relabel the packet as corrupt merely because it cannot become an animation.

The 60 Hz envelope is the canonical source-motion clock. A 30 or 60 fps
application delivery compiles timestamp-aligned samples over the same audio
duration; it does not make the two frame arrays identical, prove that 30 fps
preserves every high-frequency articulation detail, or replace the exact SDK
target-sample ticks retained in the inference records.

### Build milestone A — freeze real evidence before changing motion

1. Assemble a rights-cleared evaluation set of at least 40 real utterances from at least 10 speakers, including at least 200 independently identified P/B/M events, 100 F/V contacts, 100 rounded vowels, fast/slow speech, two accents, modest noise, and held-out whisper/singing stress cases.
2. Create an independent Praat/TextGrid tier with phone start, articulatory apex/contact, and release times. The annotator must not see Rhubarb, A2F, or generated motion. Double-annotate at least 20% and report agreement.
3. Record or license synchronized reference video/landmarks for all clips and 3D/HMC/4D motion for a representative subset. Sculpt and approve same-topology GNM targets for closure, open vowel, funnel/pucker, stretch, F/V contact, tongue-up, and tongue-forward.
4. Freeze v2.3 outputs, model hashes, GNM identity, renderer, and all current measurements. No candidate may tune against the held-out test split.

Stop/go: do not alter the default motion path until the independent tiers, prototypes, and baseline report exist. Synthetic prototypes and Rhubarb's own cues are useful unit fixtures but cannot satisfy this milestone.

### Build milestone B — v3 source integration

1. **Implemented contract:** define monotonically timed named skin, tongue,
   jaw, and eye controls; exact model/runtime/identity/schema and audio bindings;
   rational inference clock; signed inference-window/padding and execution-order
   provenance. Keep v2.3 and
   v3 as separately labeled implementations. Per-frame emotion conditioning is
   still an upstream worker capability and is not fabricated by the contract.
2. Deploy the exact official v3 model/runtime in a pinned container on an NVIDIA worker. Test offline full-sequence inference first, then the documented streaming mode (one-second windows, central 0.5-second segments, preserved GRU state).
3. Use NVIDIA's identity-matched geometry and official blendshape solver before GNM retarget. Do not reuse Claire-specific PCA assets for a different v3 identity.
4. Disable the generic lower-face Savitzky-Golay pass for v3 by default. Any per-region filter must win the paired real-input gates below; upper-face smoothing and strength remain separately tunable.
5. Derive closure confidence from v3 lip geometry/named controls and an independent phone-alignment tier. Apply the existing character-calibrated seal only where both agree. Preserve exact reachable contact anchors through the GNM quality-space limiter and report unresolved/false-closed cases.
6. Drive emotion as a low-rate time-varying condition or artist/LLM phrase plan, but keep articulation sourced from audio. Compose full upper-face affect, restrained lower-face affect during active speech, and no emotion-driven tongue motion. Every generated plan remains editable and auditable.

Stop/go: the integration must reproduce all current finite-geometry, duration, deterministic-seed, artifact, and contact tests before perceptual comparison. A failed worker must produce an explicit `preview_v2.3` fallback status, not silently label v2.3 as production.

### Build milestone C — paired production qualification

Run v2.3 and v3 on the same frozen real inputs, identity, retarget, contact system, renderer, and audio mux. Randomize and blind the outputs. A candidate passes only if all hard gates pass and the paired perceptual result passes; average smoothness alone cannot override a closure or timing regression.

| Area | Required real-input gate |
|---|---|
| Clock and integrity | Output duration within one rendered frame of audio; finite controls/meshes; monotonic timestamps; no silent fallback; exact model/runtime hashes recorded. |
| Independent timing | At least 100 held-out scored events; median apex/contact error <= 1 frame and p95 <= 2 frames at 30 fps. Report signed onset and release error separately so early and late errors cannot cancel. |
| Bilabial contact | P/B/M contact recall >= 90%; non-bilabial false-contact rate <= 5%; median contact-duration error <= 1 frame and p95 <= 2 frames. At least 99% of reachable contact anchors attained after retarget/continuity limiting. |
| Closure preservation | Candidate minimum inner-lip gap at independently annotated P/B/M contacts must be no worse than v2.3 plus 0.001 interocular units; approach/release smoothing may move neighboring frames, never the accepted seal frame. |
| Phone identity | Existing target-contrast median >= 0.80 and p10 >= 0.60 on artist-approved GNM prototypes. Include F/V and rounded/open contrasts rather than scoring only closure. |
| Coarticulation | Lip DDTW/LDTW on phone start-to-release windows must be lower than v2.3 on the held-out set with a paired-bootstrap 95% confidence interval entirely below zero. Report anticipatory onset, apex, release, event-local velocity/acceleration/jerk, and closure duration. |
| Motion hygiene | Mouth step max <= 0.04 interocular units and mouth speed max <= 1.20 IOD/s; active-speech stationary fraction <= 0.12; neutral return <= 2 frames; false-silence motion p95 <= 0.10 of reference amplitude. Event-local jerk must not improve by flattening target contrast or contact recall. |
| Speech/performance coupling | Speech/lip intensity correlation must improve over v2.3 on the held-out set. If a validated speech-mesh embedding is adopted, perceptual lip readability must also improve under paired bootstrap. Do not train and evaluate that metric on the same identities. |
| Expression | On at least 20 clips containing neutral-to-emotion or emotion-to-emotion phrase changes, independent raters must identify intended broad valence/arousal above the frozen baseline without a statistically significant loss in phone timing or contrast. Lower-face emotional energy during high speech activity must remain bounded by the existing regional composition rule. |
| Human perception | At least 12 blinded raters, at least 20 paired clips, randomized side/order. Candidate naturalness preference >= 60%, and the 95% confidence interval must exclude 50%. Collect separate 1-5 ratings for lip-sync, transition naturalness, expression appropriateness, and oral artifacts; median lip-sync and transition-naturalness ratings must each be >= 4. |
| Stress reporting | Noise, overlap, whisper, singing, and out-of-domain language are reported by condition. A supported condition may not be averaged with easier clean speech to hide a failure. |

The fixed numeric gates above are application acceptance criteria, not universal perceptual constants. Freeze them before the held-out run, publish all per-condition results, and revise them only from an animator-approved pilot—not by tuning on the final test set.

### Rejected immediate changes

- **Widen the current smoother:** likely to reduce jerk while delaying/flattening articulatory extrema. It does not add phone context and risks the exact P/B/M regression already guarded against.
- **Smooth the contact controls:** not justified. Current tests show that sparse contact preservation works, while no real annotations show a contact-jitter problem.
- **Let an LLM output visemes or frames:** text models can plan intent and emphasis, but acoustic alignment must remain authoritative for frame timing and phone contact.
- **Declare success from the existing real clips:** they prove executable end-to-end behavior and structural geometry, not production coarticulation or perceived acting.

### Viable alternatives if a v3 worker is unavailable

1. Keep v2.3 as an explicitly labeled local preview and expose artist editing of emotion, contact, and curve tangents.
2. Train a causal temporal adapter over v2.3/phoneme features only after paired, rights-cleared motion exists. Its objectives should include vertex/control reconstruction, velocity and acceleration, phonetic-context weighting, bilabial contact, lip distance/thickness, silence stability, expression disentanglement, and a coefficient prior. Without that data, an adapter is another unvalidated filter.
3. Train a GNM-native UniTalker/FaceFormer/CodeTalker-style head after artist-authored GNM targets and a licensed corpus exist. This may ultimately remove the intermediate ARKit solve, but it is a larger data and model-validation phase than the v3 integration.

## Unresolved perceptual-data gaps

- No independent phone start/apex/release annotations are present in the repository.
- No rights-cleared paired GNM or reference 3D facial performance is present.
- The two retained real speech fixtures are too few to estimate accent, speaker, rate, noise, or emotional-performance generalization.
- No blinded lip-sync/coarticulation/expression study has been run.
- No identity-matched speaking-style target exists; audio-to-full-face performance is one-to-many.
- GNM still lacks a physical jaw joint and exact collision model, so a temporally excellent source can still expose retarget/oral-mechanics limits.
- Current oral geometry checks deliberately use conservative proximity/order proxies on open surfaces; they cannot establish exact tongue/teeth collision or speech intelligibility.
- v3 has not yet been run through this repository, and its official GPU/runtime requirement remains an external deployment dependency.

## Why the original procedural baseline looked rigid

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

[NVIDIA's Audio2Face-3D paper](https://arxiv.org/abs/2508.16401) describes a production-oriented system trained from multi-camera 4D capture, with separate skin, tongue, jaw, and eye outputs, real-time inference, blendshape solving, and optional Audio2Emotion. Its v2.3 regression model consumes approximately 0.52 seconds of audio for a frame; v3 uses a one-second context and diffusion to emit a 30-frame retained block every 0.5 seconds, yielding a 60 fps source stream. NVIDIA reports direct mesh, joint, and blendshape workflows and time-keyed emotion control.

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
