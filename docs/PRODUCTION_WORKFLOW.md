# AutoAnim production character workflow

Status: working implementation plus production gap analysis, 2026-07-19. This
document is intentionally stricter than a product pitch. A feature is marked
implemented only when the repository contains the path and an executable test;
"production validated" means an independent acceptance gate has passed, not
merely that an output looks plausible.

## Executive decision

Build AutoAnim as a **local-first production service with a browser workspace,
a terminal interface, and DCC interchange**:

- The browser is the review/editor surface: character library, synchronized
  source media and 3D playback, quality overlays, version selection, and job
  provenance.
- The local Python service owns biometric assets, native Audio2Face inference,
  MediaPipe tracking, GNM evaluation, deterministic compilation, signing, and
  export. Raw performer media does not need to leave the workstation.
- The CLI is the automation and LLM boundary. Codex or Claude may propose an
  acting plan through a strict declarative schema, but cannot write rig values,
  invoke tools, or override lipsync.
- OpenUSD/UsdSkel is the proposed editable master for a future body-attached
  asset; glTF/VRM is the browser/runtime delivery form; Blender/Rigify is the
  first open artist round-trip.

This is a better fit than a cloud-only web app because the current learned
audio path is an Apple-Silicon native executable, facial inputs are biometric,
production files are large, and artists still need Blender/Maya/Unreal-style
correction. It keeps the convenient web UI without pretending a browser is a
complete character DCC.

## What GNM actually provides

The pinned upstream checkout is Google GNM Head 3.0 at commit
`3de70dfca5f3244620f44103c24b7cedc0dcb8b6`. The released model is a linear,
parametric head model with:

- 253 identity coefficients;
- 383 expression coefficients divided into semantic regions;
- 17,821 native vertices and 35,324 triangles;
- 4 joints: neck, head, left eye, and right eye;
- skin, eyes, teeth/gums, tongue, and mouth-interior geometry;
- 933 tongue vertices and 1,824 tongue triangles;
- triangle-corner UVs and semantic vertex groups;
- NumPy/JAX/PyTorch/TensorFlow evaluation and a learned semantic expression
  decoder.

GNM does **not** ship a released full body, physical jaw joint, muscle or
collision solver, a photogrammetry pipeline, an audio model, or a production
skin shader. Its identity/expression bases are excellent editable geometry
coordinates; they are not a one-click digital-human system.

AutoAnim holds one identity vector fixed and evaluates expression and joint
tracks over time. Audio and video therefore share the same final GNM evaluator
and exporter even though their evidence is different.

## One production graph, not separate demos

```text
rights-cleared capture
        |
        +--> identity fit ----+
        |                     |
        +--> appearance bake -+--> immutable character revision
        |                     |     identity + materials + oral/body profile
        +--> body attach -----+     consent + hashes + provenance
                              |
source audio/video -----------+--> measured performance evidence
instructions/transcript -----+--> optional LLM intent proposal
                                    |
                                    v
                         deterministic ownership compiler
                    lipsync | face affect | gaze/head | body | overrides
                                    |
                           editable integer-tick timeline
                                    |
                 USD/UsdSkel master + glTF/VRM/browser delivery
```

Ownership order is mandatory:

1. Lipsync owns visemes, jaw/aperture, and lip contacts.
2. Source video owns observed face, head, translation, and gaze in video-follow
   mode.
3. Acting direction adds bounded affect, gaze intent, posture, and gestures; it
   cannot author speech mouth controls.
4. Body motion owns root, limbs, hands, and foot contacts. The body owns the
   base neck/head transform; GNM contributes additive head micro-motion and eye
   joints.
5. Artist/user overrides are sparse, versioned, and applied last.

Every operation records one exact character revision, input hashes, compiler
version, provider envelope where relevant, and integer timebase. "Use current
character" is resolved to an exact revision before a job starts.

## Current implementation truth table

| Capability | Current repository state | Production truth |
|---|---|---|
| Single image to GNM | MediaPipe landmarks, bounded visible-geometry identity fit, neutral GLB/OBJ/overlay | Working estimate; not metric depth or hidden anatomy |
| Multiple images to GNM | Shared identity solve, optional calibrated cameras and held-out view, UV bake | Working research pipeline; ordinary photos cannot prove perfect likeness |
| Texture | Up to 1024 RGB reconstruction atlas composed in explicit linear-sRGB with one sRGB output encode, plus a CLI import path for complete base/normal/displacement/specular/roughness/SSS/radius/confidence/mask packages; exact subject/revision/identity/UV binding; immutable material revisions; bounded native-8K TIFF validation and deterministic at-most-4K glTF base/normal/roughness/specular projection | Source-precision retention, bounded chunked projection and audio/video rendering work. Recovering measured 4K/8K skin from capture, tagged-profile/chart calibration, semantic masking, selectable browser LOD pyramids, pore/relighting validation, SSS/displacement rendering and artist look-dev approval remain open |
| Audio animation | Rhubarb timing plus native Claire MLX learned motion, 52 skin/16 tongue controls, dense GNM retarget, character contact solve, authored neutral-relative aperture pass and measured lip-order repair | Working structural track; independent phone/perceptual approval and a physical oral rig remain open |
| Video animation | Exact source PTS, MediaPipe face/expression/head/translation/gaze, direct inner-lip geometry, identity-calibrated contact/aperture, and exact source-audio sample/PTS evidence. A separately named conservative repair mode now runs learned audio motion, locks reliable visual acting, repairs only weak/missing lower-face evidence, and supplies dedicated audio-inferred tongue controls | Working deterministic fusion candidate, not a trained multimodal model. RGB mouth-specific uncertainty, visible tongue truth, FACS/gaze ground truth and real audiovisual production qualification remain open |
| Character reuse | Versioned identity/preview/material revision, exact revision transport, consent scope/expiry/revocation, content hashes and HMAC trust root | Working local trust boundary; hosted deployment should move signing to KMS/HSM |
| LLM acting | Codex and Claude terminal adapters, no tools, strict schema and semantic validator, trusted envelope, measured performance windows | Produces an editable proposal; its body/gaze portion is deterministically compiled, but artist approval still follows |
| Full body | 25-joint canonical humanoid, 48 kHz integer-tick body/gaze track, OpenUSD/glTF/VRM mappings, contact constraints, GNM ownership contract, and a real pinned MakeHuman/MPFB neutral-body provider | A validated neutral mesh/skin now exists; character attachment, seam calibration, mocap reconstruction, locomotion, skinned USD/glTF export, corrective deformation and production capture remain open |
| Tongue | Real learned-audio motion reaches GNM and animated GLB; every job now emits all-frame lip/tongue/teeth geometry and GLB reconstruction audits | Structural transfer is measured; exact surface collision, camera visibility, phoneme timing and perceptual approval remain open |

## Character library specification

Each character has a mutable, cryptographically sealed governance manifest and
immutable revision directories. A revision contains:

- one finite GNM v3 `(253,)` identity;
- neutral/textured preview GLB;
- an appearance inventory (base color now; normal/displacement/specular/
  roughness/subsurface slots explicit and empty until measured);
- oral calibration state and validation flags;
- body attachment state and skeleton mapping;
- source job/input hashes and fit status;
- performer/subject, attester, authorized scope, expiry, evidence reference,
  SHA-256 of the uploaded release-document bytes, and optional note;
- SHA-256/byte size for every asset and a revision-manifest digest anchored by
  the sealed top manifest.

New job and character manifests are HMAC-SHA256 sealed with an owner-only key
outside every served directory. Rewriting a file and its adjacent JSON hash no
longer launders it into a revision; rewriting a revision anchor or erasing a
revocation invalidates the top seal. Pre-sealing jobs are usable for viewing,
but promotion or acting reuse requires an explicit `job seal-legacy` attestation
that verifies all current recorded bytes and states that earlier provenance was
not cryptographically proven. In a multi-user deployment the same interface
must be backed by a cloud KMS/HSM and append-only audit/event storage.

Consent is authorization, not a label. Every audio/video/direction job declares
`personal`, `research`, `production`, or `commercial` intended use; resolution
fails if the revision scope does not grant it. Revoked, expired, invalid, or
tampered characters cannot drive a job. The UI must never offer an expired
character as active.

Multiview character revisions now seal the exact repacked triangle-corner UV
layout alongside the base-color atlas. Audio and video exporters receive that
layout instead of falling back to GNM's original UVs; using the right PNG with
the wrong atlas coordinates is therefore an integrity failure, not a visual
warning.

Complete material imports create a new immutable child revision. Before any
copy occurs, a versioned `autoanim.material-attachment.v1` envelope binds the
validated package digest to the exact character ID, revision-manifest digest,
identity digest, GNM topology, triangle count, canonical float32 UV-array
digest, UV/normal convention, material semantics and same-subject evidence.
The operator must name the package subject and explicitly attest both
same-subject identity and exact-revision authorship; the template never infers
those claims merely from the character's existing consent record.
The current specular semantic is explicitly a linear RGB multiplier over
glTF's dielectric F0, not an absolute measured F0 map; an absolute-F0 package
needs a future `KHR_materials_ior` conversion before it can attach.
Imports use a per-character filesystem lock and compare-and-swap against the
current revision. Source files are opened without following symlinks, rehashed
while copied, and retained byte-for-byte at source precision. Material-package
v2 records encoded and decoded byte counts, dtype, dimensions, segment counts,
per-file hash and the decoder strategy. High-resolution TIFFs must be a single
2D `YX`/`YXS` image with interleaved samples, top-left stored row order, no
depth/SubIFDs, supported lossless compression (`none`, Deflate or PackBits), at
most 16,384 bounded strips/tiles, and no decoded segment over 64 MiB. Decoded
files are capped at 1 GiB each and 6 GiB per package. PNG remains supported only
when its whole decoded image fits the 128 MiB resident budget; native 8K RGB(A)
PNG therefore fails closed with instructions to use bounded tiled TIFF.

TIFF validation and projection decode into unlinked, capacity-checked scratch
files, scan/process bounded row chunks and never memory-map the mutable package
file. Deterministic browser PNG derivatives use a power-of-two box filter:
base color is averaged in linear light, tangent normals are vector-averaged and
renormalized, roughness stays linear, and linear specular is encoded to sRGB for
`KHR_materials_specular`. The tangent-normal green channel is reflected once
for the UV-origin conversion. Source displacement, SSS/radius, confidence and
masks remain sealed but are honestly reported as not rendered. Every derivative
in character-material v3 is bound to its source-package digest, exact source
hash/dtype/color-space/resampling metadata, runtime hash/size and projection
profile. The attachment runtime accepts source maps up to native 8192x8192 and
derives one at-most-4096 delivery LOD. It does not claim that the 4K derivative
is native 4K evidence.

The retained executable E2E fixture is a programmatically generated,
non-biometric complete native 8192x8192 tiled-TIFF package. It exercises all
required material slots, validates every native pixel, keeps `pore_resolved`
false, and derives four 4096x4096 glTF textures in 31.09 seconds on the retained
development machine. The measured pytest process peak was 1.054 GiB RSS,
including application/test imports; the comparable tiny-package process peaked
at 0.503 GiB, for about 0.551 GiB incremental peak. It proves the bounded
transport/projection path, not real
skin recovery, likeness, pore fidelity or look-development quality. A
selectable 1K/2K/4K external-texture pyramid, GPU-budget selection, cancellation
and texture disposal are still required before calling the browser path
production-scalable; the current GLB embeds its single selected texture set.

The browser does not accept archive uploads yet. Complete packages can be many
gigabytes, so the first safe interface is local CLI import with file-count,
byte, dimension, pixel and path-depth limits. A future hosted uploader needs
streaming extraction into a quarantine volume, the same safe-open validation,
quota enforcement and an explicit publish transaction.

## Appearance and the "8K pores" requirement

### What ordinary images can do

One good image can constrain visible low-frequency head proportions and front
color. Ordered front, three-quarter and profile images improve identity and
texture coverage. Back-of-head images help the atlas, but GNM identity still
depends on correspondence/calibration quality. Uncontrolled RGB mixes diffuse
albedo, specular highlights, shadows, white balance, and camera response; an
8K resize of that signal is not an 8K measured skin material.

### What a high-resolution production capture needs

The recommended capture tier is a calibrated neutral-performance session:

1. Rights/consent and color chart; fixed focus/exposure/white balance; camera
   intrinsics and multi-view extrinsics.
2. Cross- and parallel-polarized flash sequences in a dark/controlled room.
   Polarization separates diffuse and specular response rather than baking a
   highlight into albedo.
3. Neutral and a small expression calibration set, including teeth and visible
   tongue references.
4. Joint optimization of GNM identity/camera plus a higher-frequency neutral
   surface residual. GNM identity should remain the editable low-frequency
   layer; pores belong in measured normal/displacement, not forced into 253
   coefficients.
5. Bake color-managed UDIM maps: diffuse/base color, specular color or weight,
   roughness, tangent/object-space normal, scalar displacement, thickness/
   subsurface inputs, masks, and confidence/observed coverage.
   Normal-map representation is mandatory metadata: `unorm` for `[0,1]`
   integer/float pixels or `signed_float` for float `[-1,1]` pixels. AutoAnim
   does not guess this from image minima. The attached GNM contract is tangent
   space, positive Y, lower-left UV; the glTF derivative reflects green once
   when V is converted and clamps atlas sampling at its edges.
6. Author a MaterialX/OpenPBR master and derive a glTF preview material. OpenPBR
   is a practical interchange baseline, but its own specification notes that
   specialized high-end skin may still need a renderer-specific network.
7. Validate against held-out polarized views and relighting, not the training
   frames. Report texel coverage and error by map; pore claims require a
   calibrated macro/scan reference and target viewing distance.

The CVPR 2023 polarized-smartphone system is strong evidence that inexpensive
capture can recover high-resolution normal plus diffuse/specular appearance,
but that paper's trained reconstruction is not present in this repository. It
is a research direction/dependency, not a feature we already have.

### Validation data and licensing result

No audited public dataset simultaneously provides a real subject, small
deterministic download, synchronized calibrated RGB including distortion,
independent scan truth, measured pore-level reflectance, and commercial-use
permission. The defensible validation split is therefore:

- Meta Multiface subject `6795937`, neutral expression, for an immediate
  calibrated research fixture: 39 RGB cameras with `K`, five-term distortion,
  world-to-camera extrinsics and tracked OBJ, about 3.146 GB. It is CC BY-NC
  4.0, and its tracked mesh is reconstructed from the same capture rather than
  independent scanner truth.
- NeRSemble Benchmark participant 475 after access approval for a smaller
  synchronized research fixture: 13 public cameras and about 325 MB plus one
  point cloud. Its calibration has `K` and OpenCV world-to-camera transforms
  but no published distortion; ingestion must require a source-backed
  `rectified` declaration rather than assuming zero distortion.
- USC Digital Emily 2 for appearance research: actual cross/parallel-polarized
  references, diffuse/specular/single-scatter maps, displacement and
  microgeometry. It is noncommercial and does not document a ready OpenCV
  calibration bundle or guarantee native 8K maps on its public page.
- One separately consented in-house capture for every commercial acceptance
  claim. Public research faces and derived artifacts must not ship in product
  fixtures, screenshots, builds, or customer demonstrations.

Fixture manifests must fail closed on unknown rights and retain source URL,
archive hash, license snapshot/hash, subject, allowed purpose, retention, and
`commercial_allowed`. Calibration gates require explicit axes/transform
direction, valid rotations and intrinsics, explicit distortion or rectification,
at least five fit plus two untouched holdout views, median reprojection below
2 px and p95 below 5 px. A pore-level claim additionally requires native
`>=8192` maps (not upsampling), measured mm/texel, raw polarized or independent
reflectance truth, unseen-light rendering, and preservation of approximately
0.1–0.5 mm spatial detail.

### Likeness audit decision: evaluate before increasing resolution

The current single-image solve is a 68-point weak-perspective fit restricted to
10 or 20 identity modes. Multiview can optimize all 170 head-supported modes,
but still uses the mapped 68 points as identity evidence; modes `170:253` have
no landmark support and the dense 478-point face is currently used for texture
coverage rather than dense surface fitting. A larger atlas would therefore make
the present geometric mismatch and baked illumination sharper, not make the
person more accurate.

The next appearance/identity vertical slice must start with one consented real
subject and then expand to at least ten subjects. Each subject needs five or
more fit cameras, two physically distinct held-out cameras, at least 120
degrees of yaw, two repeated rig sessions, raw ChArUco observations, lossless
or RAW color with chart/profile evidence, metric scale, and an independent
neutral scan. Polarized/multilight sequences are additionally required for any
reflectance or pore claim. Fit/held-out membership and hashes are frozen before
optimization; changing a held-out input must leave the identity artifact
bit-identical.

The 68-point solve remains the initializer. The production candidate adds
part-aware dense correspondences or normal/UV evidence, profile silhouettes,
explicit ear/nose/eye/mouth/cheek/jaw/scalp losses, semantic occluder masks,
shared calibrated cameras, uncertainty/effective-rank reporting and a bounded
normal-direction residual surface stored separately from GNM identity. Pilot
gates are recomputed calibration RMS `<=0.40 px`, held-out reprojection median
`<2 px` / p95 `<5 px`, no-scale scan point-to-surface median `<=1.0 mm` / p95
`<=2.5 mm`, normal error median `<=8 degrees` / p95 `<=20 degrees`, repeated-fit
median vertex drift `<=0.5 mm`, p95 `<=1.5 mm`, and coefficient saturation
`<=10%`. Both GNM-only and GNM-plus-residual results must pass independently;
residual geometry cannot hide a failing parametric identity.

Appearance should expose two proposed product labels (the current manifest
schemas do not yet encode these names). Arbitrary portraits would produce
`baked_rgb_nonrelightable`. Only controlled capture may produce measured
diffuse/specular/normal/roughness/displacement maps. A measured skin atlas must
be at least 90% observed, 0% generic, at most 5% inpainted, record native
mm/texel, achieve chart delta-E00 median `<=2` / p95 `<=4`, and pass held-out
light rendering before `relight_validated=true`. Hair stays a separate
strand/card/volume asset; generated rear views are labeled inferred and never
count as likeness evidence.

Suggested tiers:

- **Preview:** current 1K RGB atlas, explicitly unvalidated and non-relightable.
- **Studio real-time:** 4K/8K UDIM color + roughness + normal, held-out
  relighting error and confidence masks.
- **Cinematic close-up:** calibrated displacement/normal frequency split,
  subsurface/thickness, eye/tearline/hair assets, renderer-specific shader,
  and artist look-development approval.

## Audio, video, acting, and the mouth

### Audio mode

The learned path analyzes real audio with Claire, conditions 52 skin and 16
tongue controls, uses released geometry solves to retarget into all 383 GNM
expression coefficients, then adds bounded affect/secondary motion. The
procedural Rhubarb fallback is deterministic and now uses the same
character-specific bilabial contact calibration. Contact anchors are restored
after the temporal limiter so closure is not silently erased.

The optional mouth-opening control is an authored, identity-calibrated geometry
edit, not a global coefficient multiplier. It changes only GNM lower-face modes
`200:350`; P/B/M labels, learned contact evidence and existing contact anchors
are hard vetoes. Upper-face, tongue and reserved coefficients, pose and timing
remain byte-identical. Because GNM uses a PCA basis rather than spatially local
blendshapes, lower-face modes can leave a small tongue-vertex tail; that tail is
measured and bounded rather than described as exact locality. Final face-local
mouth steps are constrained without smoothing the whole take, and an inverted
inner-lip pose is minimally projected toward the character neutral while
preserving tongue and upper-face coefficients.

On the retained real learned-audio clip, a requested `1.08` aperture gain
changed 132/211 frames. Twenty-eight local frames were continuity-limited,
83.3% of changed frames reached the full requested geometry target, the final
step maximum was `0.03995` interocular, one inherited lip-order inversion was
repaired, and the final reports measured zero lip-order and tongue/teeth
proximity risks. GNM tongue controls and isolated tongue geometry were active
on 209/211 frames. This is structural evidence, not independent phoneme,
surface-collision, visibility or perceptual approval.

The next audio backend is not another smoothing layer. It is an external,
GPU-capable Audio2Face v3 sequence worker behind the already versioned sequence
contract; the Mac service remains the evidence/retarget/orchestration host. The
worker must bind model, runtime, PCM, character identity, retarget calibration,
chunk overlap/state and output timebase hashes. The current local v2.3 path
remains useful as an offline candidate and regression oracle, not the production
claim.

Qualification freezes at least 40 rights-cleared utterances from at least ten
speakers, with independent phone intervals and visible P/B/M, F/V, rounded-vowel
and open-vowel apex labels. Release gates are median apex error `<=1` video
frame, p95 `<=2`, P/B/M contact recall `>=90%`, false contact `<=5%`, no added
lip-order/tongue-teeth structural risk, and blinded artist preference `>=60%`
with the confidence interval above 50% against the current v2.3 baseline.
Sequence onset/release, fast speech, coarticulation, non-speech vocals,
multilingual material and deliberately shifted audio are scored separately; an
aggregate pass cannot hide a failed subgroup.

An LLM does not replace the acoustic model. AutoAnim sends it a bounded summary
of measured energy, speech activity, pitch/accent and existing motion windows,
plus optional transcript and direction. The output is declarative beats with
intent, valence/arousal, stance, gesture tags, face tags, gaze and constraints.
Codex runs ephemeral with user config/rules ignored and its shell, unified
execution, browser, computer, apps/plugins, multi-agent, image-generation and
workspace-dependency tool features disabled before the untrusted prompt is
written; Claude runs safe-mode with no tools, MCP, settings, sessions, or
permission prompts. Tool events remain a post-run rejection layer as defense
in depth. Both are schema and semantically validated. Provider logs and
proposal hashes are retained; lip controls remain owned by the deterministic
solver.

### Video mode

Video has two explicit policies. The default API/CLI policy remains
`video_follow`; its motion is derived from **visual frames**, not the audio
track:

- MediaPipe blendshapes drive expression;
- inner-lip landmark geometry drives contact and open-mouth aperture;
- transform matrices drive head rotation and translation;
- eye-look controls drive the two GNM eye joints;
- exact decoded PTS is preserved;
- source audio is copied only into the proxy/playback clock.

The browser can request `audio_visual_repair`. That named revision first runs
the learned Audio2Face v2.3 source in neutral-affect mode, then consumes the
exact retained-source sample/PTS join. It never replaces head rotation,
translation, gaze/pupil controls, upper-face acting, reliable lip geometry or
visible contact. It may repair GNM lower-face modes `200:350` only where the
full-face observation is missing or below the frozen global tracker-quality
threshold; the current tracker does not expose a mouth-specific uncertainty.
It may supply
dedicated tongue modes `350:382` during speech because the current RGB capture
has no dedicated tongue channel. Trusted visual ownership and observed contact
are hard protections; audio/visual contact disagreements on trusted frames are
diagnostics, not a separate causal veto. Every input, resampled control, weight and output is stored
in `audio-visual-repair.npz`; the JSON artifact binds source PTS, native PCM,
input/audio/output controls and the authority locks by SHA-256.
The repair resolves Claire's pretrained MLX bundle to a concrete local
directory, passes that directory explicitly to the native runner, and hashes
all model files before and after inference. Its source manifest also binds the
runner, Metal library, Rhubarb executable/resources, all dense-retarget assets,
the selected primary audio stream and every retained intermediate. Production
readiness independently byte-verifies those causal artifacts and the original
audio/video timing report. It additionally requires a separately retained
qualification-profile artifact whose ledger hash exactly matches the profile
hash claimed by the repair; setting approval booleans or an arbitrary hash is
insufficient.

The earlier retained CREMA-D run had 0.974 lip-opening timing correlation but
an affine amplitude slope of 0.582: roughly 40% under-opening. The current
identity-calibrated retarget plus authored `1.08` aperture pass reaches 0.988
correlation, 0.939 slope and 0.939 p95 source/output amplitude ratio on the same
retained capture. It corrects 35/67 frames, hard-protects 25 contact/evidence
frames, and vetoes four frames adjacent to source motion above 0.08
interocular/frame rather than smoothing the captured transition. It preserves
exact source PTS and introduces no lip-order inversion. The
acceptance gate is correlation `>=0.90`, slope `0.90–1.10`, and open-frame p95
ratio `0.90–1.10`. One soft contact target remains below the 95% attainment
review threshold even though all high-confidence source contact events are
retained, so the take is still review-required.

Video-follow tongue remains zero because the pinned result schema does not
expose or ingest a tongue channel and the RGB lane has no pixel-derived tongue
surface solve. Repair-mode tongue is audio-inferred and therefore retains
`tongue_visible_validated=false`; it is not evidence that a tongue seen in the
source was reconstructed. Contradictory media (silent expressive video,
expressive audio with neutral face, and dubbed/misaligned video) remain required
qualification fixtures; default video-follow continues to honor the visible
performance byte-for-byte.

### Video fidelity audit: what the implementation follows today

The current video lane is a real visual-performance retargeter, not audio-only
lipsync hidden behind a video upload. `video_capture.py` decodes every display
frame, supplies MediaPipe VIDEO mode with strictly increasing timestamps, and
preserves the source PTS. `video_retarget.py` uses the tracked inner lips,
blendshapes, facial transform and eye-look channels. `video_pipeline.py` copies
audio into the review proxy in `video_follow` mode. When the separate repair
flag is requested it also creates a learned neutral audio track and routes it
through `audio_visual_repair.py`; learned acoustic mouth/tongue motion is then a
constrained repair source. Prosodic speech activity and Rhubarb mouth-category
cues are causal inputs, but no lexical transcript is generated and automatic
audible affect still cannot overwrite visible acting.

The exact-timing transport is stronger than the semantic evidence:

- `CaptureTrack` stores 478 landmarks, the pinned 52-column result schema
  (`_neutral` plus 51 expression/gaze channels), facial transforms and exact
  PTS. Frame/PTS count mismatches and non-monotonic timestamps fail closed.
- The preferred frame confidence is median landmark visibility/presence when
  MediaPipe exposes it. Otherwise `effective_capture_quality` is only the
  fraction of landmarks inside a generous image bound. That fallback does not
  measure focus, motion blur, mouth/eye occlusion, crop resolution, landmark
  reprojection error, face identity continuity or whether a particular facial
  region is trustworthy.
- The adaptive filter preserves fast blink/contact controls and decays missing
  data, but it consumes one global quality value. There is no confidence per
  mouth, eye, brow, cheek or blendshape channel. Consequently a hidden mouth
  and a clear brow can be accepted or rejected together, and microexpression
  retention cannot be distinguished from tracker jitter. This is a production
  ship blocker, not a tuning issue: the repair candidate may decline to repair
  a genuinely hand/prop/facial-hair-occluded mouth. Viable next implementations
  are retained-pixel mouth/occluder segmentation plus temporal reprojection
  residual, a depth/TrueDepth or synchronized multiview lane, or a trained
  audiovisual model with calibrated uncertainty. Until one passes a real
  occlusion set, the UI and schema intentionally call the current gate a global
  tracker-quality heuristic.
- Dense retargeting is geometry-calibrated when the Claire calibration assets
  are installed. It is still a mapping from MediaPipe's semantic coefficients,
  not a subject-trained performance solve. `mouthClose` is deliberately
  quarantined because its calibrated direction opens this GNM rig; direct
  inner-lip geometry instead owns closure and aperture. That repair is useful,
  but it is not a jaw hinge, lip collision model, teeth constraint or observed
  tongue solve.
- Head rotation/translation comes from a baseline-relative canonical face
  transform. Translation has an approximate canonical scale, not calibrated
  performer metric scale. Eye rotation comes from four eye-look coefficients
  and a fixed 25-degree range, not an iris/eyeball calibration or known gaze
  target; it is not gated by blink/eye occlusion.
- Neutral correction searches for a low-activity initial window and disables
  itself when none is credible. On the retained angry CREMA-D take more than
  40% of one-sided negative baseline residuals are clipped, so the current
  baseline must remain an auditable heuristic rather than a production-neutral
  claim.
- The installed result schema and `MEDIAPIPE_BLENDSHAPE_NAMES` do not ingest a
  `tongueOut` channel, and the pipeline derives no tongue surface evidence from
  pixels. Google's blendshape model card lists `tongueOut`, while Apple's ARKit
  exposes a dedicated `tongueOut` coefficient; this makes sensor/model/schema
  capability negotiation a required ingest feature, not a reason to fabricate
  tongue motion in the current RGB lane.
- The browser uses source media time as the GNM animation clock. Observation-v2
  jobs now expose exact previous/next source-frame stepping, raw PTS, the 48 kHz
  project tick, and mouth/eyes/upper-face/head confidence with explicit
  missing/unknown state. The source video is still a small reference panel; no
  locked source/output camera, landmark overlay, confidence curves, gaze rays,
  neutral-window display, or jump-to-conflict review exists yet. Artifact
  timestamps can therefore be exact while subtle visual errors remain hard for
  an artist to diagnose.

The verification command and current result are recorded in the phase ledger
below instead of as a hand-maintained test count. The new real-input gate decodes the
retained CREMA-D video, tracks it with MediaPipe, runs its real audio through
learned A2F skin/jaw/eye/tongue inference, joins both streams through native
exact display timestamps relative to decoded-audio start, verifies native PCM
sample coverage, and validates the resulting GNM controls and
GLB. It proves video-owned upper face/head/gaze, active dedicated tongue
transfer, zero reported tongue/teeth proximity-risk frames on that take, and
fail-closed learned-source selection. CREMA-D supplies no frame-level FACS,
gaze target, lip-contact, tongue surface or dense 3D ground truth, so those
tests do not establish production expression accuracy.

### Constrained audio-visual performance repair: first pass implemented

Do not average audio and video indiscriminately. The exact clock-join evidence
and first named `audio_visual_repair` policy are now implemented: high-confidence
visible performance stays authoritative and learned audio may repair only
speech-correlated lower-face controls when image evidence is weak. Audio may
also supply dedicated tongue modes, which RGB cannot observe. Default
`video_follow` remains visual-only.

The versioned observation contract should use the source video time base as the
master and record, for every display frame, `source_pts`, rational time base,
48 kHz project tick, decoded frame index and corresponding audio-sample span.
Audio events retain exact start/end samples rather than being rounded to a
nominal frame rate. A sync report records measured A/V offset, confidence,
drift and all resampling/transcode transforms. No fusion is allowed after a
missing timestamp, non-monotonic PTS, frame-count mismatch, or unexplained
offset larger than one source-frame interval; the take remains usable in named
visual-only mode with a review flag.

Each visual observation must carry global and regional evidence, not a single
plausibility score:

| Evidence | Required measurements | Ownership when valid |
|---|---|---|
| Face/identity | detector score, face crop size, blur/exposure, identity continuity, transform jump | Select one continuous performer or fail closed |
| Mouth | inner/outer-lip visibility, crop pixels, landmark temporal and image residuals, aperture/contact/pucker geometry | Visible lip contour, asymmetry, closure and aperture |
| Eyes/gaze | iris/eyelid visibility, blink state, pupil/eyeball solve residual, head/eye separation | Visible blinks, squint/wide and calibrated eye direction |
| Brows/cheeks/nose | region visibility and motion residual per semantic channel | Visible non-speech expression and microexpression |
| Head | transform residual, pose range, crop margin and calibration scale | Rotation; translation only when its capture tier is calibrated |
| Audio speech | VAD, phone/viseme span, alignment confidence, energy and voicing | Coarticulation prior and repair of occluded speech mouth controls |
| Audio acting | timestamped prosody/affect evidence with confidence | Suggestion/diagnostic only when source video visibly disagrees |
| Tongue | explicit sensor coefficient or visible oral landmark/surface evidence | Tongue only for measured visible motion; otherwise unknown |

Initial routing thresholds are explicit preview defaults, not learned truths:
regional visual confidence `>=0.75` preserves video; `<0.45` is unknown and may
accept audio repair only when phone confidence is `>=0.80` and the sync gate
passes; the middle band uses a constrained solve biased toward video. These
thresholds must be calibrated on the held-out labeled set before a production
claim. Unknown observations are never encoded as neutral zero.

Fusion is control-family aware:

1. Head, visible upper-face acting, asymmetric expressions and visible gaze are
   video-owned. An LLM may label or propose edits but cannot overwrite them.
2. At a clear mouth, video geometry owns contour and contact; audio supplies a
   soft phone/coarticulation constraint. At an occluded or blurred mouth, audio
   may repair jaw/aperture and speech lip controls inside the affected span.
3. A visual/audio disagreement in contact state lasting more than
   `max(one source frame, 40 ms)` emits a conflict interval. Default resolution
   is visible video ownership, not coefficient averaging. Dubbed/misaligned
   media therefore becomes review-required rather than silently "corrected."
4. Audio may condition unobserved interior tongue timing, but RGB output must
   keep `tongue_visible_validated=false`; visible tongue animation requires an
   explicit supported capture signal and annotated validation.
5. All filled, held, decayed, fused, clamped and artist-overridden values retain
   their source/confidence and reason code through the final GNM frame.

Deliver the phase in four build/review/test loops:

1. **Observation v2 + exact A/V clock join (implemented):**
   `performance-evidence.json`
   now preserves raw PTS/timebase, exact rational and nearest 48 kHz ticks,
   explicit observed/missing/unknown states, and conservative mouth/eyes/
   upper-face/head confidence from existing tracker evidence. Geometry-only
   confidence is capped at 0.5 and the artifact is explicitly not consumed by
   retargeting. `audio-video-timing.json` additionally hashes a deterministic
   native-rate mono PCM decode and maps every exact display interval to its
   covering sample span, with rational offset/drift and complete-coverage
   evidence. FFprobe and FFmpeg consume one descriptor-copied, read-only
   snapshot; probe output and the final artifact are capped at 64 MiB, tool
   failures redact filesystem paths, and the snapshot hash is rechecked after
   both tools finish. On the retained CREMA-D source it measures a `-27 ms` audio start
   offset and `+37.653061 ms` duration drift. The fusion gate remains blocked
   because no phone/audio semantics have been classified, and the artifact is
   diagnostic-only. Remaining work is identity continuity, pixel-derived
   blur/occlusion, mouth/eye crops and reprojection residuals. Test
   variable-frame-rate, B-frame, rotation, blur, occlusion,
   multiple-face, cut and missing-frame fixtures; exact source PTS must remain
   bit-identical.
2. **Constrained fusion (deterministic first pass implemented):**
   `audio_visual_repair.py` validates learned `[T,383]` controls, samples at
   exact video display start minus decoded-audio start while requiring native
   sample coverage, resamples without time warp, locks video
   head/translation/gaze/pupil/upper-face controls, protects reliable visual
   lips and contacts, repairs only weak/missing lower-face frames, and supplies
   active dedicated tongue controls. A GNM face-local geometry limiter may only
   reduce repair weights inside weak intervals; it cannot smooth or modify a
   trusted video frame. It emits hash-bound JSON/NPZ evidence and
   fails when learned A2F is unavailable instead of silently using Rhubarb.
   Unit adversaries cover fallback rejection, PTS tampering, partial audio
   coverage, contact conflict and authority locks. The retained real CREMA-D
   video+audio E2E passes. Remaining: pixel-derived regional uncertainty,
   conflict intervals rather than frame flags, silent-expression,
   neutral-face-speech, dubbed-offset and deliberate mouth-occlusion real
   fixtures, and comparison with a trained audiovisual model. High-confidence
   video controls must remain bit-identical in all of them.
3. **Artist review viewer (foundation implemented):** the viewer now steps the
   source and paused GLB action to the exact Observation-v2 timestamp and shows
   PTS/tick/time, regional confidence, and missing/unknown state. Remaining:
   synchronized side-by-side and overlay modes, locked camera, mouth/tongue
   close-up, region/confidence/control curves, gaze rays, and jump-to-conflict
   flags. Browser tests must compare requested PTS with the sampled GLB frame
   and capture deterministic screenshots of flagged real frames.
4. **Ground-truth acceptance:** a consented calibration take with a real
   neutral segment, phone/contact annotation, known gaze targets and calibrated
   head pose, plus a separate natural acting take and occlusion stress take.
   Report by region and capture condition; do not collapse the result into one
   average score.

Acceptance gates for the resulting phase are: exact source PTS and audio-sample
lineage; unexplained A/V drift below one source frame; no overwrite of
high-confidence video-owned controls; bilabial contact precision/recall
`>=0.95` within one annotated frame; open-mouth aperture correlation `>=0.95`
and p95 amplitude ratio `0.90–1.10`; strong blink precision/recall `>=0.95`;
known-target gaze median angular error `<=5 degrees`; calibrated head rotation
median error `<=2 degrees`; expression/microexpression F1 reported per FACS
action and visibility slice; zero false visible tongue events; and artist
approval of every automatically flagged conflict interval. Transport metrics
from CREMA-D remain regression gates, not substitutes for these labels.

### Tongue evidence and remaining oral rig

The retained real 49.728-second learned-audio job has 1,492 frames, nonzero GNM
tongue motion on 1,490, a maximum conditioned tongue control of 0.488790, and a
maximum final GNM tongue coefficient of 1.075043. Isolated GNM tongue motion is
1.3502 mm p95 / 4.7114 mm maximum. The animated GLB maps all 933 native tongue
vertices (967 seam-split render vertices), all 19 morph targets move tongue,
and peak-frame tongue reconstruction error is 0.0291 mm p95 / 0.0364 mm max.

That proves data flow, not perfect animation. Lateral/roll calibration errors
remain roughly 0.52–0.67; the normal MP4 excludes oral anatomy; no phone-aligned
tongue corpus, visibility review, palate/teeth collision, or rigid jaw exists.
Production oral work therefore needs a virtual jaw hinge, rigid lower teeth,
tongue-root/tip ownership, mouth-sock/palate collision SDFs, an oral preview,
and annotated `/p b m f v t d n l s z th r w/` plus open-vowel tests.

The integrated validator now evaluates all required GNM lip, tongue, upper- and
lower-teeth vertices for every control frame. It reports interocular-normalized
lip gaps, lip-order inversion risk, tongue/teeth nearest-distance risk,
face-local tongue displacement, target-contact attainment and isolated
tongue-control transfer. A second report reconstructs the actual GLB and, when
the viewer is animated, compares it with the source controls at every frame.
GNM's oral meshes are open surfaces, so these are proximity and ordering
proxies, not signed penetration tests. The report deliberately keeps
`phoneme_correctness_validated`, `perceptual_correctness_validated`,
`penetration_free_validated` and `production_validated` false.

The GLB exporter also enforces the control track's oral semantics during
low-rank compression. It prioritizes the landmark-support vertices used for
face scale and inner-lip geometry, then rejects a candidate rank if it changes
any contact classification or introduces a signed lip-order risk. This closed
a real retained-video defect where a globally acceptable rank-13 approximation
created five viewer-only inversions. Re-exporting the same 67-frame controls
selected rank 29, kept all twelve source contact frames, introduced zero
inversions, and passed at 0.0187 mm p95 / 0.0933 mm maximum oral error. Public
job results report both control and viewer counts and conservatively aggregate
them for readiness checks.

The same video audit also establishes a separate tongue limitation. The source
contains no dedicated tongue observation and all 32 dedicated GNM tongue
coefficients remain zero, yet GNM's lower-face statistical modes have support
on tongue vertices: 62/67 control frames exceed 0.1 mm face-local tongue
motion and the maximum is 12.56 mm. This is not captured tongue acting and is
not primarily caused by the aperture edit (its maximum incremental tongue tail
is `2.27e-6` interocular on this take). Video results therefore label the motion
`gnm_lower_face_basis_coupling_no_dedicated_source` and emit
`ORAL_UNSOURCED_TONGUE_BASIS_COUPLING`. A dedicated tongue signal, constrained
tongue-neutral projection, or artist pass is required before visible-video
tongue performance can be approved.

## Body choice and head connection

GNM Head has no released body. Do not silently call SMPL-X or MetaHuman an
open, drop-in GNM body:

- SMPL-X is technically attractive and unifies body/hands/face, but the model
  and software terms require a license review. The separate **SMPL-X Body**
  subset is CC BY 4.0 and excludes the shape blendshapes/tools needed to create
  parametric body identities.
- MetaHuman is production-capable but governed by the Unreal/MetaHuman license
  and its own rig ecosystem. It is an optional adapter, not the canonical open
  core.
- Rigify is a GPL Blender add-on that can generate an artist control rig; its
  generated asset workflow is useful for correction, but Rigify is not a body
  identity model.

The selected production-open baseline is MakeHuman's hm08 core mesh and CC0
system/body targets, authored through a pinned Blender 4.5 LTS + MPFB + Rigify
worker. MediaPipe Pose Heavy and Hand Landmarker provide timestamped video
observations; AutoAnim owns calibration, confidence filtering, constrained IK,
root/camera separation, contact solving and retargeting. OpenUSD/UsdSkel is the
editable master and glTF 2.0 plus VRM 1.0 is the runtime form. This choice gives
the project a commercially auditable parametric starting body, not film-quality
deformation: shoulders, hips, elbows, knees, wrinkles and muscle effects still
need corrective shapes and artist review.

Blender, MPFB and Rigify remain a separate GPL authoring process. AutoAnim
writes a versioned request and reads neutral mesh, deform bones, weights, morph
targets and baked motion through files; it does not import GPL Python modules
or depend on Rigify control-bone names at runtime. A desktop distribution that
bundles the worker must include applicable licenses, notices and corresponding
source. Community MakeHuman assets remain deny-by-default unless their
individual license is recorded. SMPL-X is a future premium provider only after
a written commercial agreement covers inference, storage, generated outputs
and redistribution. WHAM is excluded from the production baseline because its
MIT code still depends on separately restricted SMPL/SMPLify data and external
checkpoints.

The open core should accept a rights-cleared humanoid mesh/skeleton and motion,
with these contracts:

- Canonical skeleton paths, parent indices, rest transforms, units in meters,
  +Y up / +Z forward, quaternion normalization, and stable joint IDs.
- Required pelvis/spine/chest/neck/head/limbs/feet; optional hands/fingers.
- Body owns root through base head. GNM neck/head are additive deltas in the
  shared head bind space; GNM eyes remain face-local.
- One watertight/hidden seam strategy, matched skin tone/material, scalp/neck
  normals, and LOD boundaries. A transform-only socket is insufficient for
  close-ups.
- Foot-contact constraints are hard anchors. LLM plans may say "guarded" or
  "small open palm"; a deterministic compiler maps those tags to bounded
  curves and rejects unavailable gestures.
- OpenUSD/UsdSkel stores the editable skeleton, skin, joint animation and face
  blendshape channels. glTF/VRM carries a runtime humanoid, animation and
  conventional expressions. VRM's procedural override rules reinforce the
  need to separate lipsync/blink/gaze from emotional expressions.

The current Phase-4 foundation implements that contract as 25 parent-ordered
joints, canonical inverse binds, UsdSkel paths, glTF nodes and VRM humanoid
roles. Every successful acting-direction job now also emits a compressed
numeric `body-track.npz`, a small manifest, the skeleton, and the GNM
attachment/ownership document. The track uses 48,000 integer ticks per second,
bounded smooth transitions, immutable normalized quaternions, GNM eye-local
residuals, and explicit foot contacts. It is labeled an **unapproved preview**:
the present UI has no beat editor/approval transaction, so publish must not
treat it as an approved compilation. Takes are preflight-limited to 30 minutes
before invoking the LLM. It is upper-body procedural direction only: there is
no motion estimation from the source video, locomotion, or performer-approved
body performance, and acting jobs do not yet bind/export the provider mesh.

The provider boundary is implemented and has now executed against real pinned
dependencies. `autoanim.blender-body-request/1.0` pins Blender
4.5.11, MPFB 2.0.16, hm08, the default deformation rig, the reviewed 25-joint
map, and caller-pinned CC0 system-asset bytes. The isolated Blender worker uses
MPFB's real `HumanService` path and exports a neutral mesh, canonical rest and
inverse-bind transforms, collapsed skin weights, and an explicitly
uncalibrated GNM head socket. The application-side validator independently
checks strict JSON/NPZ layout, request/artifact hashes, units/axes, hierarchy,
rigid matrices, weights, geometry, seam ownership, and license provenance.
Those content hashes are not remote-worker authentication.

The project-local bootstrap verified Blender's publisher SHA-256, Apple code
signature and notarization, pinned MPFB 2.0.16 extension bytes and recorded its
upstream Git commit (without proving the archive/commit relationship), and
verified byte-identical MakeHuman system-asset archives from two official
mirrors. MakeHuman publishes no signed archive checksum, so that last digest is
recorded as corroborated rather than publisher-signed. The retained real output
contains 13,380 vertices, 26,756 triangles and 25 joints, is 1.65943 m tall,
uses at most six aggregated canonical influences, and has weight-sum and
inverse-bind errors of `1.19e-7` and `1.23e-7`. Its artifact SHA-256 is
`b42f2a9cda5f8e6138fdc3d93c918dd799c627d157d7e23233c30476ca2bf621`.
Production requests reject any other caller-selected system-pack digest. The
isolated installer records complete stable-file digests for the installed MPFB
extension and MakeHuman data trees, and the worker recomputes both before body
generation; a stale or subsequently modified profile therefore fails closed.
This proves a reproducible neutral mesh/skin provider. GNM/body seam
calibration, character-revision attachment, body motion estimation, corrective
deformation and combined skinned face/body export remain unimplemented and
must not be inferred from the provider smoke result.

The stable 25-joint schema may be extended, without renumbering the core, by 15
VRM finger bones per hand. Body owns root through macro head pose; GNM owns
expression, lipsync, oral anatomy and face-local eye rotation. Final local
rotation is `normalize(Qbody_base * Qgnm_additive * Qartist_additive)`, and the
macro source head pose must be written exactly once. A production character
revision also needs a measured GNM/body bind calibration, matched or hidden
neck seam, partitioned weights and material/normal blending.

Body implementation gates are intentionally real-input gates: three distinct
MakeHuman proportions must round-trip with bind/inverse-bind error `<=1e-6`,
normalized quaternions within `1e-5`, weights summing to `1+/-1e-4`, neutral
vertex RMS `<=0.1 mm` and maximum `<=0.5 mm`, zero USD/glTF/VRM validator
errors, neck seam gap `<=0.25 mm`, and unchanged GNM lipsync coefficients. A
consented video set covering walking, sitting, turning, fast/crossed-limb
gestures, occlusion and planted feet must then meet p95 reprojection `<=2%` of
body height, MPJPE `<=60 mm` where truth exists, bone-length variation `<=0.5%`,
contact precision/recall `>=0.90`, planted-foot drift p95 `<=20 mm` and ground
penetration p99 `<=10 mm`. Moving-camera takes stay review-required until a
separately validated camera/root solve passes equivalent ground truth.

## Application workspace specification

The production UI is a project workspace, not four unrelated upload forms:

1. **Characters** — capture jobs, revision history, consent/status, geometry,
   texture map inventory, oral/body readiness, exact version promotion and
   comparison.
2. **Takes** — audio/video ingest, transcription, source-mode selection,
   character revision and intended use.
3. **Direction** — user notes, measured evidence, optional Claude/Codex
   proposal, beat approval/editing.
4. **Timeline** — owned layers (source, lipsync, face affect, gaze/head, body,
   artist overrides), mute/solo, integer ticks and diagnostics.
5. **Review viewer** — synchronized source/3D, orbit and locked cameras,
   texture/material modes, mesh/wireframe, mouth/tongue close-up, source/output
   curves and flagged frames.
6. **Publish** — validation report, license/consent check, USD master, glTF/VRM
   preview, MP4 review, hashes and manifest.

Required nonfunctional properties: local/offline operation, bounded uploads,
no arbitrary file serving, restrictive viewer CSP, single-writer job lock until
a queue is implemented, deterministic artifacts, crash recovery, immutable
revisions, explicit deletion/retention policy, and no hidden network use by LLM
adapters.

## Phased execution and gates

### Phase 0 — preserve and measure the prototype (complete)

- Commit the known-good four-pipeline baseline.
- Pin GNM, native runner/assets, viewer and test fixtures.
- Gate: full baseline suite and real audio/photo/video fixtures pass.

### Phase 1 — character/library and trust boundary (implemented; regression pass active)

- Exact character revisions, identity-aware rig/contact/export.
- Appearance inventory with honest missing-map fields.
- Release-document digest, scope, expiry and revocation enforcement.
- HMAC-sealed job and character trust roots; explicit legacy migration.
- Gate: source/result tamper, revision/top-manifest tamper, traversal, symlink,
  revoke, expiry, scope and historical-revision tests fail closed.

### Phase 2 — unified performance evidence (implemented for face)

- Apply selected identity/texture to audio and video animated GLBs.
- Preserve exact PTS; report that source video, not audio, owns video motion.
- Emit Observation-v2 per-frame video evidence with exact rational/project
  clocks, explicit missing/unknown states and conservative regional confidence;
  keep it read-only so diagnostics cannot silently change retargeting.
- Character-calibrated procedural and learned lip contact.
- Visual video aperture matching and source/output regression metrics.
- Bind independent audio annotations and artist GNM prototypes to exact
  source/character/identity/rig/timebase/provenance hashes and deterministically
  rescore a retained controls track. The v1 scope is apex pose plus motion
  hygiene; sequence start/release and perceptual validation remain false.
- Validate a source-agnostic Audio2Face v3 sequence-worker contract with exact
  model/runtime/identity/schema/audio hashes, rational output timebase,
  overlapping chunks and state provenance. No v3 inference or network path is
  installed on this Mac; the existing v2.3 runtime remains explicitly preview.
- Gate: real fallback contact attainment `>=0.75`, mouth step `<=0.040`; video
  aperture correlation `>=0.90`, p95 ratio `0.85–1.15`, exact PTS/contact.

### Phase 3 — safe acting direction (implemented proposal layer)

- Measured performance windows plus transcript/instructions.
- Codex and Claude no-tool adapters; strict schema and semantic validator.
- Signed provider envelope and edit-ready beats; lipsync override prohibited.
- Gate: prompt injection, tool attempt, malformed/oversized/timeout/refusal,
  overlap and forbidden-field tests; one real successful run per provider.

### Phase 4 — editable body foundation (provider boundary implemented; real mesh/capture open)

- Implemented: canonical skeleton, inverse binds, integer-tick body/gaze track,
  GNM neck/head/eye ownership, foot contacts and deterministic tag compiler.
- Implemented: fail-closed pinned hm08/MPFB Blender request/worker/result
  boundary and independent skinned-body validator. The local dependency audit
  blocks execution, so this is not an attached body asset.
- Remaining: install and attest the exact worker dependencies; produce real
  rights-cleared body fixtures; calibrate the GNM socket/seam; add USD/UsdSkel
  and glTF/VRM skin writers, Pose/Hands IK, locomotion/contact solve, and Rigify
  correction round-trip.
- Gate: hierarchy/rest validation, quaternion continuity, exact ticks, foot
  drift, head seam transform, round-trip error, unsupported gesture failure.

### Phase 5 — measured production appearance (bounded 8K attachment/runtime implemented; recovery open)

- Implemented: automatic multiview RGB baking decodes declared sRGB inputs to
  linear-sRGB before resampling, affine harmonization, blending and fill, then
  applies the sRGB output transfer exactly once. Result metadata records the
  complete color-space contract; confidence and provenance remain unchanged.
- Implemented: fail-closed CLI structural/attestation gating for complete aligned atlas/UDIM map
  inventories, lossless decode, channel/depth/normal semantics, source/native
  resolution, lineage, commercial rights, hashes, and evidence-backed
  native/pore/relightable claims. The output calls these claim gates, never
  perceptual validation; pore-frequency and unseen-light tests remain false.
- Implemented: revision/identity/UV/subject attachment envelope, locked
  compare-and-swap immutable import, source-map retention, deterministic glTF
  derivatives, tangent export and base/normal/roughness/specular propagation
  through neutral, audio and video GLBs. Float normal encoding is explicit and
  atlas sampling is clamped rather than repeated.
- Implemented: package-v2 decoded-byte/segment/depth/codec limits, safe-open and
  unlinked scratch decoding, full chunked dtype/range/normal/alpha validation,
  power-of-two linear-light/vector-aware at-most-4K projection, and
  character-material-v3 source/derivative provenance. A complete synthetic
  native 8192x8192 package passes the real decode/projection E2E; it makes no
  pore or real-subject claim.
- Remaining: freeze the rights-cleared real-person/scan evaluator, add dense
  part-aware GNM identity plus separately scored residual geometry, calibrated
  polarized recovery, a real rights-cleared consented 4K/8K skin package, browser-selectable
  1K/2K/4K external LODs and GPU-budget lifecycle, MaterialX/OpenPBR authoring,
  displacement/SSS renderer adapters and perceptual look-dev approval.
- Gate: held-out geometry/reprojection, diffuse/specular separation,
  relighting error, coverage, seam delta, color chart, pore-frequency
  validation and artist look-dev approval.

### Phase 6 — oral anatomy and audiovisual fusion (conservative fusion implemented; anatomy open)

- Visible oral preview, virtual jaw/rigid teeth, tongue attachment and SDF
  collision; phone-aligned corpus.
- Implemented: all-frame control and GLB structural reports for lips, tongue
  and teeth, including transfer activity, contact, proximity and reconstruction.
- Implemented: the read-only timestamped Observation-v2 foundation specified
  above, exact source-frame stepping and regional evidence readout in the
  interactive viewer, and a hash-bound native-audio-sample/video-PTS join with
  rational offset/drift and typed fail-closed evidence.
- Implemented: optional named `audio_visual_repair` with a required learned
  source, exact display-time/decoded-audio clock joining, visible-video ownership
  locks, globally weak-observation lower-face repair, dedicated audio tongue,
  contact disagreement diagnostics and complete
  revision artifacts. Default `video_follow` remains unchanged.
- Remaining: pixel-derived regional uncertainty, time-interval conflict review,
  trained multimodal fusion, real contradictory/occlusion fixtures,
  side-by-side/overlay review and the physical oral system.
- Gate: rigid-teeth residual `<0.10 mm`, oral penetration p99 `<0.10 mm` / max
  `<0.25 mm`, tongue timing within one annotated frame, no false visibility,
  and the audio-visual acceptance gates defined in the video fidelity audit.

### Phase 7 — production editor and publish (not complete)

- Replace upload dashboard with the six-pane project workspace.
- Layer editing, sparse overrides, approval state, queue/cancel/resume and DCC
  round-trip.
- Gate: real performer project from capture through signed publish; restart,
  tamper, privacy, performance, accessibility and artist acceptance tests.

No phase may advance on mocked unit tests alone. Each phase runs build → code
review → defined tests → real inputs → fix → complete re-run. Failures remain
typed artifacts; `production_validated` stays false until independent data and
artist gates pass.

### 2026-07-19 execution ledger

- Baseline production-workflow safeguards were committed as `a9b48f2` before
  this slice began.
- Build: added the opt-in conservative audiovisual repair, exact PTS/decoded-
  audio clock join, visual authority locks, audio contact propagation,
  dedicated tongue track, complete revision/evidence artifacts, browser/API/
  CLI controls and fail-closed production-readiness gate.
- Swarm review: the audiovisual and likeness reviewers challenged mouth-local
  uncertainty, temporary-source ownership, audio stream selection, downstream
  contact preservation, model-weight provenance and incomplete causal-artifact
  verification. The implementable correctness findings were fixed and
  re-reviewed; the measurement/data gaps remain explicit blockers.
- Focused phase gate: `79 passed, 1 skipped, 1 deselected` across audiovisual
  fusion, A2F, audio, acting, browser/API and readiness contracts.
- Real-input gate: the checksum-pinned CREMA-D video ran through MediaPipe,
  Rhubarb, the explicitly resolved Claire MLX bundle, dense GNM retarget,
  audiovisual repair, final oral/tongue validation and animated GLB export;
  `1 passed in 22.75s`. The exact post-review rerun, including the original
  timing-artifact gate, passed in `23.06s`. A fresh app job then exposed that
  the first combined tongue track exceeded the viewer's 32-target compression
  gate and correctly fell back to static. The exporter now retains bounded
  sparse oral corrective targets when a sub-SVD-threshold residual would
  otherwise flip contact/lip ordering; the same take exports a full animated
  rank-20 GLB and passes all-frame structural oral reconstruction.
- Full post-review regression: `418 passed, 1 skipped, 1 dependency warning in
  382.79s`. The single skip is the duplicate opt-in released-Claire asset test;
  the checksum-pinned real learned video/audio route above ran and passed.
- Honest release state: the implementation is a working candidate, not a
  production approval. Production remains blocked on mouth-local visual
  uncertainty or a trained multimodal model, a rights-cleared labeled A/V
  cohort with phone/contact/tongue truth and blinded artist preference, dense
  consented likeness/appearance ground truth, calibrated body attachment and
  production mocap/skin export.

## Additional uses unlocked

- Reusable consent-aware digital cast and revisioned likeness library.
- Dubbing/localization while retaining the visible actor's performance.
- Previsualization and virtual production blocking with editable face/body
  layers.
- Game/NPC dialogue generation with deterministic runtime exports.
- Accessibility avatars and signed-language research once hand/body evidence
  is added.
- Performance transfer, emotion variants and non-destructive director notes.
- Synthetic training/QA data with exact ground-truth rig channels and
  provenance.
- Telepresence and live-avatar modes after latency/privacy gates.
- Automated continuity checks across takes, characters and localized lines.

## Primary external references

- [OpenUSD UsdSkel introduction](https://openusd.org/25.05/api/_usd_skel__intro.html)
  — skeletons, skinned models, joint animation and blendshape interchange.
- [VRM 1.0 humanoid specification](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm-1.0/humanoid.md)
  and [expression/procedural override specification](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm-1.0/expressions.md).
- [Khronos glTF registry/specification](https://registry.khronos.org/glTF/).
- [MaterialX specification](https://materialx.org/Specification.html) and
  [OpenPBR Surface specification](https://academysoftwarefoundation.github.io/OpenPBR/).
- Azinović et al., [High-Res Facial Appearance Capture from Polarized
  Smartphone Images, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Azinovic_High-Res_Facial_Appearance_Capture_From_Polarized_Smartphone_Images_CVPR_2023_paper.pdf).
- [SMPL-X model/software license](https://github.com/vchoutas/smplx/blob/main/LICENSE)
  and the separate [SMPL-X Body CC BY 4.0 license](https://smpl-x.is.tue.mpg.de/bodylicense.html).
- [Blender Rigify manual](https://docs.blender.org/manual/en/4.1/addons/rigging/rigify/index.html).
- [MakeHuman licensing](https://static.makehumancommunity.org/about/license.html),
  [base-mesh contract](https://static.makehumancommunity.org/about/concepts/basemesh.html)
  and [MPFB Rigify workflow](https://static.makehumancommunity.org/mpfb/docs/rigging_posing/rigify.html).
- [MediaPipe Face Landmarker VIDEO API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarker)
  and [Google's 52-blendshape model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf).
- NVIDIA, [Audio2Face-3D research paper](https://arxiv.org/html/2508.16401),
  [v3 model card](https://huggingface.co/nvidia/Audio2Face-3D-v3.0),
  [SDK](https://github.com/NVIDIA/Audio2Face-3D-SDK) and
  [training framework](https://github.com/NVIDIA/Audio2Face-3D-Training-Framework)
  — the intended sequence backend and deployment boundary; licenses and GPU
  runtime requirements must be accepted explicitly.
- [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)
  — reproducible phone-interval annotation tooling; its alignment remains an
  evaluator input, not automatic ground truth.
- Chen et al., [Joint Audio-Video Driven Facial Animation, ICASSP
  2018](https://research.snap.com/publications/joint-audio-video-driven-facial-animation.html)
  — phone alignment and video tracking jointly outperform either source alone.
- Chatziagapi and Samaras, [AVFace: Towards Detailed Audio-Visual 4D Face
  Reconstruction, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chatziagapi_AVFace_Towards_Detailed_Audio-Visual_4D_Face_Reconstruction_CVPR_2023_paper.html)
  — temporal audio/video fusion improves robustness when either modality is
  insufficient, including visual occlusion; this is architectural evidence,
  not an installed model.
- Shi et al., [AV-HuBERT, ICLR
  2022](https://openreview.net/pdf?id=Z1Qlm11uOM) — evidence that correlated
  audio and mouth video improve learned speech representations; it is research
  support for fusion, not a drop-in animation backend.
- Chung and Zisserman, [Out of Time: Automated Lip Sync in the
  Wild](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/) — a
  two-stream learned A/V offset measure suitable for a separately calibrated
  synchronization gate.
- Laine et al., [Production-Level Facial Performance Capture Using Deep
  Convolutional Neural Networks](https://diglib.eg.org/items/fe7d0fb0-dc25-40c1-8211-e48b1790505a)
  — production-quality monocular inference was subject-trained from 5–10
  minutes of high-end multiview/artist-enhanced ground truth, unlike the current
  generic tracker path.
- Apple, [ARKit face blendshapes](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapes),
  [gaze point](https://developer.apple.com/documentation/arkit/arfaceanchor/lookatpoint)
  and [tongue-out coefficient](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation/tongueout)
  for the higher-tier sensor capability contract.
- Feng et al., [DECA: Learning an Animatable Detailed 3D Face Model from
  In-the-Wild Images](https://deca.is.tue.mpg.de/) — separates person detail
  from expression-dependent detail; useful research direction for
  subject-calibrated detail, not proof of current video fidelity.
- Google, [MediaPipe Face Mesh V2 model
  card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)
  — monocular landmark depth is relative rather than metric identity evidence.
- OpenCV, [ChArUco camera calibration](https://docs.opencv.org/trunk/da/d13/tutorial_aruco_calibration.html)
  and COLMAP, [camera model guidance](https://colmap.github.io/cameras.html) —
  retain raw observations, per-view errors/uncertainty and simple shared camera
  models instead of trusting requester-declared calibration scores.
- Han et al., [High-Quality Facial Geometry and Appearance Capture at Home,
  CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Han_High-Quality_Facial_Geometry_and_Appearance_Capture_at_Home_CVPR_2024_paper.pdf)
  and Wang et al., [3D Face Reconstruction with the Geometric Guidance of
  Facial Part, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_3D_Face_Reconstruction_with_the_Geometric_Guidance_of_Facial_Part_CVPR_2024_paper.html)
  — primary references for controlled appearance capture and part-aware dense
  geometry; their assets/licenses are not silently adopted here.
- [MediaPipe repository](https://github.com/google-ai-edge/mediapipe),
  [BlazePose GHUM model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf)
  and [hand-tracking model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf).
- [WHAM repository and dependency setup](https://github.com/yohanshin/WHAM)
  for the license-boundary decision.
- [MetaHuman licensing](https://www.metahuman.com/license?lang=en-US) for the
  optional Unreal adapter decision.
- [Meta Multiface repository and data format](https://github.com/facebookresearch/multiface)
  and [CC BY-NC dataset license](https://github.com/facebookresearch/multiface/blob/main/LICENSE).
- [NeRSemble Benchmark package](https://pypi.org/project/nersemble-benchmark/)
  for the approval-gated calibrated fixture.
- [USC Digital Emily 2](https://vgl.ict.usc.edu/Research/DigitalEmily2/) for
  polarization-separated appearance and microgeometry research data.
