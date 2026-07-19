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
| Texture | Up to 1024 RGBA color atlas with coverage/confidence/provenance; separate CLI structural/attestation gate for complete base/normal/displacement/specular/roughness/SSS/radius/confidence/mask packages and 2K/4K/8K claims | Package structure/lineage/rights gating works, but map recovery, perceptual pore/relighting validation and attachment to a character revision are not implemented; current character reuse renders base color only |
| Audio animation | Rhubarb timing plus native Claire MLX learned motion, 52 skin/16 tongue controls, dense GNM retarget, character contact solve | Working; independent phoneme/perceptual approval and physical oral rig remain open |
| Video animation | Exact source PTS, MediaPipe face/expression/head/translation/gaze, direct inner-lip geometry, identity-calibrated contact and aperture | Follows the video visually, not its audio; microexpression/FACS ground truth remains open |
| Character reuse | Versioned identity/preview/material revision, exact revision transport, consent scope/expiry/revocation, content hashes and HMAC trust root | Working local trust boundary; hosted deployment should move signing to KMS/HSM |
| LLM acting | Codex and Claude terminal adapters, no tools, strict schema and semantic validator, trusted envelope, measured performance windows | Produces an editable proposal; its body/gaze portion is deterministically compiled, but artist approval still follows |
| Full body | 25-joint canonical humanoid, 48 kHz integer-tick sampled body/gaze track, OpenUSD/glTF/VRM mappings, contact constraints and GNM attachment ownership | Foundation works; no body mesh, mocap reconstruction, locomotion, USD/glTF skin writer, or production capture is shipped |
| Tongue | Real learned-audio motion reaches GNM and animated GLB | Computationally verified; camera visibility, phoneme timing and collision are not yet approved |

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

Video motion is derived from **visual frames**, not the audio track:

- MediaPipe blendshapes drive expression;
- inner-lip landmark geometry drives contact and open-mouth aperture;
- transform matrices drive head rotation and translation;
- eye-look controls drive the two GNM eye joints;
- exact decoded PTS is preserved;
- source audio is copied only into the proxy/playback clock.

The earlier retained CREMA-D run had 0.974 lip-opening timing correlation but
an affine amplitude slope of 0.582: roughly 40% under-opening. The new
identity-calibrated aperture layer reaches 0.988 correlation, 0.883 slope and
0.885 p95 source/output amplitude ratio on the same retained capture while
preserving contact. The acceptance gate is correlation `>= 0.90` and open-frame
p95 ratio `0.85–1.15`. This directly addresses the perceived "mouth too
closed" issue without inflating true closed frames.

Video tongue remains zero because MediaPipe does not observe it. A future
audio-visual fusion mode must be separately named and tested on contradictory
media (silent expressive video, expressive audio with neutral face, and dubbed
misaligned video); default video-follow must continue to honor the visible
performance.

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
still no body mesh, motion estimation from the source video, skin export,
locomotion, or performer-approved body performance.

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
- Character-calibrated procedural and learned lip contact.
- Visual video aperture matching and source/output regression metrics.
- Gate: real fallback contact attainment `>=0.75`, mouth step `<=0.040`; video
  aperture correlation `>=0.90`, p95 ratio `0.85–1.15`, exact PTS/contact.

### Phase 3 — safe acting direction (implemented proposal layer)

- Measured performance windows plus transcript/instructions.
- Codex and Claude no-tool adapters; strict schema and semantic validator.
- Signed provider envelope and edit-ready beats; lipsync override prohibited.
- Gate: prompt injection, tool attempt, malformed/oversized/timeout/refusal,
  overlap and forbidden-field tests; one real successful run per provider.

### Phase 4 — editable body foundation (partially implemented)

- Implemented: canonical skeleton, inverse binds, integer-tick body/gaze track,
  GNM neck/head/eye ownership, foot contacts and deterministic tag compiler.
- Remaining: rights-cleared body mesh/capture, USD/UsdSkel writer, glTF/VRM
  skinned runtime adapter and Rigify round-trip.
- Gate: hierarchy/rest validation, quaternion continuity, exact ticks, foot
  drift, head seam transform, round-trip error, unsupported gesture failure.

### Phase 5 — measured production appearance (validator implemented; recovery/attachment open)

- Implemented: fail-closed CLI structural/attestation gating for complete aligned atlas/UDIM map
  inventories, lossless decode, channel/depth/normal semantics, source/native
  resolution, lineage, commercial rights, hashes, and evidence-backed
  native/pore/relightable claims. The output calls these claim gates, never
  perceptual validation; pore-frequency and unseen-light tests remain false.
- Remaining: calibrated polarized recovery, high-frequency residual geometry,
  package signing/attachment to character revisions, 4K/8K bake,
  MaterialX/OpenPBR authoring and renderer adapters.
- Gate: held-out geometry/reprojection, diffuse/specular separation,
  relighting error, coverage, seam delta, color chart, pore-frequency
  validation and artist look-dev approval.

### Phase 6 — oral anatomy and audiovisual fusion (partially measured)

- Visible oral preview, virtual jaw/rigid teeth, tongue attachment and SDF
  collision; phone-aligned corpus.
- Optional named audio-visual fusion policy with conflict diagnostics.
- Gate: rigid-teeth residual `<0.10 mm`, oral penetration p99 `<0.10 mm` / max
  `<0.25 mm`, tongue timing within one annotated frame and no false visibility.

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
- [MetaHuman licensing](https://www.metahuman.com/license?lang=en-US) for the
  optional Unreal adapter decision.
- [Meta Multiface repository and data format](https://github.com/facebookresearch/multiface)
  and [CC BY-NC dataset license](https://github.com/facebookresearch/multiface/blob/main/LICENSE).
- [NeRSemble Benchmark package](https://pypi.org/project/nersemble-benchmark/)
  for the approval-gated calibrated fixture.
- [USC Digital Emily 2](https://vgl.ict.usc.edu/Research/DigitalEmily2/) for
  polarization-separated appearance and microgeometry research data.
