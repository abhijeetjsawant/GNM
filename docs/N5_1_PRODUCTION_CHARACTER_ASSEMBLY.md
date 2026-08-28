# N5.1 — Production Character Assembly

Status: detailed-hand milestone verified; production promotion remains blocked  
Primary user: solo animator preparing a reusable character and performer-driven take  
First qualification take: retained 70-frame `research-squat-640` GEM-X/SOMA capture

## Why this phase exists

N5 produced one connected GNM head and MPFB body, but the result is still a
preview. A production character must be reusable as one immutable revision:
identity, body proportions, complete deformation rig, neck attachment,
materials, textures, animation ownership, validation evidence, and publish
state must travel together.

The first visible release blocker is hand fidelity. In the qualification take,
the performer clasps both hands. The output wrists follow the source while the
fingers stay spread in the MPFB rest pose. This is not a shading problem:

- SOMASKEL77 contains articulated fingers;
- the retained take contains non-constant finger rotations, with individual
  joints moving by up to about 26 degrees relative to the first frame;
- MPFB's default rig contains 15 deforming finger bones per hand and the
  corresponding skin weights;
- `project_soma_to_body_track()` projects only wrists into the 25-joint track;
- `blender_body_worker.py` collapses every MPFB finger weight to its wrist.

The current runtime therefore cannot reproduce the reference hand pose even
when source motion and target deformation data are available.

## Product outcome

A promoted character revision can be selected in a Take, receive video body
motion and GNM facial performance, play and scrub deterministically, and export
one connected character without losing high-value articulation or silently
weakening quality gates.

For N5.1 completion, the solo animator must be able to:

1. create or open a character revision;
2. bind a GNM head to a selected body revision;
3. inspect body proportions, neck seam, skeleton, influences, UV/material
   inventory, and readiness blockers;
4. apply the retained video take, including articulated hands;
5. compare source and 3D at exact timestamps and replay repeatedly;
6. promote only when every required assembly gate passes;
7. export an editable master plus a runtime GLB with the same approved revision
   and motion hashes.

## Scope and ordering

### N5.1A — Detailed-hand contract

Preserve the stable 25 core joint IDs and append 15 VRM-compatible deforming
finger joints per hand. Old 25-joint tracks remain valid previews. Detailed
tracks and assets use a new schema/profile and cannot be mislabeled as the old
contract.

Required joints per side:

- metacarpal, proximal, and distal thumb;
- proximal, intermediate, and distal index;
- proximal, intermediate, and distal middle;
- proximal, intermediate, and distal ring;
- proximal, intermediate, and distal little finger.

SOMA's extra non-thumb segment and end markers are folded into the three-bone
target chain using bind-relative global deltas. No target bone scaling is
allowed. Missing or untrusted hand observations remain explicitly
`review_required`; they do not fall back to an unreported canned gesture.

### N5.1B — Production skin deformation

- retain up to eight normalized influences in the editable/native master;
- emit a separately measured four-influence runtime projection when required by
  the renderer;
- preserve original finger groups instead of collapsing them to the wrist;
- qualify shoulders, elbows, wrists, fingers, hips, knees, and the connected
  neck using neutral and animated deformation tests;
- add corrective shapes only after the base skin contract is measured.

### N5.1C — Character identity and attachment

- store body provider inputs, proportions, neutral geometry, skeleton, skin,
  GNM identity revision, attachment calibration, and ownership contract in one
  immutable character revision;
- replace the uncalibrated identity socket with measured head/body bind
  calibration;
- use common resampled collar topology or a hidden production seam;
- blend normals/materials across the attachment and reject collar stretch,
  inversion, or open boundaries.

#### MAMMA four-camera attachment execution — 2026-08-04

The production attachment path was exercised with the official MAMMA A001–D001
four-camera example over source frames `[60, 90)`.  Both fitted people were
imported as 55-joint body tracks, rebased only from their first capture-world
root sample, and composed against an intentionally neutral GNM track on the
same source PTS clock.  This proves the following, and only the following:

- the 30 body frames, source PTS and GNM frames agree exactly;
- MAMMA's capture-world root orientation is corrected before the canonical
  anatomical reflection, so the connected character is upright rather than
  lying on its side;
- GNM remains the owner of face, lips, tongue, teeth and eyes, while MAMMA owns
  root, body, hands and macro head/neck motion;
- the source is a lifting shot with no qualified close dialogue-face evidence,
  so zero facial expression is a deliberate neutral mode, not a lipsync claim.

The first connected collar had 90 body and 234 GNM cut-loop vertices.  Very
short GNM plane-intersection segments (minimum 0.039 mm) forced needle bridge
triangles: p95/max aspect was 152/1367 on the real subject-00 take.  A
deterministic cut-loop topology hygiene pass now collapses only consecutive
cut-loop samples below a declared threshold, preserving the retained endpoint's
source interpolation provenance.  At the conservative 0.75 mm setting, it
reduced the loops to 88/184 and improved aspect to 52/110, but still fails the
required p95/max limits of 5/10.  A 1.5 mm exploratory pass improved aspect to
22/42 but made the animated bridge stretch fail (0.733x).

Therefore this is **not** a tuning knob to promote.  The next N5.1C work is a
common arc-length-resampled collar: split both clipped boundary edges to a
single correspondence count, interpolate positions, source provenance and
eight-influence skin weights on both sides, build quads/paired triangles from
the shared parameters, and qualify that new boundary across held-out neck,
shoulder and head motion.  Promotion remains blocked by identity calibration,
look development, runtime skin projection, collar shape residual, bridge
triangle quality and animated stretch.

### N5.1D — Materials, UVs, and look development

- retain the provider UV set and publish an explicit material-slot inventory;
- bind base color, normal, roughness, specular, displacement, subsurface/radius,
  masks, and confidence maps to exact character and UV revisions;
- keep native 8K/16-bit source maps as authoring assets;
- build bounded 1K/2K/4K runtime LODs instead of calling a browser GLB “8K”;
- require controlled-light relighting and close-up pore/seam review before
  `lookdev_approved=true`.

### N5.1E — Qualification and promotion

Run held-out bodies and takes, preserve evidence, and make all failures visible
in the Character and Publish tabs. No upstream model self-score can promote a
revision.

## Architecture contracts

### Append-only skeleton compatibility

- indices `0..24` retain their existing names, parents, and semantics;
- detailed hand indices are appended, parent-before-child;
- body tracks declare their exact skeleton schema and joint order;
- exporters derive counts from the bound asset/track instead of assuming 25;
- a 25-joint track may animate a detailed asset only as a visibly labeled
  wrist-only preview;
- a detailed track may not be truncated during production export.

### Animation ownership

- body track: root, torso, limbs, wrists, fingers, macro neck/head;
- GNM: face identity, expression, lipsync, tongue, teeth, jaw, eye-local motion;
- artist layer: additive corrections with explicit ownership and revision;
- each channel is written exactly once before additive correction.

### Evidence and failure behavior

- every artifact stores provider, source, skeleton, character, binding, motion,
  material, and texture hashes;
- schema, order, side convention, rest transforms, or hashes that disagree fail
  closed;
- low-confidence/occluded fingers are flagged by frame range;
- unresolved hand intersections, neck failures, missing identity calibration,
  and unapproved lookdev block publish but remain reviewable;
- old preview artifacts remain readable and are never silently rewritten.

## Acceptance gates

### Detailed hands

- 55 total joints: unchanged 25 core plus 30 finger joints;
- exactly 15 deforming finger joints per side with valid parent-before-child
  hierarchy;
- normalized, finite, sign-continuous quaternions within `1e-5`;
- skin weights sum to `1 +/- 1e-4`;
- no finger-region vertex is bound only to the wrist when an original MPFB
  finger influence exists;
- retained qualification take exports non-constant animation for both hands;
- fingertip and knuckle motion agrees with the retargeted SOMA target within
  `<= 2 mm RMS` when evaluated on the target skeleton;
- first, middle, and final source timestamps pass visual hand-pose review;
- replay, second replay, backward scrub, and end restart animate source and 3D
  from the same media clock.

### Skin and attachment

- bind/inverse-bind identity error `<= 1e-6`;
- neutral eight-influence reconstruction RMS `<= 0.1 mm`, max `<= 0.5 mm`;
- measured four-influence runtime projection RMS `<= 0.1 mm`, max `<= 0.5 mm`;
- neck seam gap `<= 0.25 mm`;
- no open connected-neck boundary, inverted bridge triangles, or animated
  collar stretch outside `0.8x..1.25x`;
- held-out shoulder/wrist/finger poses have no visible collapse, spikes, or
  self-intersection at approved review cameras.

### Identity and lookdev

- body identity parameters and GNM identity are immutable revision inputs;
- subject measurements meet the approved calibration tolerances recorded by the
  character revision;
- UV/material inventory round-trips without slot or color-space drift;
- authoring maps retain native resolution and bit depth;
- runtime LOD selection stays within the declared GPU budget;
- controlled-light face/body match, collar seam, hand close-up, and pore detail
  receive artist approval.

## Test strategy

1. **Contract tests:** skeleton append-only identity, hierarchy, schema
   rejection, side mapping, quaternion and hash validation.
2. **Retarget tests:** synthetic per-finger rotations, folded SOMA chains,
   identity/rest frames, sign continuity, old 25-joint compatibility.
3. **Provider tests:** real MPFB rig export, preserved finger bones, retained
   vertex groups, normalized eight-influence skin, deterministic hashes.
4. **Export tests:** body-only and connected GLB joint/channel counts,
   inverse-bind validity, runtime influence projection, animation round-trip.
5. **Real-input tests:** retained clasped-hands video, at least two additional
   consented hand gestures, three body proportions, and the existing connected
   GNM take.
6. **Browser/native review:** exact source/3D timestamps, locked cameras,
   start/middle/end, replay twice, scrub backward, hand close-up, wireframe,
   normals, seam, and material LOD modes.
7. **Full regression:** complete Python suite, Rust physics suite, macOS app
   build/tests, glTF validation, and retained artifact verification.

## Milestones and stop/go rules

| Milestone | Go condition | Stop condition |
|---|---|---|
| M1 detailed skeleton | append-only schema and compatibility tests pass | any core ID/order changes |
| M2 MPFB hand skin | real provider exports 30 finger bones and valid weights | collapsed or missing finger influences |
| M3 SOMA hand retarget | qualification take has finite, non-constant, side-correct finger curves | frozen, mirrored, or basis-misaligned fingers |
| M4 connected export | one GLB replays and scrubs source-synchronously with articulated hands | exporter truncates or viewer desynchronizes |
| M5 deformation | eight/four influence, collar, and held-out pose gates pass | spikes, stretch, inversion, or threshold miss |
| M6 identity/lookdev | calibrated revision and controlled-light approval pass | placeholder identity, unbound maps, or unapproved seam |
| M7 promotion | all evidence hashes verify and every blocker is cleared | any required metric null or failed |

## Dependencies and known limits

- The retained GEM-X run is an Apple-Silicon preview, not a production-approved
  provider. Its finger channels can validate integration but cannot establish
  production capture quality.
- Single-camera hand tracking is vulnerable to the exact case shown here:
  crossed, clasped, mutually occluded hands. Production qualification therefore
  needs dedicated hand observations and confidence, plus artist correction.
- Exact performer body identity and clothing are not solved by the neutral MPFB
  body. They require a measured parametric fit or a separately licensed body
  reconstruction provider.
- Texture generation cannot recover measured pores or reflectance hidden by the
  source. Native high-resolution capture, calibrated lighting, and human
  lookdev approval remain required.
- Cloth, hair, muscle, and soft-tissue simulation remain downstream of this
  character-assembly phase; they cannot compensate for a wrong skeleton, bind,
  or identity.

## Qualification checkpoint — 2026-07-30

The first retained real-input milestone now passes:

- the pinned MPFB worker exported a 55-joint asset with 30 deforming finger
  joints;
- 3,012 vertices retain at least one finger influence and 2,365 of those do
  not depend on either wrist;
- maximum skin-weight sum error is `1.1921e-7` with up to eight influences;
- all 30 detailed target finger joints are non-constant in the retained
  70-frame SOMA take, with a maximum observed rotation span of about
  `26.49 degrees`;
- body-only and connected GLBs contain all detailed joints and animation
  channels and pass Khronos glTF validation with zero messages;
- source/3D start and middle timestamps show clasped, curled hands instead of
  the old splayed rest pose;
- play-to-end, replay from end, and backward scrub remain synchronized from the
  source media clock with no browser warnings or errors.
- MPFB palm roll is calibrated from independent palm-length and palm-width
  axes, while every finger segment receives its own bind-frame correction;
  measured provider-to-canonical finger bind-axis residual is now `0 degrees`
  instead of the previous `88.8-degree` mean / `132.5-degree` maximum;
- retained-take palm-frame error is reduced from about `25 degrees` to
  `6..12 degrees`, matching the measured difference between the canonical
  neutral hand and the GEM-X source rest frame.

This checkpoint does **not** promote the character. Exact inter-finger contact
cannot be recovered reliably from the mutually occluded single-camera take,
and the following measured blockers remain:

- body identity is not calibrated to the performer;
- bridge look development is not artist-approved;
- the SOMA detailed-hand provider is preview-qualified only;
- neck planar residual, four-weight maximum error, bridge triangle quality, and
  animated collar stretch exceed their production gates.

## Rollback

The 25-joint schemas and preview exporters remain readable. The detailed-hand
path is introduced under new schema/profile identifiers and can be disabled
without rewriting old artifacts. A failed detailed export leaves the last
verified preview intact and records a typed blocker rather than truncating the
motion.
