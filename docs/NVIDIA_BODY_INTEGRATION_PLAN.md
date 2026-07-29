# NVIDIA SOMA, GEM-X, and Kimodo body-motion integration plan

Status: implementation specification, 2026-07-29. This document is
repository-grounded and intentionally does not claim that NVIDIA inference,
SOMA retargeting, a calibrated GNM/body attachment, or production body capture
already works in AutoAnim.

## Executive decision

Use the 77-joint `somaskel77` contract shared by GEM-X and Kimodo as a
**source-motion interchange**, not as a replacement for AutoAnim's character
model or GNM face rig. Keep it explicitly distinct from the current public
SOMA-X 78-slot body-model layout (one dummy root plus 77 articulated joints).

- **GEM-X** is the preferred video-to-body provider. It estimates 77-joint
  whole-body motion, hands, a global trajectory, and camera-relative/world
  motion from monocular video.
- **Kimodo-SOMA** is the preferred offline body-acting generator. It converts
  approved text and kinematic constraints into editable SOMA motion.
- **SOMA-X** supplies the body-model vocabulary and an independent conformance
  oracle, but its public arrays include a dummy root plus 77 articulated joints
  and are not byte/schema-identical to GEM-X/Kimodo `somaskel77`. AutoAnim still
  needs explicit versioned adapters and its own source-to-canonical-25
  retargeter.
- The existing pinned hm08/MPFB asset remains AutoAnim's default, commercially
  auditable neutral body. The first integration moves SOMA motion onto that
  body; it does not silently replace the mesh with MHR, SOMA-shape, SMPL, or
  SMPL-X.
- GNM remains the only owner of facial expression coefficients, speech mouth,
  lips, teeth, tongue, and face-local eye rotation. GEM-X face joints are
  retained as observation evidence, not copied into GNM coefficients. Kimodo
  cannot author any GNM facial channel.
- NVIDIA code, models, assets, and containers run behind the same kind of
  fail-closed, out-of-process boundary already used for MPFB. The native macOS
  application remains the production workspace. Full CUDA inference may run
  on a separately attested NVIDIA worker; an Apple-Silicon preview path is not
  assumed to have feature or quality parity.

The first shippable vertical slice is:

```text
source video
    |-- existing visual/audio face pipeline --> GNM face/oral/eyes
    `-- GEM-X worker --> native SOMA-77 motion
                           |
                    validated retarget
                           |
                  canonical body-track v2
                           |
              GNM/body ownership compositor
                           |
          pinned MPFB body + calibrated GNM head
                           |
       reviewable shot component + USD/glTF output
```

Kimodo enters after that path is measured, using an approved acting plan as its
prompt/constraint source and producing the same native SOMA and canonical body
artifacts.

## Why this fits the repository

### Current body foundation

`src/autoanim_gnm/body.py:1-7` says exactly what the current implementation is:
an interchange foundation and bounded deterministic upper-body compiler, not
motion capture. Its public contracts are:

- `autoanim.humanoid-skeleton/1.0`,
  `autoanim.body-track/1.0`, and
  `autoanim.gnm-body-attachment/1.0`
  (`src/autoanim_gnm/body.py:22-34`);
- a parent-before-child, 25-joint, right-handed, +Y-up, +Z-forward skeleton in
  meters, with local `[x,y,z,w]` quaternions and UsdSkel/glTF/VRM mappings
  (`src/autoanim_gnm/body.py:64-98`);
- immutable sampled arrays for root translation, 25 local rotations, two foot
  contacts, gaze, and GNM eye-local rotations
  (`src/autoanim_gnm/body.py:371-450`);
- a fixed 48 kHz project timebase, maximum 120 Hz sampling, normalized
  quaternions, exact canonical joint order, and hard contact-drift validation
  (`src/autoanim_gnm/body.py:650-739`).

The current compiler maps a small vocabulary of stance/gesture labels to
hand-authored poses. It holds the root and lower body still. It is a useful,
deterministic fallback, not a source-video body solve.

### Current body asset boundary

`src/autoanim_gnm/body_provider.py` already contains a strong pattern to copy:

- strict request/response/asset schemas and byte limits
  (`src/autoanim_gnm/body_provider.py:36-103`);
- pinned Blender, MPFB, MakeHuman assets, hashes, source URLs, and licensing;
- an out-of-process GPL boundary;
- exact NPZ member, shape, finite-value, hierarchy, bind matrix, skin-weight,
  seam, and hash validation
  (`src/autoanim_gnm/body_provider.py:730-979`);
- an explicit rule that a provider response cannot mark itself production
  validated (`src/autoanim_gnm/body_provider.py:617-695`).

The provider produces a real neutral 25-joint skinned body, but its GNM socket
is deliberately identity and `attachment_calibrated=false`. SOMA motion must
not bypass that gate.

### Current acting boundary

`src/autoanim_gnm/acting.py:88-190` limits the LLM to declarative beats:
intent, valence/arousal, stance, gesture tags, face tags, gaze, and preservation
constraints. `preserve_lipsync` must be true. The provider prompt explicitly
forbids rig values, paths, URLs, commands, and code, and both terminal adapters
run with tools disabled (`src/autoanim_gnm/acting.py:451-535`).

This remains the correct security and authorship boundary. Kimodo constraints
must be compiled deterministically from an approved acting plan. The LLM must
not emit Kimodo JSON, joint rotations, worker arguments, or files directly.

### Current service and production-store gaps

`AutoAnimService.direct()` currently:

1. accepts a sealed audio-animation or video-performance job;
2. runs the LLM proposal;
3. immediately compiles the proposal with `compile_body_track()`;
4. writes the v1 NPZ/manifest/skeleton/attachment artifacts; and
5. labels the body result `unapproved_preview` and
   `production_validated=false`
   (`src/autoanim_gnm/service.py:1570-1765`).

`ProductionStore.link_job()` accepts only one successful audio or video job per
new shot version, checks its exact source-media and character-revision hashes,
then seals the single job manifest
(`src/autoanim_gnm/production.py:503-600`). It cannot yet attach:

- a body-motion provider result;
- an acting-direction child job;
- an artist-approved body correction;
- a GNM/body composition report; or
- multiple component hashes as one immutable shot version.

The macOS project library can create/list projects, characters, shots and link
completed audio/video performances, but it has no body source, provider status,
retarget report, owned-layer approval, or publish gate. See
`native/autoanim-macos/Sources/AutoAnimMacCore/ProductionLibraryModels.swift`
and
`native/autoanim-macos/Sources/AutoAnimMac/ProductionLibraryView.swift`.

### Current tests to preserve

The implementation must keep the existing fail-closed behavior covered by:

- `tests/test_body.py`: canonical hierarchy, ownership, deterministic compile,
  exact ticks, quaternion normalization, foot anchoring, and round-trip;
- `tests/test_body_provider.py` and
  `tests/test_body_provider_bootstrap.py`: pinning, safe JSON/NPZ, hashes,
  provenance, body asset semantics, dependency attestation, and typed blocks;
- `tests/test_acting.py`: tool-free direction, prompt injection rejection,
  lipsync preservation, video/audio authority, and duration limits;
- `tests/test_production.py` and `tests/test_production_api.py`: immutable
  character pins, exact source matching, sealed job linking, tamper rejection,
  API transport, and restart persistence.

No NVIDIA phase may weaken these tests or relabel a preview as production
validated.

## What the NVIDIA stack contributes

### GEM-X

The official [GEM-X repository](https://github.com/NVlabs/GEM-X) describes a
monocular-video system that recovers a 77-joint SOMA pose, including body,
hands, and face, handles dynamic cameras, and recovers global trajectories. It
ships its own 77-keypoint detector. The code is Apache-2.0; the associated
model is governed separately by the NVIDIA Open Model License. The main
installation path uses PyTorch/CUDA 12.6. NVIDIA separately documents an
Apple-Silicon real-time webcam demo.

Implications:

- This is the released provider closest to AutoAnim's missing
  video-to-full-body lane.
- The global/camera outputs directly address the current "moving camera versus
  performer root" gap, but they still require validation against known camera
  motion and 3D truth.
- The public macOS demo is a valuable preview/runtime experiment, not evidence
  that the full offline path, checkpoints, numerical outputs, or quality are
  equivalent to CUDA.
- GEM-X's face joints do not solve GNM retargeting. They remain evidence until
  an independently validated mapping exists.
- The repository calls its source code commercially usable, but AutoAnim must
  still retain and approve the exact model license and all transitive
  attributions before distribution or hosted inference.

### SOMA-X

The official [SOMA-X repository](https://github.com/NVlabs/SOMA-X) defines a
canonical topology and rig intended to pivot among MHR, Anny, SMPL/SMPL-X,
SOMA-shape, and GarmentMeasurements. Its code is Apache-2.0 and its evaluation
uses NVIDIA Warp. It supports pose correctives and conversion tools for some
source models. The current public SOMA-X body-model layout has 78 slots when its
dummy root is counted; its model card calls this 77 articulated joints and
explicitly excludes the dummy root from the pose tensor. GEM-X/Kimodo motion
interchange uses the separate 77-joint `somaskel77` contract. They must have
different schema identifiers, hashes, and adapters.

Implications:

- `somaskel77` is the right source skeleton shared by GEM-X and Kimodo.
- The public SOMA-X dummy-root-plus-77 layout is a related body-model/runtime
  contract, not an interchangeable spelling of `somaskel77`. AutoAnim must not
  drop or insert the dummy root implicitly.
- It is not a general "arbitrary rig to AutoAnim 25 joints" product. AutoAnim
  must own and test the SOMA-to-canonical-25 transfer.
- Identity backends have separate terms. SMPL/SMPL-X model files explicitly
  require separate licenses and cannot be redistributed under the SOMA-X
  repository license.
- The upstream README calls SOMA-shape a "proprietary PCA-based" backend, while
  the current official model card says the SOMA release is ready for commercial
  use under Apache-2.0. Treat that wording as architecture/data provenance, not
  a substitute for asset-level terms: retain and approve the exact release
  license and every backend asset before use.
- Phase 1 should separately pin `somaskel77` names/parents/basis conventions and
  the public SOMA-X dummy-root-plus-77 contract. A moving upstream alias is not
  a schema.

### Kimodo

The official [Kimodo repository](https://github.com/nv-tlabs/kimodo) provides
text- and constraint-conditioned motion diffusion for SOMA, Unitree G1, and
SMPL-X. It supports full-body keyframes, joint position/rotation constraints,
2D root paths/waypoints, end effectors, multiple prompts, deterministic seeds,
and foot-skate/constraint postprocessing. Since its March 2026 breaking change,
its external SOMA input/output uses `somaskel77`.

Its default NPZ contains global joint positions, global and local rotation
matrices, four foot contacts, smoothed and raw root positions, and global root
heading. The code is Apache-2.0; checkpoints and datasets use separate terms.
The SOMA RP/SEED weights are listed under the NVIDIA Open Model License, while
the SMPL-X checkpoint is an NVIDIA R&D model. The RP model was trained on
roughly 700 hours of proprietary Bones Rigplay motion; the lower-capability
SEED variant uses 288 hours of public BONES-SEED motion. Local all-GPU
generation is documented at roughly 17 GB VRAM; moving the text encoder to CPU
reduces it below 3 GB with a speed cost.

Implications:

- Kimodo is an offline candidate generator and repair tool, not a live facial
  acting model.
- Use the SOMA RP v1.1 or a later explicitly approved SOMA Open Model
  checkpoint. Never route commercial AutoAnim work through the released
  SMPL-X R&D checkpoint.
- LLM acting should provide semantic intent; a deterministic compiler should
  translate approved beats into Kimodo prompts and constraints.
- Generate multiple candidates, score constraints/contact/continuity
  automatically, then require artist selection. A diffusion sample is never
  silently accepted as performer truth.

## Target architecture

### Process boundaries

```text
native macOS workspace / Python API
       |
       | sealed request + input hashes + nonce
       v
NVIDIA provider adapter
       |-- local Apple-Silicon GEM preview worker (optional)
       |-- local/remote CUDA GEM-X worker
       `-- local/remote CUDA Kimodo worker
       |
       | strict response + native SOMA archive + attestation
       v
application-side validators
       |
       |-- source timing/camera validation
       |-- SOMA schema validation
       |-- SOMA -> canonical-25 retarget
       |-- contact/continuity/quality report
       `-- license/provenance gate
       |
       v
shot component composer
       |-- existing GNM facial performance
       |-- canonical body-track v2
       |-- approved acting plan
       `-- sparse artist overrides
       |
       v
USD/UsdSkel master + glTF/VRM review + sealed shot version
```

Provider code is not imported into `autoanim_gnm`. Worker environments are
version-pinned and network-disabled during inference. Model acquisition is a
separate explicit installation operation that records terms and hashes.

### Channel ownership

| Channel | Video-follow source | Audio/LLM acting source | Final owner | Rule |
|---|---|---|---|---|
| Root translation and heading | GEM-X | Kimodo or current deterministic fallback | Body | One base source only; artist correction is a sparse override |
| Pelvis, spine, chest, shoulders, limbs, hands, feet | GEM-X | Kimodo or fallback | Body | Canonical body-track v2 |
| Fingers | GEM-X native SOMA retained | Kimodo native SOMA retained | Deferred body extension | Not discarded from native archive; not emitted by canonical 25-joint output |
| Base neck/head pose | GEM-X | Kimodo/fallback | Body | Written once as canonical base |
| Neck/head micro-motion | Existing GNM/video evidence | Existing acting layer | GNM additive | Applied after body base |
| Eye rotation, blink, eyelids | Existing video/GNM lane | Existing GNM gaze/acting lane | GNM | SOMA eyes are evidence only |
| Facial affect | Existing video/GNM lane | Approved GNM acting layer | GNM | GEM-X/Kimodo never write coefficients |
| Visemes, jaw/aperture, lip contact | Existing audio/video face lane | Existing speech pipeline | GNM lipsync | Byte-identical with body layer muted or enabled |
| Teeth, tongue, oral collision | Existing learned audio/video repair | Existing oral pipeline | GNM | Body providers have no authority |
| Artist correction | Sparse versioned curves | Sparse versioned curves | Artist additive | Last, with per-channel ownership validation |

The neck/head composition remains:

```text
Qfinal = normalize(Qbody_base * Qgnm_additive * Qartist_additive)
```

GNM base head rotation must not be copied into both `Qbody_base` and
`Qgnm_additive`. GEM-X facial joints, jaw, and eyes cannot be smuggled through
the body layer.

### Provider selection

| Requested result | Primary | Allowed fallback | Forbidden silent fallback |
|---|---|---|---|
| Follow a source video's body | GEM-X CUDA | GEM-X Apple preview for preview-labelled jobs; current deterministic acting compile | Treating MediaPipe face tracking or audio prosody as full-body capture |
| Generate body acting from approved direction | Kimodo-SOMA RP | Kimodo-SOMA SEED if explicitly selected; current deterministic compiler | Kimodo-SMPLX R&D model, GEM-SMPL/GENMO noncommercial research weights |
| Preserve source-video facial acting | Existing AutoAnim video/GNM pipeline | Existing conservative A/V repair | GEM-X face joints directly driving GNM |
| Neutral body identity | Existing pinned MPFB/hm08 | Artist-supplied rights-cleared canonical rig | Implicit SOMA/MHR/SMPL-X body substitution |

## Exact interchange contracts

All JSON readers reject duplicate members, unknown members, non-finite
numbers, unsafe artifact names, non-UTF-8 input, and files over declared limits.
All NPZ readers use `allow_pickle=False`, accept only `.npy` entries, reject
symlinks/path traversal/duplicate members/object dtypes, and verify compressed
and expanded size limits before allocation.

### NVIDIA body-motion request

Schema: `autoanim.nvidia-body-motion-request/1.0`.

The root object has exactly these keys:

```json
{
  "schema_version": "autoanim.nvidia-body-motion-request/1.0",
  "request_id": "lowercase-safe-id",
  "operation": "video_capture",
  "provider": {
    "id": "nvidia_gem_x",
    "execution_mode": "remote_cuda",
    "code": {
      "repository_url": "https://github.com/NVlabs/GEM-X",
      "git_commit_oid": "40 lowercase hex characters",
      "tree_sha256": "64 lowercase hex characters",
      "license_spdx": "Apache-2.0",
      "notice_sha256": "64 lowercase hex characters"
    },
    "model": {
      "model_id": "nvidia/GEM-X:gem_soma",
      "revision": "immutable model revision",
      "artifact_sha256": "64 lowercase hex characters",
      "license_id": "NVIDIA-Open-Model",
      "license_text_sha256": "64 lowercase hex characters"
    },
    "runtime": {
      "image_digest": "sha256:64-lowercase-hex",
      "python_version": "3.12.x",
      "torch_version": "pinned",
      "cuda_version": "12.6",
      "soma_schema_sha256": "64 lowercase hex characters"
    }
  },
  "input": {
    "kind": "video",
    "artifact": "input.mp4",
    "sha256": "64 lowercase hex characters",
    "bytes": 123,
    "media_type": "video/mp4",
    "source_time_base": {"numerator": 1, "denominator": 90000},
    "source_start_pts": 0,
    "duration_ticks": 240000,
    "ticks_per_second": 48000
  },
  "character": {
    "character_id": "26-character ULID",
    "revision_id": "26-character ULID",
    "revision_manifest_sha256": "64 lowercase hex characters",
    "body_asset_manifest_sha256": "64 lowercase hex characters",
    "gnm_attachment_sha256": "64 lowercase hex characters"
  },
  "direction": null,
  "options": {
    "sample_rate_hz": 30,
    "world_motion": true,
    "retain_camera_motion": true,
    "candidate_count": 1,
    "seed": 0,
    "text_encoder_device": null
  },
  "output": {
    "native_motion_json": "soma-motion.json",
    "native_motion_npz": "soma-motion.npz",
    "provider_raw_manifest_json": "provider-raw-motion.json",
    "provider_raw_motion_npz": "provider-raw-motion.npz",
    "canonical_track_json": "body-track-v2.json",
    "canonical_track_npz": "body-track-v2.npz",
    "validation_report_json": "body-validation.json"
  },
  "nonce": "at least 128 bits encoded as 32 lowercase hex characters"
}
```

Normative alternatives:

- `operation="video_capture"` requires
  `provider.id="nvidia_gem_x"`, `input.kind="video"`,
  `direction=null`, `candidate_count=1`, and a source timebase.
- `operation="motion_generate"` requires
  `provider.id="nvidia_kimodo_soma"`,
  `input.kind="acting_plan"`, `direction` as defined below, and
  `candidate_count` in `[1,8]`.
- `operation="retarget_only"` requires
  `provider.id="autoanim_soma_retarget"`,
  `input.kind="soma_motion"`, and `direction=null`.
- `provider.model` is the shown exact object for GEM-X and Kimodo. It is null
  for `autoanim_soma_retarget`, whose application code revision is instead
  bound in `provider.code`.
- `sample_rate_hz` is one of `24, 25, 30, 48, 50, 60, 100, 120` and must divide
  48,000.
- Maximum source duration remains 30 minutes until load/streaming gates raise
  it. Video bytes are bounded by the existing upload policy; native motion NPZ
  is limited to 512 MiB compressed and 2 GiB expanded.
- `execution_mode` is `local_apple_preview`, `local_cuda`, or `remote_cuda`.
  Kimodo disallows `local_apple_preview` in v1.
- A provider/version/model/runtime value is selected from an application-owned
  allowlist. Callers cannot submit arbitrary repositories, images, commands,
  models, or URLs.

For Kimodo, `direction` has exactly:

```json
{
  "acting_plan_sha256": "64 lowercase hex characters",
  "approval_record_sha256": "64 lowercase hex characters",
  "compiled_constraints_artifact": "kimodo-constraints.json",
  "compiled_constraints_sha256": "64 lowercase hex characters",
  "prompt_segments": [
    {
      "start_tick": 0,
      "end_tick": 48000,
      "text": "performer-facing motion description, maximum 500 characters"
    }
  ]
}
```

Prompt text and constraints are application-compiled from an approved
`autoanim.acting-plan/1.0`. They are data, never shell fragments. Segment ticks
are ordered, non-overlapping, within the source duration, and use the existing
48 kHz timebase. The approval record binds the exact plan hash, editor ID,
approval timestamp, intended use, and character revision.

### NVIDIA body-motion response

Schema: `autoanim.nvidia-body-motion-response/1.0`.

The root object has exactly:

```json
{
  "schema_version": "autoanim.nvidia-body-motion-response/1.0",
  "request_id": "same request id",
  "status": "succeeded",
  "production_validated": false,
  "artifacts": {
    "native_motion_json": {"name": "soma-motion.json", "sha256": "...", "bytes": 1},
    "native_motion_npz": {"name": "soma-motion.npz", "sha256": "...", "bytes": 1},
    "provider_raw_manifest_json": {"name": "provider-raw-motion.json", "sha256": "...", "bytes": 1},
    "provider_raw_motion_npz": {"name": "provider-raw-motion.npz", "sha256": "...", "bytes": 1},
    "canonical_track_json": {"name": "body-track-v2.json", "sha256": "...", "bytes": 1},
    "canonical_track_npz": {"name": "body-track-v2.npz", "sha256": "...", "bytes": 1},
    "validation_report_json": {"name": "body-validation.json", "sha256": "...", "bytes": 1}
  },
  "issues": [],
  "attestation": {
    "request_sha256": "64 lowercase hex characters",
    "worker_key_id": "allowlisted key id",
    "worker_build_sha256": "64 lowercase hex characters",
    "started_at": "RFC3339 UTC",
    "finished_at": "RFC3339 UTC",
    "nonce": "request nonce",
    "signature_algorithm": "Ed25519",
    "signature": "base64 signature over RFC 8785 canonical response without signature"
  }
}
```

For `status="blocked"` or `"failed"`, `artifacts` is null and `issues` is a
non-empty array of exact objects:

```json
{
  "code": "MODEL_HASH_MISMATCH",
  "component": "gem_soma checkpoint",
  "expected": "allowlisted digest",
  "observed": "observed digest or null",
  "message": "human-readable bounded message",
  "retryable": false
}
```

The application verifies every artifact hash, the original request hash,
nonce, worker key, signature, worker build, clock skew, and freshness. A
successful worker response still has `production_validated=false`; only the
application's independent quality and approval gates can change shot
eligibility.

For a local-only build without remote execution, attestation uses an
application HMAC key and `signature_algorithm="HMAC-SHA256"`. It must never be
described as remote-host authentication.

### Native SOMA motion manifest and archive

Schema: `autoanim.soma-motion/1.0`.

The JSON manifest has exactly:

```json
{
  "schema_version": "autoanim.soma-motion/1.0",
  "provider_id": "nvidia_gem_x",
  "operation": "video_capture",
  "coordinate_system": {
    "handedness": "right",
    "up_axis": "+Y",
    "forward_axis": "+Z",
    "linear_unit": "meter",
    "rotation": "pose-delta quaternion [x,y,z,w] in source joint basis"
  },
  "source_coordinate_system": {
    "handedness": "right",
    "up_axis": "+Y",
    "forward_axis": "+Z",
    "linear_unit_in_meters": 1.0,
    "source_to_canonical_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    "canonicalization_applied": true
  },
  "skeleton": {
    "name": "somaskel77",
    "semantic_sha256": "hash of exact names, parents, and coordinate semantics",
    "joint_count": 77,
    "joint_names_sha256": "hash of the exact ordered UTF-8 names",
    "parents_sha256": "hash of exact little-endian int16 parents"
  },
  "timebase": {
    "ticks_per_second": 48000,
    "duration_ticks": 240000,
    "sample_rate_hz": 30,
    "sample_rate_semantics": "nominal; ticks are exact source PTS projection",
    "source_time_base": {"numerator": 1, "denominator": 90000},
    "source_start_pts": 0,
    "tick_rounding": "round-half-away-from-zero"
  },
  "semantics": {
    "motion_kind": "observed",
    "root_translation": "world-space SOMA Hips origin in meters",
    "local_rotations": "pose-delta quaternion [x,y,z,w] in the pinned source joint basis",
    "face_joints": "retained evidence; not GNM controls",
    "confidence": "provider evidence confidence; zero is unknown, not invalid"
  },
  "contact_schema": {
    "id": "gem_x_somaskel77_contacts/1.0",
    "contact_names": ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase", "LeftHand", "RightHand"],
    "joint_indices": [69, 70, 74, 75, 14, 42],
    "semantics": ["left_foot", "left_toe_base", "right_foot", "right_toe_base", "left_wrist", "right_wrist"]
  },
  "arrays": {
    "artifact": "soma-motion.npz",
    "sha256": "64 lowercase hex characters",
    "bytes": 1,
    "frame_count": 151,
    "members": ["ticks", "source_pts", "local_rotations_xyzw", "root_translation_m", "rest_joint_positions_m", "rest_world_rotations_xyzw", "joint_positions_m", "identity_coefficients", "scale_parameters", "joint_valid", "evidence_confidence", "contacts", "camera_to_world_matrices", "camera_valid"]
  },
  "source": {
    "request_sha256": "64 lowercase hex characters",
    "input_sha256": "64 lowercase hex characters",
    "provider_response_sha256": "64 lowercase hex characters",
    "provider_raw_motion_sha256": "64 lowercase hex characters",
    "candidate_index": 0,
    "seed": 0
  },
  "license_provenance_sha256": "64 lowercase hex characters",
  "limitations": [
    "monocular 3D inference is ambiguous and requires review",
    "SOMA face joints do not drive GNM"
  ]
}
```

The NPZ has exactly these arrays:

| Member | Dtype and shape | Rule |
|---|---|---|
| `ticks` | `int64 [F]` | Strictly increases from zero and ends at `duration_ticks`; every tick equals the corresponding source PTS relative to the first PTS under the declared exact rational and round-half-away rule |
| `source_pts` | `int64 [F]` | Exact decoded presentation timestamps for GEM-X; Kimodo uses frame ordinal and declares timebase `1/sample_rate_hz` |
| `local_rotations_xyzw` | `float32 [F,77,4]` | Finite normalized, sign-continuous pose-delta quaternions in the pinned source joint bases and exact joint order |
| `root_translation_m` | `float32 [F,3]` | Finite meters; first-frame origin policy recorded in validation |
| `rest_joint_positions_m` | `float32 [77,3]` | Identity/scale-specific source rest joints used for retarget calibration |
| `rest_world_rotations_xyzw` | `float32 [77,4]` | Normalized identity/scale-specific source bind-frame orientations; required to remove local-basis differences during retarget |
| `joint_positions_m` | `float32 [F,77,3]` | Finite FK/world positions matching rotations, rest skeleton, scales, and root within 1 mm RMS |
| `identity_coefficients` | `float32 [I]` | Provider identity vector, possibly length zero; exact semantic name/count declared by the pinned model |
| `scale_parameters` | `float32 [S]` | Provider bone/body scale vector, possibly length zero; exact semantic name/count declared by the pinned model |
| `joint_valid` | `bool [F,77]` | Whether the provider emitted a usable joint |
| `evidence_confidence` | `float32 [F,77]` | `[0,1]`; Kimodo-generated motion sets all values to zero because it is not observed evidence |
| `contacts` | `bool [F,C]` | `C` and column meaning exactly match the versioned `contact_schema`; no column is assumed to be a foot by position alone |
| `camera_to_world_matrices` | `float32 [F,4,4]` | Rigid affine transforms for GEM-X; identity for Kimodo |
| `camera_valid` | `bool [F]` | False for Kimodo; true only where GEM-X camera evidence passes validation |

The application freezes the exact 77 names, parents, and coordinate semantics
from the pinned GEM-X/Kimodo revision in
`src/autoanim_gnm/data/somaskel77-v1.json`. Identity-specific rest positions
and bind-world rotations travel in each motion artifact and are independently
FK-validated. It also freezes the distinct public SOMA-X
dummy-root-plus-77 contract under its own schema file if that runtime is
installed.
It never trusts the worker's spelling/order alone. A future upstream skeleton
change requires a new AutoAnim schema and migration, not an in-place digest
update.

For Kimodo, `motion_kind="generated"`, camera validity is false, and
`evidence_confidence` is zero. Model quality scores belong in the validation
report, not in observation-confidence channels. Its contact schema is
`kimodo_somaskel77_heel_toe/1.0`, with exactly
`["left_heel","left_toe","right_heel","right_toe"]`, four columns, and
the same four exact values as its semantic labels.

GEM-X's pinned contact configuration uses joint indices
`[69,70,74,75,14,42]`: four foot-related contacts and two wrist contacts. The
manifest's `contact_names` must be the exact frozen `somaskel77` names at those
indices, and its semantics must distinguish the first four foot-related
columns from the two wrists. Wrist contacts remain native evidence and are
never interpreted as planted feet.

Provider-native rotations are retained separately and unchanged in
`provider-raw-motion.npz`: GEM-X may expose axis-angle parameters, while Kimodo
exposes local/global rotation matrices. The provider-raw manifest records the
exact upstream format/version/member names and hashes. The worker converts
these into the common normalized `local_rotations_xyzw [F,77,4]` archive, and
the application independently checks quaternion normalization and FK
equivalence against the retained raw artifact.

The provider-raw manifest schema is
`autoanim.provider-raw-body-motion/1.0` and has exactly:

```json
{
  "schema_version": "autoanim.provider-raw-body-motion/1.0",
  "provider_id": "nvidia_gem_x",
  "format_id": "gem_x_soma_axis_angle/1.0",
  "artifact": {
    "name": "provider-raw-motion.npz",
    "sha256": "64 lowercase hex characters",
    "bytes": 1
  },
  "members": [
    {
      "name": "upstream member name",
      "dtype": "float32",
      "shape": [151, 77, 3],
      "semantic": "provider-documented axis-angle pose radians"
    }
  ],
  "request_sha256": "64 lowercase hex characters",
  "input_sha256": "64 lowercase hex characters",
  "canonical_motion_sha256": "64 lowercase hex characters",
  "license_provenance_sha256": "64 lowercase hex characters"
}
```

`format_id` is allowlisted. Initial values are
`gem_x_soma_axis_angle/1.0` and `kimodo_default_npz/1.0`. Kimodo's default four
contacts and any upstream six-column SOMA expansion are different format/
contact schema versions; a validator never guesses between them from shape.
The raw NPZ remains subject to the same safe archive, dtype, size, and
finite-value checks as the common archive.

### SOMA-to-canonical-25 retarget profile

Schema: `autoanim.soma25-retarget-profile/1.0`.

The profile is application-owned, immutable, and bound to one exact SOMA
skeleton digest and one exact AutoAnim skeleton digest. It includes:

- source/target coordinate basis matrices;
- source/target global rest matrices;
- the following semantic transfer;
- per-joint rest alignment quaternions;
- height/leg-length scaling policy;
- pelvis heading decomposition policy;
- contact joints and solver tolerances; and
- a profile hash included in every canonical track.

Core mapping:

| AutoAnim target | SOMA source | Transfer |
|---|---|---|
| `Root` | world root + `Hips` | Root translation; +Y twist/heading from Hips |
| `Hips` | `Hips` | Hips swing after removing root heading |
| `Spine` | `Spine1` | Rest-aligned global delta |
| `Chest` | `Spine2` | Rest-aligned global delta |
| `UpperChest` | `Chest` | Rest-aligned global delta |
| `Neck` | `Neck1` | Rest-aligned global delta |
| `Head` | `Neck2` then `Head` | Compose the two local deltas into one target head delta |
| `LeftEye`, `RightEye` | none | Identity; GNM owns eyes |
| `LeftShoulder` / `RightShoulder` | matching `Shoulder` | Rest-aligned global delta |
| `LeftUpperArm` / `RightUpperArm` | matching `Arm` | Rest-aligned global delta |
| `LeftLowerArm` / `RightLowerArm` | matching `ForeArm` | Rest-aligned global delta |
| `LeftHand` / `RightHand` | matching `Hand` | Wrist delta only; fingers retained only in native SOMA |
| `LeftUpperLeg` / `RightUpperLeg` | matching `Leg` | Rest-aligned global delta |
| `LeftLowerLeg` / `RightLowerLeg` | matching `Shin` | Rest-aligned global delta |
| `LeftFoot` / `RightFoot` | matching `Foot` | Rest-aligned global delta |
| `LeftToes` / `RightToes` | matching `ToeBase` | Toe-base delta; toe-end retained only in native SOMA |

For each one-to-one mapped joint `s -> t`, compute source animated global
rotation `Gs`, source rest global `Bs`, target rest global `Bt`, and source-to-
target basis alignment `A`. The target desired global rotation is:

```text
Delta = A * Gs * inverse(Bs) * inverse(A)
Gt = Delta * Bt
Lt = inverse(Gt_parent) * Gt
```

Matrices are converted to normalized sign-continuous `[x,y,z,w]`
quaternions. Hips heading uses quaternion swing-twist decomposition around the
target +Y axis. The +Y twist goes to `Root`; the residual swing goes to
`Hips`. `Neck2` and `Head` local deltas are composed before target-rest
alignment so the missing target neck segment does not discard head motion.

Root scale is one scalar:

```text
scale = target_neutral_hip_to_floor / source_neutral_hip_to_floor
```

It is clamped to `[0.75,1.35]`; values outside the range fail and require an
artist-created profile. No per-frame bone scaling is allowed. Source bone
lengths may not vary by more than 0.5%.

Time resampling uses source PTS, not frame index, for GEM-X. Translation is
cubic Hermite with bounded tangents; rotations use shortest-arc quaternion
SLERP followed by a sign-continuity pass. No smoothing is allowed before a raw
retarget is stored. Optional cleanup writes a separate derived track and report.

Mapped joint observation/confidence is copied from its source joint; a folded
chain uses the minimum valid source confidence. Canonical eyes are unobserved
with zero confidence. Contact conversion dispatches on the exact schema ID,
never column count alone. For GEM-X it ignores wrist columns and marks a side
planted only when that side's two frozen foot-related columns are true. For
Kimodo it requires both heel and toe for the side. Heel-only, toe-only, and
wrist contact remain in the native archive and do not activate the stronger
canonical whole-foot anchor.

### Canonical body-track v2

Schema: `autoanim.body-track/2.0`.

Version 2 separates body motion from gaze/eyes and generalizes the provenance
field that v1 calls `source_plan_sha256`. The JSON manifest has exactly:

```json
{
  "schema_version": "autoanim.body-track/2.0",
  "skeleton_schema_version": "autoanim.humanoid-skeleton/1.0",
  "attachment_schema_version": "autoanim.gnm-body-attachment/1.0",
  "approval_status": "unapproved_preview",
  "timebase": {
    "ticks_per_second": 48000,
    "duration_ticks": 240000,
    "sample_rate_hz": 30
  },
  "joint_names": ["the exact existing 25-joint order"],
  "source": {
    "kind": "performance_capture",
    "provider_id": "nvidia_gem_x",
    "source_document_sha256": "64 lowercase hex characters",
    "native_motion_sha256": "64 lowercase hex characters",
    "retarget_profile_sha256": "64 lowercase hex characters",
    "candidate_index": 0
  },
  "ownership": {
    "channels": ["root", "body_base"],
    "face": "GNM",
    "lipsync": "GNM",
    "eyes": "GNM",
    "tongue": "GNM",
    "neck_head": "canonical body base; GNM additive only"
  },
  "arrays": {
    "artifact": "body-track-v2.npz",
    "sha256": "64 lowercase hex characters",
    "bytes": 1,
    "frame_count": 151,
    "members": ["ticks", "root_translation_m", "local_rotations_xyzw", "joint_valid", "joint_observed", "joint_confidence", "foot_contacts", "foot_contact_confidence"]
  },
  "quality": {
    "report": "body-validation.json",
    "report_sha256": "64 lowercase hex characters",
    "automatic_gate_passed": false,
    "artist_approved": false,
    "production_validated": false
  },
  "limitations": ["bounded human-readable limitations"]
}
```

The NPZ has exactly:

| Member | Dtype and shape | Rule |
|---|---|---|
| `ticks` | `int64 [F]` | Existing 48 kHz invariants |
| `root_translation_m` | `float32 [F,3]` | Target-scene meters |
| `local_rotations_xyzw` | `float32 [F,25,4]` | Normalized, sign-continuous local quaternions |
| `joint_valid` | `bool [F,25]` | Valid transferred joint |
| `joint_observed` | `bool [F,25]` | True only for source-video evidence; false for generated/fallback curves |
| `joint_confidence` | `float32 [F,25]` | `[0,1]`, zero where unobserved |
| `foot_contacts` | `bool [F,2]` | Left/right combined planted contact |
| `foot_contact_confidence` | `float32 [F,2]` | `[0,1]`, zero for unknown |

`source.kind` is one of `performance_capture`, `generative_motion`,
`deterministic_acting`, `manual`, or `legacy_v1`.

A v1 loader remains supported. Promotion from v1 to v2:

- copies ticks/root/rotations/foot contacts;
- sets all joints valid but unobserved;
- sets joint/contact confidence to zero;
- maps `source_plan_sha256` to `source.source_document_sha256`;
- sets `source.kind="deterministic_acting"` and
  `provider_id="autoanim_acting_compiler_v1"`; and
- writes the old gaze and GNM-eye arrays to a separate
  `autoanim.gaze-track/1.0` artifact without changing their numbers.

There is no v2-to-v1 conversion for captured locomotion in production. A
preview-only downgrade may discard confidence and source metadata, but it must
be explicitly labeled lossy.

### Body validation report

Schema: `autoanim.body-motion-validation/1.0`. Exact root keys:

```json
{
  "schema_version": "autoanim.body-motion-validation/1.0",
  "request_sha256": "...",
  "native_motion_sha256": "...",
  "canonical_track_sha256": "...",
  "character_revision_manifest_sha256": "...",
  "retarget_profile_sha256": "...",
  "checks": {
    "schema": {"passed": true, "issues": []},
    "timing": {"passed": true, "max_tick_error": 0},
    "kinematics": {"passed": false, "bone_length_variation_max": 0.0, "quaternion_norm_error_max": 0.0},
    "contacts": {"passed": false, "planted_foot_drift_p95_m": 0.0, "ground_penetration_p99_m": 0.0},
    "capture": {"passed": false, "reprojection_p95_body_fraction": null, "mpjpe_m": null, "camera_root_validated": false},
    "ownership": {"passed": true, "forbidden_gnm_channels_present": false},
    "license": {"passed": true, "unapproved_dependencies": []}
  },
  "automatic_gate_passed": false,
  "artist_review": {
    "status": "not_reviewed",
    "reviewer_id": null,
    "reviewed_at": null,
    "approval_record_sha256": null
  },
  "production_validated": false,
  "issues": []
}
```

Missing ground truth is `null` and causes the relevant production gate to
remain false. It is never replaced with a provider self-score.

### License and provenance record

Schema: `autoanim.external-model-provenance/1.0`. Every installed provider
version records:

- provider ID and purpose;
- repository URL, exact commit, recursive submodule commits, clean tree digest,
  code SPDX license, license file hash, attribution file hash;
- model host/repository, immutable revision, every artifact name/size/SHA-256,
  model license identifier and full license text hash;
- all auto-downloaded SOMA assets and their individual licenses/hashes;
- training-data statement as published by the provider, explicitly labeled a
  provider claim rather than independently audited fact;
- container/base-image digest and locked dependency manifest hash;
- install timestamp, installer version, accepting operator, approved usage
  scopes, redistribution/hosting decision, expiry/re-review date;
- security scan result and explicit disposition for pickle/TorchScript assets;
  and
- a notice bundle copied into `assets/notices/`.

The install is denied until all code, model, asset, data, and transitive terms
have a recorded disposition. Model files are not committed to Git or bundled
into the macOS application unless the recorded terms explicitly allow it.

Minimum policy:

| Component | Code | Model/assets | AutoAnim decision |
|---|---|---|---|
| GEM-X | Apache-2.0 | NVIDIA Open Model plus transitive assets | Candidate for video capture after legal/security review and exact pinning |
| SOMA-X | Apache-2.0 release/model-card terms; verify exact asset files | Backend-dependent; upstream uses "proprietary PCA-based" to describe SOMA-shape, while the model card calls the release commercially ready; SMPL/SMPL-X stay separate | Pin the dummy-root-plus-77 runtime and every asset/license; do not infer one backend's terms from another |
| Kimodo-SOMA RP/SEED | Apache-2.0 | NVIDIA Open Model; training datasets differ | Candidate for offline generation after exact checkpoint approval |
| Kimodo-SMPLX | Apache-2.0 | NVIDIA R&D Model plus SMPL-X | Forbidden in production |
| GEM-SMPL/GENMO | Research path with separately restrictive weights | Noncommercial/research risk | Not integrated |

## Production-workspace integration

### Componentized shot version

Migrate the durable store to
`autoanim.production-workspace.v2` while retaining a read-only v1 migration.
A shot version has exactly:

```json
{
  "schema_version": "autoanim.production-workspace.v2",
  "record_type": "shot_version",
  "project_id": "ULID",
  "shot_id": "ULID",
  "take_id": "ULID",
  "shot_version_id": "ULID",
  "ordinal": 2,
  "parent_shot_version_id": "ULID or null",
  "character_revision": {
    "character_id": "ULID",
    "revision_id": "ULID",
    "revision_manifest_sha256": "...",
    "identity_sha256": "...",
    "texture_sha256": "digest or null",
    "texture_uvs_array_sha256": "digest or null",
    "material_manifest_sha256": "...",
    "usage_scope": "production"
  },
  "source": {
    "media_kind": "video",
    "sha256": "take source hash",
    "timebase_sha256": "exact source timing manifest hash"
  },
  "components": {
    "facial_performance": {
      "job_id": "ULID",
      "job_kind": "video_performance",
      "job_manifest_sha256": "...",
      "approval_status": "approved"
    },
    "body_motion": {
      "job_id": "ULID",
      "provider_id": "nvidia_gem_x",
      "job_manifest_sha256": "...",
      "body_track_sha256": "...",
      "validation_report_sha256": "...",
      "approval_status": "not_reviewed"
    },
    "direction": null,
    "artist_overrides": null,
    "composition": null
  },
  "readiness": {
    "automatic_gates_passed": false,
    "all_required_components_approved": false,
    "production_validated": false
  },
  "lifecycle": "review",
  "created_at": "RFC3339 UTC"
}
```

V1 migration maps its single job to `facial_performance`, leaves other
components null, and preserves the original signed bytes/hash. It does not
rewrite historical records.

Component attachment rules:

1. Every component must match the exact project, shot, take source hash,
   character revision, intended-use scope, and duration/timebase.
2. A video take may attach GEM-X body capture. An audio take may attach Kimodo
   or deterministic acting body motion but cannot claim observed body motion.
3. The body job may be replaced only by creating a new shot version. It is
   never overwritten in place.
4. Composition is allowed only after facial/body ownership validation.
5. Publish requires a calibrated GNM/body attachment and both automatic and
   human approvals. Preview remains available with conspicuous status.

### Native macOS workflow

Extend the existing project/character/shot views; do not create a separate
NVIDIA demo window.

1. **Take setup**
   - Choose Body source: `Follow video`, `Generate from approved direction`,
     `Deterministic preview`, or `No body`.
   - Show execution location, exact provider/model version, expected download,
     privacy boundary, and license status.
2. **Provider progress**
   - Show upload/local-only decision, dependency readiness, worker attestation,
     stage, cancellation, retryability, and typed issue.
3. **Body review**
   - Synchronized source video and character.
   - Overlays for SOMA keypoints, canonical bones, root trajectory, camera
     trajectory, foot contacts, confidence, invalid/occluded joints, and
     retarget residuals.
   - A/B candidates for Kimodo; source truth and generated candidates are
     visually distinct.
4. **Ownership timeline**
   - Separate rows for GNM lips/oral, GNM face/eyes, body root/limbs/base head,
     GNM head additive, and artist override.
   - Conflicting writes block composition with a named channel/frame range.
5. **Approval**
   - Approve/reject one immutable body component with reviewer note.
   - Rejection keeps artifacts for diagnosis but prevents publish.
6. **Publish**
   - Show source/model/license hashes, attachment calibration, automatic
     metrics, reviewer approvals, and unresolved warnings.

The UI must never display a CUDA/Apple preview run as equivalent merely because
both return a SOMA archive.

## Phased execution

Every phase uses the strict loop:

1. build only the phase scope;
2. review correctness, architecture, security, licensing, edge cases, and
   claims against this specification;
3. run unit, integration, tamper, and defined real-input tests;
4. fix every failure;
5. rerun the phase suite plus the complete existing regression; and
6. record exact commands, versions, hashes, metrics, skips, and blockers.

No phase advances because mocked tests pass. Optional dependency skips do not
satisfy a phase's real-input gate.

### Phase N0 — freeze baseline and licensing

**Build**

- Add a dependency ledger for GEM-X, SOMA-X, Kimodo, every submodule, model,
  model license, auto-downloaded asset, container, and notice.
- Select immutable upstream revisions and record recursive tree hashes.
- Add explicit provider installation commands that download to a dedicated
  cache, verify before load, then operate offline.
- Freeze the current full Python, Rust, Swift, real-audio, CREMA-D face/video,
  MPFB body, and native-app baseline.
- Obtain written product/legal dispositions for hosted inference, local
  inference, derived motion storage, model redistribution, telemetry, and
  commercial output.

**Files planned**

- `docs/NVIDIA_BODY_DEPENDENCIES.md`
- `assets/notices/NVIDIA_GEM_X_NOTICE.txt`
- `assets/notices/NVIDIA_SOMA_X_NOTICE.txt`
- `assets/notices/NVIDIA_KIMODO_NOTICE.txt`
- `scripts/bootstrap_nvidia_body_provider.py`
- `tests/test_nvidia_body_bootstrap.py`

**Tests**

- Hash/version/license mutation tests.
- Git LFS pointer mistaken for an asset must fail.
- Moving branch/tag, incomplete submodule, missing notice, unapproved model,
  pickle/TorchScript without security disposition, and unexpected network
  access must fail.
- Reproduce one official GEM-X sample and one official Kimodo-SOMA sample in
  quarantined environments. Retain only artifacts whose redistribution is
  approved.

**Gate**

- All exact dependencies are reproducibly installed.
- All required terms have explicit disposition.
- Current full regression is unchanged.
- If legal or security review rejects a required asset, stop. Do not begin an
  adapter around it.

### Phase N1 — contracts, safe provider boundary, and source timing

**Build**

- Implement the request, response, attestation, native SOMA, provenance, and
  validation schemas in a new `nvidia_body_provider.py`.
- Implement safe JSON/NPZ readers independently of provider code.
- Freeze `somaskel77-v1.json` from the exact pinned GEM-X/Kimodo asset and,
  if SOMA-X is installed, freeze its distinct dummy-root-plus-77 contract
  separately.
- Implement local HMAC and remote Ed25519 attestation verification.
- Preserve exact video PTS and rational timebase from the existing video
  capture/probe layer.
- Add typed job states: `dependency_blocked`, `queued`, `running`,
  `validating`, `succeeded`, `failed`, `cancelled`.

**Files planned**

- `src/autoanim_gnm/nvidia_body_provider.py`
- `src/autoanim_gnm/data/somaskel77-v1.json`
- `src/autoanim_gnm/data/soma-x-78-v1.json` if the SOMA-X runtime is approved
- `tests/test_nvidia_body_provider.py`
- `tests/test_nvidia_body_security.py`

**Tests**

- Unknown/duplicate JSON members, NaN/Inf, oversized input, traversal,
  symlinks, ZIP bombs, object arrays, duplicate NPZ members, dtype/shape
  confusion, unsafe names, and hash mismatch all fail before allocation/use.
- Request/response ID, nonce, request hash, worker build, key ID, signature,
  timestamp, artifact names, and hashes are inseparable.
- Replayed, expired, cross-request, cross-character, and cross-worker
  responses fail.
- All 77 `somaskel77` names/parents/basis semantics match the frozen schema,
  and artifact-specific rest positions/bind rotations pass independent FK
  exactly; a SOMA-X array containing its dummy root is rejected by the
  `somaskel77` loader.
- GEM-X and Kimodo contact names, joint IDs, semantics, and column counts match
  their exact versioned contact schemas; wrist contacts cannot reach canonical
  foot-contact output.
- VFR and non-zero-start videos preserve exact source PTS and stable 48 kHz
  projection.

**Real-input test**

- Pass an actual official SOMA motion archive and an actual probed VFR video
  through the boundary without running retargeting. Tamper one byte in each and
  prove fail-closed behavior.

**Gate**

- Provider artifacts cannot claim production validation.
- No provider module is imported into the application process.
- The complete existing body/provider/acting/video/production suite passes.

### Phase N2 — SOMA-to-canonical retarget and BodyTrack v2

**Build**

- Implement `BodyTrackV2`, v1 promotion, validators, canonical JSON hashing,
  and the separate gaze-track extraction.
- Implement exact basis/rest alignment, hips swing-twist decomposition,
  two-neck-to-one-head transfer, PTS-based resampling, scale calibration, and
  sign-continuous quaternion output.
- Preserve the complete SOMA-77 archive alongside the lossy canonical-25
  result.
- Implement raw and cleanup derivations as separate immutable artifacts.
- Implement foot-contact fusion and bounded leg IK cleanup. Cleanup cannot
  change an unconstrained frame by more than configured translation/rotation
  limits and never changes GNM channels.
- Extend the Rust physics/contact core only if profiling shows its existing
  kernels materially help deterministic contact cleanup. NVIDIA inference and
  retarget correctness do not depend on a Rust rewrite.

**Files planned**

- `src/autoanim_gnm/body_track_v2.py`
- `src/autoanim_gnm/soma_retarget.py`
- `src/autoanim_gnm/body_motion_validation.py`
- `tests/test_body_track_v2.py`
- `tests/test_soma_retarget.py`
- `tests/test_soma_retarget_real.py`

**Tests**

- Identity/rest motion maps to exact target rest within `1e-6` radians and
  `1e-6` m.
- Known single-axis rotations map to the intended anatomical joint and sign.
- Root yaw, pelvis swing, composed neck/head, left/right, toe, hand, and
  coordinate-basis mutations are individually tested.
- Quaternions remain normalized within `2e-5`, sign-continuous, and finite.
- Common quaternion FK matches the retained provider-native GEM-X axis-angle or
  Kimodo matrix artifact within 1 mm RMS and 0.1 degrees on valid joints.
- GEM-X wrist contact toggles never change canonical foot contacts; each
  provider's exact left/right heel/toe/foot combinations are tested.
- FK agrees with the retarget reference within 1 mm RMS on mapped joints.
- Bone length varies `<=0.5%`.
- Source PTS to target tick error is at most half one output sample and exact at
  shared sample instants.
- V1 promotion is numerically lossless and preserves gaze values in its
  sidecar.
- Finger/face source arrays remain hash-identical in the retained native
  archive and are absent from the canonical body's owned channels.

**Real-input tests**

- Retarget at least:
  - one official Kimodo-SOMA 77-joint generated sample with locomotion;
  - one official GEM-X SOMA sample;
  - one rights-cleared non-model BVH/mocap clip converted to the exact SOMA
    schema; and
  - three real retained MPFB body proportions.
- Round-trip each canonical track through the planned UsdSkel and glTF
  animation representation when those writers exist; until N5, compare FK
  matrices and an independent reference implementation.

**Gate**

- Neutral vertex RMS `<=0.1 mm`, maximum `<=0.5 mm` for body export round-trip.
- Planted-foot drift p95 `<=20 mm`, ground penetration p99 `<=10 mm` on
  labeled retained clips.
- No GNM coefficient, eye, lip, tongue, or oral artifact is created or changed.
- An independent reviewer signs off the left/right, axes, root, and neck/head
  mapping.

### Phase N3 — GEM-X video-follow body capture

**Build**

- Implement pinned GEM-X CUDA worker and optional separately named
  Apple-Silicon preview worker.
- Integrate source-video staging without changing the existing facial
  performance path.
- Export raw 2D evidence, camera-space motion, world-space motion, camera
  trajectory, SOMA motion, and provider diagnostics.
- Detect multi-person scenes. Require explicit stable subject selection; never
  jump identities silently.
- Add confidence/occlusion segmentation. Short gaps may use bounded
  interpolation; long gaps remain invalid and review-blocking.
- Add static/moving-camera classification and root/camera diagnostics.
- Add a `body_capture` child job to the service and API.

**Files planned**

- `workers/gem_x/`
- `src/autoanim_gnm/gem_x_worker.py` or a worker launcher with no GEM imports
- `src/autoanim_gnm/body_capture_pipeline.py`
- `tests/test_gem_x_worker.py`
- `tests/test_body_capture_pipeline.py`
- `tests/test_body_capture_real.py`

**Tests**

- No-person, multiple-person, subject exit/re-entry, truncation, severe
  occlusion, mirrored video, variable frame rate, dropped/corrupt frames,
  portrait rotation, non-zero PTS, static camera, moving camera, and camera cut.
- GEM-X failure never changes or suppresses a successful GNM face result.
- CUDA and Apple preview outputs are tagged with distinct provider/runtime
  identities and are never cache-interchanged.
- Cancel/restart leaves no successful partial result and does not reuse an
  unsigned archive.

**Real-input cohort**

Use an explicitly consented, product-cleared cohort, not CREMA-D alone:

- walking toward/across camera;
- sitting and standing;
- turning 180/360 degrees;
- fast, crossed-limb, hands-to-face, and hand-behind-body gestures;
- planted-foot weight shifts;
- loose clothing and at least three body proportions;
- static, handheld, pan/tilt, and dolly camera;
- full visibility, partial crop, and temporary occlusion; and
- at least one synchronized optical/inertial mocap and camera-calibration
  reference.

CREMA-D may remain a face/audio regression, but its framing and labels do not
qualify full-body reconstruction.

**Gate**

- Exact PTS and frame coverage.
- p95 2D reprojection `<=2%` of body height.
- MPJPE `<=60 mm` where 3D truth exists.
- Bone-length variation `<=0.5%`.
- Contact precision and recall `>=0.90`.
- Planted-foot drift p95 `<=20 mm`.
- Ground penetration p99 `<=10 mm`.
- Identity switches: zero on the locked single-subject cohort.
- Moving-camera results remain review-required until the camera/root solve
  passes its independently calibrated subset.
- Two body animators prefer GEM-X-retarget output over the deterministic
  fallback on at least 80% of representative video clips, with disagreements
  retained.

### Phase N4 — Kimodo generated acting and constrained repair

**Build**

- Add an explicit acting-plan approval record. Unapproved LLM output cannot
  invoke Kimodo.
- Compile approved beats to bounded prompt segments and allowlisted constraints.
  Use existing stance/gesture vocabulary plus exact root path, end-effector,
  foot-contact, and pose keyframes supplied by an artist or measured source.
- Run pinned Kimodo-SOMA with fixed candidate count, seed set, diffusion
  settings, checkpoint, and postprocess policy.
- Generate 2-4 candidates; validate and rank only by declared automatic
  criteria. Artist selection remains mandatory.
- Support constrained gap repair by pinning trusted boundary poses, feet,
  root, and unaffected limbs. Do not regenerate the whole take for one bad
  interval.
- Keep Codex/Claude in the current no-tool declarative role. The compiler—not
  the LLM—writes Kimodo constraints.

**Files planned**

- `src/autoanim_gnm/acting_approval.py`
- `src/autoanim_gnm/kimodo_constraints.py`
- `src/autoanim_gnm/kimodo_provider.py`
- `tests/test_acting_approval.py`
- `tests/test_kimodo_constraints.py`
- `tests/test_kimodo_provider.py`
- `tests/test_kimodo_real.py`

**Tests**

- Prompt injection, paths, URLs, commands, unknown joints, overlapping
  segments, out-of-range ticks, invalid candidates/seeds, overlong prompts,
  and unapproved plans fail before the worker.
- LLM output cannot choose repository/model/runtime/worker arguments.
- Constraints are character/timebase/source-bound and cannot be replayed.
- Generated output has zero observation confidence.
- Candidate reproducibility is exact for the same pinned runtime/seed, or the
  provider explicitly records nondeterminism and cache reuse is disabled.
- Boundary-pose, root path, end-effector, foot-contact, velocity, and
  acceleration discontinuity tests cover repaired intervals.
- GNM lipsync/oral/face/eye controls are byte-identical before and after body
  generation.

**Real-input tests**

- Run the released Kimodo-SOMA model on at least 20 approved prompts covering
  calm dialogue, angry dialogue, seated conversation, pointing, open palm,
  shrugging, walking, turning, stop/start, asymmetric gesture, and two-part
  multi-prompt transitions.
- Run at least five constraints files with exact hand/foot positions, root
  paths, and boundary poses.
- Repair at least ten deliberately masked segments from rights-cleared real
  mocap and compare to held-out truth.

**Gate**

- Required positional constraints median `<=20 mm`, p95 `<=50 mm`.
- Required rotational constraints median `<=5 degrees`, p95 `<=12 degrees`.
- Root-path lateral p95 `<=50 mm`.
- Foot contact/drift/penetration meet N3 gates.
- Repair boundary velocity jump `<=0.20 m/s` and angular velocity jump
  `<=30 degrees/s`, unless an approved impact event says otherwise.
- Blinded animator approval of the selected candidate `>=70%`, and no automatic
  semantic metric may substitute for this review.

### Phase N5 — unified character composition, export, and native review

**Build**

- Implement production-workspace v2 component records and non-destructive v1
  migration.
- Add body component create/attach/replace APIs with exact character/source/
  duration/timebase checks.
- Calibrate GNM-to-body head socket per character revision; bind the calibration
  and seam/material strategy to exact body and GNM hashes.
- Implement the owned-layer compositor and conflict detector.
- Bind canonical body motion to the real MPFB skin.
- Add UsdSkel editable-master and glTF/VRM review exporters.
- Extend review bundle/viewer/native UI for body, source/camera trajectories,
  confidence, contacts, candidates, layers, and approvals.
- Add publish readiness checks for body license, worker attestation, model
  provenance, character consent, attachment calibration, automatic metrics,
  and artist approval.

**Files planned**

- `src/autoanim_gnm/production.py` v2 migration and component methods
- `src/autoanim_gnm/body_compositor.py`
- `src/autoanim_gnm/body_export.py`
- `src/autoanim_gnm/review_bundle.py`
- `native/autoanim-macos/Sources/AutoAnimMacCore/ProductionLibraryModels.swift`
- `native/autoanim-macos/Sources/AutoAnimMac/ProductionLibraryView.swift`
- `native/autoanim-macos/Sources/AutoAnimMac/ReviewWorkspaceView.swift`
- matching Python, API, viewer, Swift, and export tests

**Tests**

- V1 records verify without rewrite; v2 records survive restart and tampering
  fails closed.
- Cross-source, cross-character, cross-revision, cross-duration, cross-model,
  and cross-timebase component attachment fails.
- Face, lipsync, eye, tongue, body, head-additive, and override conflicts are
  checked at exact frame/tick ranges.
- Body-on versus body-muted comparison proves all GNM coefficient and oral
  arrays are byte-identical.
- Base head rotation appears exactly once.
- Three body proportions pass bind/inverse-bind, weight, seam, and animation
  round-trip tests.
- USD validation, glTF validation, VRM humanoid validation, restart,
  cancellation, storage exhaustion, corrupt cache, offline launch, and
  accessibility tests.

**Real-input end-to-end tests**

1. Consented video -> existing GNM facial performance + GEM-X body ->
   calibrated character -> approved shot version -> USD/glTF -> native review.
2. Real audio -> existing lipsync/oral track + approved LLM direction -> Kimodo
   body candidates -> artist selection -> calibrated character -> publish.
3. Consented video with a bad body interval -> GEM-X raw -> constrained Kimodo
   repair -> source-authority comparison -> approved composite.

Every test uses actual model inference, an actual MPFB body, an actual GNM
character revision, actual source media, and the real native review path.

**Gate**

- Neck seam gap `<=0.25 mm` in neutral and the defined head/neck range of
  motion; normal discontinuity and skin/material seam pass artist look-dev.
- Neutral vertex RMS `<=0.1 mm`, maximum `<=0.5 mm` after export/import.
- Skin weights sum to `1 +/- 1e-4`; bind/inverse-bind error `<=1e-6`.
- Zero USD/glTF/VRM validator errors.
- GNM lipsync, facial coefficients, eyes, teeth, tongue, and oral validation
  remain unchanged when the body component changes.
- A complete shot can be reopened offline and all hashes, approvals, source
  synchronization, and exact frame cursor checks still pass.

### Phase N6 — production qualification and release

**Build**

- Freeze evaluation cohorts, ground truth, demographics/body-proportion slices,
  difficult-camera slices, language/acting slices, rights, retention, deletion,
  and reviewer protocol.
- Run reliability, load, recovery, disk-pressure, worker-loss, cancellation,
  retry, cache-corruption, privacy, and security tests.
- Benchmark CUDA cost/latency and Apple preview latency separately.
- Establish model/update governance: no automatic model revision upgrades.
- Complete legal, privacy/biometric, security, accessibility, animator, and
  operations sign-off.

**Gate**

- N3-N5 metric gates hold on held-out data, not provider demos.
- At least two independent body-animation reviewers approve every release
  slice.
- No known critical/high security findings.
- 100% provenance, consent, model, code, asset, and notice coverage.
- Restart/recovery loses no approved work and never promotes partial output.
- `production_validated` changes to true only in the sealed release report and
  eligible shot versions whose exact component hashes were evaluated.

## Dependency graph and sequencing

```text
N0 pins/licenses
   |
   v
N1 contracts + attested worker boundary
   |
   v
N2 SOMA retarget + BodyTrack v2
   | \
   |  \--> N4 Kimodo generation/repair
   v
N3 GEM-X video capture
   \        /
    \      /
      N5 composition/export/native review
                    |
                    v
              N6 qualification
```

N3 and N4 can be developed in parallel only after N2 passes. N5 may prototype
UI against retained N2 artifacts, but its phase gate requires real N3 and N4
outputs. N6 cannot begin release evaluation against unfrozen model revisions.

## Failure alternatives

| Failure | Required behavior | Viable alternative |
|---|---|---|
| GEM-X dependency/model/license unavailable | Typed blocked job; face pipeline remains usable | Deterministic acting body preview, or external rights-cleared mocap imported through retarget-only |
| Apple preview differs from CUDA | Preserve both identities/reports; preview cannot publish | Queue exact CUDA worker or artist imports approved motion |
| Remote worker cannot be authenticated | Reject all artifacts regardless of valid-looking motion | Local attested worker or manual offline transfer with operator-signed provenance |
| SOMA schema changes upstream | Reject digest/order mismatch | Add a new versioned schema and reviewed migration |
| Retarget axes/left-right/rest test fails | No canonical track | Retain native SOMA for DCC/manual correction; author a new immutable profile |
| Long video occlusion/multi-person ambiguity | Invalid intervals and review block | Explicit subject selection, alternate take, external mocap, or constrained Kimodo repair with source boundaries |
| Moving-camera root solve fails | No world-motion production claim | Camera-space preview or static-camera recapture |
| Kimodo cannot meet constraints | Reject candidate; do not relax silently | New seed/candidate, more constraints, deterministic fallback, or animator keyframes |
| Insufficient GPU memory | Typed dependency/resource block | Kimodo text encoder on CPU, approved remote CUDA, or deterministic fallback |
| Attachment calibration/seam absent | Separate face/body preview only | Artist calibrates attachment or publish head-only |
| License/provenance incomplete | Publish blocked | Select an approved provider/model or acquire written terms |
| Ground truth unavailable | Metric is null and production gate false | Commission consented capture; do not substitute provider self-evaluation |

## Explicit non-goals

- Replacing GNM facial identity, expression, lipsync, eye, tongue, teeth, or
  oral geometry with SOMA, GEM-X, or Kimodo.
- Driving GNM directly from GEM-X face/jaw/eye joints in this plan.
- Claiming single-image body identity or clothing reconstruction. The existing
  image-to-GNM face work and MPFB body profiles remain separate until a
  rights-cleared, measured body-identity project is specified.
- Bundling or using SMPL/SMPL-X, Kimodo-SMPLX, SOMA-shape, or other separately
  restricted identity assets by implication.
- Integrating GEM-SMPL/GENMO or any noncommercial/research-only weight into the
  production route.
- Supporting Unitree G1, robot control, physical execution, or robotics safety.
- Treating generated Kimodo motion as observed performer motion.
- Treating the Apple-Silicon GEM demo as automatically equivalent to the full
  CUDA path.
- Real-time Kimodo generation in the first release.
- Full finger animation on the existing canonical 25-joint runtime. Native
  SOMA fingers are retained for a later versioned skeleton extension.
- Solving hair, garments, cloth, muscle, soft-tissue, collision, or secondary
  motion. The existing Rust physics core remains a downstream optional layer.
- Training or fine-tuning NVIDIA models before rights-cleared paired data,
  training terms, evaluation, and update governance exist.
- Sending media, likeness, transcript, or acting instructions to an external
  worker without explicit user choice, consent, retention policy, encryption,
  and authenticated transport.
- Calling any phase production-ready because an upstream demo or mocked test
  runs.

## Definition of completion

This integration is complete only when:

1. an exact, approved GEM-X video job produces a validated native SOMA archive
   and canonical body track from actual consented video;
2. an exact, approved Kimodo-SOMA job produces selectable, constraint-checked
   body acting from an approved AutoAnim acting plan and real audio context;
3. both tracks animate a real, retained MPFB body with a calibrated GNM head;
4. GNM face, lips, eyes, teeth, tongue, and oral results remain independently
   validated and ownership-conflict-free;
5. the native macOS app can inspect source/body/camera/contact/confidence,
   compare candidates, approve immutable components, reopen them after restart,
   and publish verified USD/glTF;
6. exact code/model/asset/license/provenance, consent, component, approval, and
   result hashes survive tamper tests;
7. every phase gate and full regression passes with real inputs and no
   required-test skip; and
8. remaining limitations are visible in the sealed validation/release report,
   with `production_validated=false` for every artifact not independently
   qualified.

## Primary upstream references

- [NVIDIA GEM-X repository](https://github.com/NVlabs/GEM-X)
- [NVIDIA GEM-X model repository](https://huggingface.co/nvidia/GEM-X)
- [NVIDIA SOMA-X repository](https://github.com/NVlabs/SOMA-X)
- [NVIDIA SOMA-X documentation](https://nvlabs.github.io/SOMA-X/stable/)
- [NVIDIA Kimodo repository](https://github.com/nv-tlabs/kimodo)
- [NVIDIA Kimodo documentation](https://research.nvidia.com/labs/sil/projects/kimodo/docs/)
- [NVIDIA Kimodo motion representation](https://research.nvidia.com/labs/sil/projects/kimodo/docs/api_reference/motion_rep.html)
- [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
