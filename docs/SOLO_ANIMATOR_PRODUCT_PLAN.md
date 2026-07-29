# Solo animator production workflow

Status: implementation plan, 2026-07-29.

## Product decision

AutoAnim is a local-first macOS application for one animator. Its durable
objects are Projects, Characters and Shots. Background jobs are implementation
detail shown in Activity, not the primary way work is organized.

The working promise is: **create the character once; perform, revise, and
deliver it many times.**

## Object model

```text
Project
  ├─ character references → CharacterRevision (immutable and pinned)
  └─ Shot
       ├─ Take (audio or video source)
       └─ ShotVersion
            ├─ solve jobs
            ├─ direction job
            ├─ correction layers (future)
            └─ review/delivery state (future)
```

`CharacterRevision` remains the authority for identity, UVs, materials,
consent and readiness. A Shot stores its exact character/revision pair when
created. Publishing a later character revision never changes an existing shot.

## v1 screens

The native sidebar is: Projects, Characters, Shots, Review, Deliveries,
Activity and Diagnostics.

The initial build delivers the first three as durable working screens. A shot
contains Setup and Performance stages. Setup selects an immutable character
revision and creates a named audio or video take. Performance links the existing
audio/video job result to that take. Review continues to use the existing exact
frame Review Workspace. Activity continues to use the existing job library.

## Delivery phases

1. **Foundation**: persisted project/shot/take/version model; authenticated API;
   regression tests for revision pinning and invalid lifecycle transitions.
2. **Native library**: production navigation, projects and shot creation, and
   character selection from the existing revision library.
3. **Performance binding**: create audio/video jobs from a shot and attach their
   result to a take without changing identity selection rules.
4. **Polish**: non-destructive correction layers, timeline, mouth/tongue/eye
   controls, and acting-direction beats.
5. **Review and delivery**: timecoded notes, approval states, export presets,
   package manifests and validation gates.

## v1 acceptance tests

1. A project survives service restart and lists its shots.
2. A shot pins a real character revision and rejects unknown/mismatched IDs.
3. A new character revision does not alter a pre-existing shot's pinned revision.
4. An audio or video take can be linked only to a compatible completed job.
5. Native macOS UI lists projects and shots using the authenticated loopback API.
6. Existing character, job, audio, video and review endpoints remain compatible.

## Explicitly deferred

- Multi-user accounts, collaboration and cloud sync.
- Full timeline correction layers and approval/notes entities.
- Renderer-grade 8K skin acquisition, high-frequency pore claims and groom.
- Video-to-character enrollment and subject-specific expression refinement.
- Full-body performance capture and final USD delivery.
