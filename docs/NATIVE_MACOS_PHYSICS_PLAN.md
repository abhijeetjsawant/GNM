# Native macOS application and facial-physics plan

Status: implementation contract with retained execution results

Audited repository revision: `45d0e9ffc5c80cbf9ad5f2c526d33aa9d469d5fe`

## Decision summary

AutoAnim will become a native Apple Silicon macOS application built with
SwiftUI and focused AppKit/WebKit bridges.  The existing Python
`ApplicationService`, sealed job/character storage, and verified pipelines
remain authoritative behind supervised helpers.  The current Three.js viewer
is embedded during migration and is later replaced by an `MTKView` renderer.

The facial-physics layer will be a separate Rust core.  Learned audio/video
performance, director edits, gaze, blink, jaw, and tongue articulation remain
authoritative kinematic targets.  Physics operates in target-relative space and
adds only tissue inertia, regional compliance, volume preservation, and
contact.  It must not act as a generic smoother: that would soften consonants,
delay contacts, and damage acting.

The first solver is CPU-first.  One GNM v3 head has 17,821 vertices, 35,324
triangles, and 53,135 unique edges.  Current NumPy GNM evaluation measures
8.82 ms/frame on the retained M2 Max.  Metal or wgpu compute is promoted only
when an end-to-end benchmark, including synchronization and readback, proves a
material win.

The first executable physics slice is honestly named
`surface_secondary_candidate`.  It is real target-driven surface dynamics, but
it is not production facial tissue: GNM supplies open render surfaces and no
skull, mandible joint, palate, closed tongue volume, fat/muscle volume, or
personal material calibration.

## Execution record — 2026-07-19

Phase P0 passed its correctness and CPU-performance gates after a review/fix
loop.  Independent reruns on this M2 Max measured the 11,556-vertex benchmark
at 1.571 ms/frame p95 with eight threads versus 3.577 ms/frame p95 with one
thread, a 2.321x p95 speedup.  The full-reporting path measured 2.471
ms/frame p95.  The actual GNM v3 C-ABI path measured 3.587 ms/frame p95 for
17,821 vertices, 35,324 triangles, and 53,135 unique edges.  Sequential GNM
evaluation measured 9.098 ms/frame p95, giving a conservative summed p95 of
12.685 ms/frame against the 16.67 ms combined budget.

The SIMD promotion gate failed honestly: stable auto-vectorization and explicit
NEON each measured only about 1.01x the isolated scalar predictor in the
independent rerun, below 1.10x.  `Auto` therefore does not select NEON and
production reports set `simd_claim=false`.  No wgpu or Metal backend was built,
because the CPU single-character workload does not satisfy the evidence needed
to justify that complexity; the conditional multi-character gate remains.

Phase N1's SwiftUI source, authenticated supervisor, native job library,
diagnostics, WKWebView, release build, and headless authenticated smoke are
implemented.  The linker-signed native executable launches and supervises the
real backend through `scripts/run_macos_app.sh`.  The assembled `.app` does not
pass the launch gate on this host: Developer Mode is disabled and AMFI kills
both ad-hoc and local Apple-Development re-signed bundles with SIGKILL/137.
The bundle remains a failed artifact rather than a claimed success.  Viable
paths are the retained developer-executable launcher now, or a properly
provisioned/Developer-ID signed and notarized bundle in Phase R.  Rebuilt Swift
test bundles are now denied by the same host policy; the last pre-policy run
passed 11 tests, and the later source still compiles in release mode.

Phase P1 has not started.  The Rust binding is opt-in and is deliberately not
inserted into audio/video artifacts until shared evaluated frames, protected
oral regions, retained reports, and the real media regression gates exist.

## Research basis

The implementation choices are grounded in:

- [Apple SwiftUI application architecture](https://developer.apple.com/documentation/technologyoverviews/swiftui)
- [Apple NavigationSplitView](https://developer.apple.com/documentation/swiftui/navigationsplitview)
- [Embedding a command-line helper in a sandboxed app](https://developer.apple.com/documentation/xcode/embedding-a-helper-tool-in-a-sandboxed-app)
- [App Sandbox file and helper access](https://developer.apple.com/documentation/security/accessing-files-from-the-macos-app-sandbox)
- [Apple MTKView](https://developer.apple.com/documentation/metalkit/mtkview)
- [Apple Metal compute guidance](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu)
- [Apple Silicon Metal guidance](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)
- [XPBD](https://matthias-research.github.io/pages/publications/XPBD.pdf)
- [Projective Dynamics](https://infoscience.epfl.ch/entities/publication/00c49830-4e0c-485d-a752-939abba0e2a6)
- [Enriching Facial Blendshape Rigs with Physical Simulation](https://la.disneyresearch.com/wp-content/uploads/Enriching-Facial-Blendshape-Rigs-with-Physical-Simulation-Paper2.pdf)
- [Building and Animating User-Specific Volumetric Face Rigs](https://users.cs.utah.edu/~ladislav/ichim16building/ichim16building.pdf)
- [Interactive Facial Animation (2025)](https://animation.rwth-aachen.de/media/papers/95/2025-PACMCGIT-Facial_Animation.pdf)
- [Efficient IPC for Actuated Face Simulation](https://cgl.ethz.ch/Downloads/Publications/Papers/2023/Yan23b/Yan23b.pdf)
- [Stable Neo-Hookean Flesh Simulation](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf)
- [G-OBIM tongue model](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013378)
- [Rust portable SIMD remains unstable](https://doc.rust-lang.org/beta/unstable-book/library-features/portable-simd.html)
- [Stable AArch64 intrinsics](https://doc.rust-lang.org/stable/core/arch/aarch64/index.html)
- [Rayon data parallelism](https://docs.rs/rayon/latest/rayon/)
- [wgpu Metal backend](https://docs.rs/wgpu/latest/wgpu/)

## Repository facts and constraints

- GNM Head v3: 17,821 vertices, 35,324 triangles, 17,662 quads,
  253 identity coefficients, 383 expression coefficients, and four joints.
- The expression layout is left eye `0:100`, right eye `100:200`, lower face
  `200:350`, tongue `350:382`, and pupil `382:383`.
- The model has only neck, head, and two eye joints.  Teeth, tongue, and oral
  skin are head-skinned; there is no mandible joint.
- Skin is an open surface.  Tongue and teeth are also open surfaces, so the
  existing proximity checks cannot prove signed penetration freedom.
- Lower-face bases move lower teeth almost rigidly.  A later phase can recover
  a physical mandible transform by Kabsch fitting those vertices.
- `ApplicationService` is already shared by HTTP and CLI.  FastAPI is a
  transport layer; completed jobs and character revisions are file-backed,
  atomically published, and integrity sealed.
- The current web API serializes all jobs with one global lock.  Native
  scheduling must eventually distinguish exclusive MLX/A2F work, bounded CPU
  work, media decode, and manifest mutations.
- The development `.venv` is roughly 1.8 GB and contains hundreds of Mach-O
  dependencies.  It is not a releasable app runtime.
- The installed Homebrew FFmpeg is GPL-enabled.  It must not be copied into a
  proprietary app without a deliberate distribution/licensing decision.
- A2F assets/runtime are hundreds of megabytes; Blender/body assets are over a
  gigabyte and belong in optional model/provider packs.

## Target architecture

```text
AutoAnim.app
├── SwiftUI/AppKit host
│   ├── native character/job/performance library
│   ├── import, toolbar, menus, settings, diagnostics
│   ├── JobCoordinator actor and resource scheduler
│   ├── AVFoundation media clock
│   └── WKWebView bridge → later MTKView renderer
├── AutoAnim Rust core
│   ├── versioned physics/profile schemas
│   ├── deterministic CPU reference
│   ├── Rayon + measured SIMD kernels
│   ├── contact/PD/FEM fidelity tiers
│   ├── stable C ABI for Swift
│   └── Python binding from the same core/version
└── supervised helpers
    ├── packaged Python worker
    ├── exclusive Swift/MLX A2F worker
    ├── reviewed media tools
    └── optional body/model providers
```

Runtime data belongs under `Application Support/com.autoanim.AutoAnim/` with
separate Jobs, Characters, Models, Integrity, and Logs directories.  Disposable
decoded frames and previews belong under the application cache directory.

The first native slice may supervise FastAPI, but it must bind loopback only,
use an unpredictable per-launch session token, validate `Host`, and provide the
token to WebKit through an HttpOnly same-site cookie.  The target production
worker protocol is versioned NDJSON or length-prefixed messages over pipes;
large binary artifacts remain files and are addressed by hash.

## Physics contract

Per frame:

```text
GNM coefficients and pose
  → exact face-local kinematic target
  → rigid collider transforms
  → dynamically rebalanced target-relative solve
  → localized contact
  → render-mesh embedding
  → existing oral/timing/export audits
```

Kinematic and editable:

- identity, learned/director expression, phonetic timing, gaze, blink;
- head/body acting and jaw trajectory;
- rigid teeth/eye transforms;
- annotated lip and tongue contact timing.

Simulated when the relevant fidelity tier exists:

- cheek, lip, jowl, chin, and soft-nose inertia;
- regional strain/bending, damping, and expression-dependent stiffness;
- passive volume and skin sliding;
- lip/lip, lip/teeth, tongue/teeth/palate, and external contact;
- high-frequency wrinkles only in a localized/offline or validated surrogate
  tier.

The initial surface candidate pins lips, tongue, teeth/gums, eyes, and oral
interior exactly.  It cannot alter lip-sync geometry.  A hard displacement cap
and retained report prevent an uncalibrated profile from creating large motion.

## Phased execution plan

### Phase 0 — baseline and research (complete)

- Commit and push the verified existing implementation.
- Audit GNM topology, runtime costs, native dependencies, and app boundaries.
- Compare SwiftUI/AppKit, Tauri, Electron, and immediate rewrites.
- Compare XPBD, projective dynamics, FEM, IPC, and GPU execution.

Gate: repository clean/pushed; architecture and limitations documented from
repo code and primary references.

### Phase N1 — executable native Mac slice (source complete; bundle gate blocked)

Build:

- SwiftUI `WindowGroup` with native navigation/sidebar, health, job list,
  selection, diagnostics, refresh/restart, and error states.
- Supervised development worker with explicit project/runtime resolution.
- Authenticated loopback service and WebKit cookie.
- Embedded existing viewer for selected viewable jobs.
- Clean child termination and no-orphan lifecycle behavior.
- Reproducible `.app` assembly with an honest development signature/label.

Tests:

- Swift unit tests for path resolution, loopback URL policy, JSON decoding,
  job selection, and launch arguments.
- Python API tests for missing/wrong/correct token and Host rejection.
- `swift test`, release build, `plutil`, `codesign --verify`, and launch smoke.
- Live service cold start, health, real recent jobs, selected real viewer, app
  termination, and orphan-process check.

Gate: a native `.app` opens and supervises the current backend, real jobs and a
real GNM viewer are usable, unauthorized loopback access fails, and the app
leaves no helper behind.  This is a developer app, not yet a notarized portable
release.

### Phase N2 — native workflows and resource scheduler

- Native image, multiview, audio, video, acting, character, and material forms.
- `NSOpenPanel`, drag/drop, progress, cancellation, and recovery.
- `JobCoordinator` resource classes: exclusive MLX, bounded CPU/media, and
  atomic manifest/character publication.
- Native production-readiness, consent, provenance, and qualification views.

Gate: every existing real-input CLI workflow runs from the native UI with the
same retained hashes/metrics, typed cancellation, crash recovery, and no
concurrent resource violation.

### Phase N3 — distributable worker and pipe IPC

- Runtime resource locator; remove repository and Homebrew path assumptions.
- PyInstaller `onedir` worker or an equivalently auditable Python distribution.
- Pipe protocol with hello/start/progress/artifact/completed/failed/cancelled.
- Reviewed LGPL media toolchain or AVFoundation replacement.
- Signed nested helpers and model packs.

Gate: clean-user offline image/audio/video smoke tests have no `.venv`, repo,
`/opt/homebrew`, or undeclared dylib dependency.

### Phase P0 — Rust topology, solver contract, and CPU baseline (passed)

Build:

- Canonical `(min,max)` sorted/unique triangle edges and fixed-order CSR.
- Stateful target-relative Jacobi XPBD solver.
- Dedicated Rayon pool; sequential frame order and parallel vertex/edge work.
- Scalar reference and stable AArch64/auto-vectorized contiguous kernels.
- Strict float32 shape/finite/configuration validation.
- Versioned report with topology/config/input/output hashes, backend, thread
  count, SIMD claim, displacement, strain, and fallback reason.
- Stable C ABI or PyO3 bridge chosen so Swift and Python share one core/version.

Tests:

- Exact 53,135-edge GNM topology; duplicate/self/out-of-range rejection.
- Constant target no-op; pinned vertices bit-identical.
- Rigid transform equivariance, bounded step response, monotonic damping.
- Chunk-size and 1/2/4/N-thread determinism.
- Scalar/SIMD parity and typed unavailable-backend failure.
- No allocations in steady-state solver loops where instrumentable.

Performance gate on retained M2 Max:

- physics-only p95 ≤4 ms/frame for the declared test profile;
- GNM evaluation plus physics p95 ≤16.67 ms/frame;
- ≥1.5× multi-thread speedup over one Rust thread;
- SIMD kernel ≥1.10× scalar or it is removed rather than advertised.

### Phase P1 — safe pipeline integration

- Build one shared evaluated-frame path so MP4, GLB, and validation consume the
  same physical frames.
- Opt-in `surface_secondary_candidate`; default physics remains off.
- Protect lips, tongue, teeth/gums, eyes, and oral interior exactly.
- Retain physics report/config and explicitly fail production readiness.
- Run retained real audio and video tracks through preview, GLB compression,
  oral validation, and media-clock checks.

Gate:

- physics-off artifact hashes unchanged;
- protected-vertex drift exactly zero;
- target-relative displacement ≤0.75 mm initial cap;
- no new lip-order or tongue/teeth-risk frames;
- final mouth-step ≤0.03995 interocular;
- GLB remains ready and within p95 0.10 mm/max 0.50 mm reconstruction gates;
- full frame count and media duration unchanged.

### Phase P2 — mandible and oral contact

- Recover rigid mandible motion from lower teeth/raw jaw evidence.
- Head-fixed maxilla; rigid lower teeth; teeth/palate SDFs.
- Lip/lip, lip/teeth, tongue/teeth/palate candidate sets and CCD.
- Tongue cage around the learned target; phonetic contacts remain timing
  constraints.

Gate: rigid teeth drift ≤0.01 mm, zero audited triangle intersections, signed
penetration ≤0.05 mm tolerance, no introduced lip inversion, contact apex shift
≤ one 60-fps frame, and annotated bilabial/tongue contacts retained.

### Phase P3 — personalized volumetric projective dynamics

- A versioned `CharacterPhysicsProfile` with skull/maxilla, mandible, dental and
  palate proxies, closed tongue/cage, 5k–10k simulation nodes, 20k–50k tets,
  render embedding, regional materials, attachment maps, and provenance.
- PD ARAP/shear, volume/determinant, target, attachment, inertial, and localized
  contact terms with dynamic rebalancing.

Gate: no inverted tets (`det(F)>0.1`), p95 volume drift <1% and max <3%, static
target p95 ≤0.10 mm/max ≤0.50 mm, refinement disagreement ≤0.25 mm, and
head-shake amplitude/phase/decay within the documented capture tolerances.

### Phase P4 — hero/offline tissue and detail

- Stable Neo-Hookean near-incompressible FEM, heterogeneous fat/muscle/skin,
  sliding/ligament attachments, active muscle fields, and calibration from
  multiple expression scans.
- Localized solid-shell or learned simulation-to-displacement wrinkle tier.

Gate: held-out scan p95 ≤1 mm (oral ≤0.5 mm), no inversions/intersections,
parameter confidence intervals retained, monotonic/reversible animator
controls, and no intelligibility regression in blind review.

### Phase G — wgpu/Metal promotion (conditional)

- Persistent buffers and pipeline state; Metal backend only on macOS.
- GPU kernels only for regular high-throughput work such as dense basis
  evaluation, per-tet projection, CSR gather, matrix-free SpMV/PCG, or batched
  characters.
- Custom Metal only if it materially beats wgpu and enables zero-copy sharing
  with the future native renderer.

Promotion gate:

- benchmark 1/4/8 characters and 120/600-frame tracks;
- include upload, encode, submit, synchronization, and readback wall time;
- ≥1.5× CPU end-to-end and ≥2 ms/frame saved for the promoted workload;
- CPU/GPU p95 positional difference ≤0.005 mm and max ≤0.02 mm;
- no oral/contact classification divergence;
- custom Metal must beat wgpu end-to-end by at least 20% for the same output.

If one face does not cross the gate, it remains on CPU.  Apple Silicon unified
memory does not remove command submission or synchronization costs.

### Phase R — hardened release

- Developer ID signing, hardened runtime, inside-out nested signing,
  notarization, stapled DMG, dependency notices, model redistribution review,
  update/model integrity channel, and clean-machine test matrix.
- An App Store build requires API-backed acting and a bundled/remote body
  provider because sandboxed apps cannot invoke arbitrary user-installed
  Codex/Claude/Blender executables.

## Required validation corpus

1. Numerical fixtures: oscillator, hanging patch, rigid motion, damped step,
   volume compression, contact drop, frictionless slide, inversion rejection.
2. Repo regressions: retained audio, tongue-active take, mouth contact/aperture
   repair, moving-face video, textured character, and animated GLB.
3. Real performance: independently annotated phones/contact apexes plus
   120/240-fps head-shake and body-motion captures.
4. Character diversity: at least five held-out identities spanning lip/jaw/fat
   geometry, age, and sex, with neutral, smile, anger, surprise, fast speech,
   whisper, and shouting.

Every performance report records p50/p95/p99, peak memory, warm/cold time,
worker count, CPU/GPU model, power mode, backend, determinism mode, and hashes.

## Truthful completion boundaries

- A native development `.app` is not a distributable, notarized application.
- A target-driven surface shell is not volumetric flesh and cannot prove volume
  preservation or oral collision freedom.
- Physics cannot repair wrong phonemes, weak expression inference, or bad A/V
  alignment.
- A photo or multiview surface cannot identify a person's skull, palate,
  dental occlusion, tongue volume, fat depth, muscle paths, or material
  parameters.  Template-warped anatomy must be labeled inferred.
- Production completion requires the real, diverse, independently annotated
  corpus and artist review—not only synthetic tests or one retained take.
