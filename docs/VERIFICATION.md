# AutoAnim GNM Final Verification

Date: 2026-07-18
GNM commit: `3de70dfca5f3244620f44103c24b7cedc0dcb8b6`
Application version: `0.1.0`
Result: **working learned prototype; production approval intentionally withheld**

## Verdict

The four implemented workflows run through one local application service, CLI,
and HTTP API. Real-input or calibrated positive-control evidence now covers:

- audio is decoded and normalized, passed through Audio2Face-3D Claire on
  Apple Silicon, solved into named ARKit/tongue controls, retargeted into
  runtime-sized GNM controls, evaluated on the real 17,821-vertex mesh, and
  rendered to an audible H.264/AAC preview; Rhubarb remains a diagnostic and
  deterministic fallback;
- a single photo is detected by MediaPipe, mapped to GNM's sparse landmark
  convention, fit to observable identity modes, and exported as a neutral OBJ,
  full parameter NPZ, overlay, confidence report, and mesh preview;
- calibrated multiview fitting solves one bounded identity and bakes a
  provenance-aware texture into a seam-correct GLB. The retained positive
  control is synthetic GNM-to-GNM evidence, not photographed-person proof;
- a real moving CREMA-D performance is tracked at native source timestamps,
  densely retargeted into GNM expression/head/eye motion, and exported with a
  synchronized browser video clock;
- the local Three.js viewer displays static, textured, audio-animated, and
  video-animated GLBs with orbit/topology controls and exact media-driven time.

This is a working, verified technical application. It is not yet an approved
production facial rig: the semantic ARKit-to-GNM retarget is not artist
calibrated, and no independent phone/contact annotation corpus or human MOS
panel was provided. The quality gate therefore cannot pass by construction.
Those distinctions are enforced in result warnings and are discussed under
**Known limitations**.

## Verification environment

| Component | Verified value |
|---|---|
| Host | Apple M2 Max, arm64 |
| OS | macOS 26.2, build 25C56 |
| Python | CPython 3.12.13 |
| GNM | v3.0 at commit `3de70dfc...` |
| NumPy / SciPy | 2.5.1 / 1.18.0 |
| MediaPipe / OpenCV | 0.10.35 / 5.0.0 |
| FastAPI | 0.139.2 |
| ffmpeg | 8.1.1 |
| Rhubarb | 1.14.0 complete release bundle |
| Learned motion | Audio2Face-3D v2.3.1 Claire, speech-swift 0.0.23, Swift/MLX |
| Claire output | 140 skin + 10 tongue PCA + 15 jaw + 4 eye controls |
| Retarget assets | SHA-checked official 52 ARKit + 16 tongue target package |
| Face landmarker | SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

The application dependency set is pinned in `requirements.lock`. Bootstrap
downloads both macOS and Linux Rhubarb archives with fixed SHA-256 values and
checks for the companion PocketSphinx dictionary before reporting readiness.

## Final test ledger

The complete application suite, static checks, validator checks, and final
browser smoke were run after the last application change. The upstream GNM and
native Swift suites were run after the final audio compiler change; the later
application-only CSP edit does not touch either codebase.

| Scope | Result | Notes |
|---|---:|---|
| Complete AutoAnim suite | **164 passed, 1 skipped** in 239.40 s; skipped Claire test then **passed separately** in 3.48 s with its required asset environment | Post calibrated-sidecar run: real fallback + learned audio, image, legacy/calibrated multiview and texture, video, viewer, app, solver, contact-anchor, held-out leakage, nonzero-distortion, accepted-set stability, matrix provenance, and adversarial quality tests; the warning is a non-functional Starlette deprecation |
| Native Swift runner tests | **7 passed** | Exact speech-swift 0.0.23 pin, Claire default, typed arguments and emotion controls |
| Google official `run_all_tests.py` | **278 passed** in 39.250 s | NumPy, JAX, PyTorch, TensorFlow, semantic sampler |
| Nested GNM fitting tests | **60 passed** in 8.792 s | Not reached by Google's top-level discovery |
| Nested camera/color tests | **21 passed, 3 skipped** in 0.483 s | Three tests require unavailable TensorFlow Graphics rasterization |
| Shell/source hygiene | **pass** | `bash -n`, `compileall`, and `git diff --check` |
| Khronos glTF validator | **4 final GLBs: 0 errors, 0 warnings** | Post-fix audio, real image, true-native video, and calibrated textured positive control |
| Browser smoke | **pass** | Dashboard healthy; image, texture, audio, and video viewers ready; audio clock played; zero fresh console errors after the embedded-texture CSP fix |

Commands:

```bash
MPLCONFIGDIR=.cache/matplotlib PYTHONPATH=.:src \
  AUTOANIM_A2F_ASSET_DIR=.cache/autoanim_gnm/a2f-claire .venv/bin/pytest -q
swift test --package-path native/a2f-runner
UV_CACHE_DIR=.cache/uv PYTHONPATH=. \
  uv run --with './gnm/shape[all,dev]' python gnm/shape/run_all_tests.py
UV_CACHE_DIR=.cache/uv PYTHONPATH=. \
  uv run --with './gnm/shape[all,dev]' python -m unittest discover \
  -s gnm/shape/fitting_utils -p '*_test.py'
TEST_UNDECLARED_OUTPUTS_DIR=/tmp/autoanim-gnm-visualization-final \
  UV_CACHE_DIR=.cache/uv PYTHONPATH=. \
  uv run --with './gnm/shape[all,dev]' python -m unittest \
  gnm.shape.visualization.camera_conversions_test \
  gnm.shape.visualization.vertex_colors_test
rg --files scripts -g '*.sh' | xargs -n1 bash -n
.venv/bin/python -m compileall -q src tests
git diff --check
```

## Phase evidence

### Phase 0 — upstream truth and environment

- Loaded the checked-in GNM v3.0 asset through Google's public NumPy API.
- Asserted 17,821 vertices, 35,324 triangles, 17,662 quads, 253 identity
  coefficients, 383 expression coefficients, and four joints.
- Recorded upstream architecture, assets, regions, frameworks, and repository
  discrepancies in `RESEARCH.md`.
- Pinned the actual repository commit, Python environment, native binaries,
  model, and test fixtures.

### Phase 1 — adapter and control rig

- Exact TensorFlow-free evaluation of the checked-in H5 expression decoder.
- Compact landmark evaluation agrees with the official GNM path to `1e-6`.
- Viseme blocks cannot write eyes or pupil; A-G also cannot write tongue, and H
  is the only tongue viseme.
- Manual emotion changes permitted face/eye blocks while the viseme delta is
  preserved exactly in the lower-face block.
- Geometry ordering, pucker width, tongue displacement, coefficient bounds,
  topology, finite vertices, OBJ output, and dense rendering pass on real GNM.

### Phase 2 — real audio

Retained job: `01kxs6pg3vnm587tnna54gmxyd`

| Metric | LibriSpeech result |
|---|---:|
| Input | 8.000 s human speech |
| Rhubarb cues | 50 |
| Track | 240 x 383 at 30 fps |
| Video frames | 240 / 240 controls |
| Cue coverage | 1.000 |
| A/V tail difference | 0.000 frames |
| Maximum coefficient | 2.9655, within 3.0 bound |
| Mouth aperture range | 0.01406 model units |
| Mesh finite | true |
| Automatic label | neutral, 0.50, **unvalidated** |

Retained job: `01kxs6qbq6w513gt3p6v5mcz6b`

| Metric | RAVDESS angry result |
|---|---:|
| Input | 4.104125 s emotional speech |
| Rhubarb cues | 14 |
| Track | 124 x 383 at 30 fps |
| Video frames | 124 / 124 controls |
| Cue coverage | 1.000 |
| A/V tail difference | 0.87624 frames |
| Maximum coefficient | 2.6860, within 3.0 bound |
| Mouth aperture range | 0.02530 model units |
| Mesh finite | true |
| Automatic label | anger, 0.62, **unvalidated** |

The first/middle/last frames of both retained videos were visually inspected.
They show non-inverted, finite meshes, meaningful mouth changes, and neutral
return. The videos contain normalized source audio and expose the required
coarse-alignment and unvalidated-emotion caveats.

#### Production-lipsync upgrade

Retained learned outputs:

- `artifacts/production-lipsync/libri-learned`
- `artifacts/production-lipsync/ravdess-learned-anger`

| Metric | Original Libri | Learned Libri | Original RAVDESS | Learned RAVDESS anger |
|---|---:|---:|---:|---:|
| Frozen lower-face transitions | 30.5% | **0.0%** | 49.6% | **0.0%** |
| Mouth-step p95 / interocular | 0.085 | **0.025** | 0.057 | **0.031** |
| Lower-face velocity p95 | 1.793 | **0.464** | 1.286 | **0.613** |
| Acceleration p95 | 2.195 | **0.282** | 1.419 | **0.336** |
| Jerk p95 | not retained | **0.455** | not retained | **0.423** |
| Video/control frames | 240/240 | 240/240 | 124/124 | 124/124 |

The learned backend emitted 241 and 124 timestamped 169-D raw frames. Both
runs exported finite `[N,52]` ARKit skin and `[N,16]` tongue arrays plus exact
length `[T,383]` GNM controls. The Libri raw stream was byte-identical across
two executions. Explicit anger was passed through Audio2Face's native emotion
input at 0.65 strength; no LLM or heuristic generated mouth timestamps.

The geometry-first benchmark has adversarial tests for plus/minus two- and
four-frame shifts, excessive smoothing, static/constant-open motion, cue
permutation, and emotion-only silence motion. Every mutant fails. Real outputs
still report `production_validated: false`, because timing/contrast approval
requires independent annotations rather than the system's own Rhubarb cues.

#### Compiler-v9 real-audio continuity and contact audit

Retained job `01kxvby11g6gg7qb978njn87t0` recompiles the seven-second real
speech input with the dense calibrated retarget, explicit anger at 0.65, and
compiler v9. The exact face-local mouth-step maximum remains
0.03900 and false-silence motion p95 fell from 0.23354 to 0.06559; both
automatic hygiene checks now pass. The track remains active on speech, returns
to neutral in one frame, stays finite, reconstructs to the exported GLB with
0.033 mm mesh p95 error, and plays from its normalized audio clock in the live
viewer. The GLB has 18,437 render vertices, 35,324 triangles, one animation,
and zero validator errors or warnings.

This is a verified improvement, not production approval. The emergency
continuity guard still intervenes on 32/211 frames, but compiler v9 restores
the one contact that v8 reopened by moving its approach into the prior frame.
All three inferred targets are now attained, with one explicitly recorded as
continuity-restored. There are still zero independent phone/contact annotations,
so the strict gate fails seven content/timing checks and `production_validated`
remains false.

### Phase 3 — real and synthetic image fitting

Twelve seeded GNM recovery trials used K=20, yaw -18/0/+18 degrees, and 0.5 px
Gaussian noise:

| Metric | Result | Gate |
|---|---:|---:|
| Median landmark NME | 0.0039225 | <= 0.015 |
| Median coefficient cosine | 0.85467 | >= 0.75 |
| Median visible-vertex mean error | 1.23085 mm | <= 1.5 mm |
| Median visible-vertex p95 error | 2.86937 mm | <= 3.0 mm |

Real-photo results:

| Input | NME | Stability RMS | Bound fraction | Confidence |
|---|---:|---:|---:|---|
| scikit-image astronaut | 0.05395 | 0.26153 | 0.15 | medium |
| official Barack Obama portrait | 0.05718 | 0.03978 | 0.20 | medium |

The retained official-portrait job is `01kxs6qxd3ndm48rdq01c3pykf`. Its overlay
was inspected at source resolution: the correspondence is not left/right
inverted, all mapped points are in bounds, and there is no gross contour drift.
The mesh is deliberately neutral; expression, teeth/eye identity, tongue, and
pupil blocks that are not observable in this fit remain zero.

Real blank, duplicate-face, 60-degree rotated, tiny, and cropped images exercise
typed `FACE_NOT_FOUND`, `MULTIPLE_FACES`, or `FIT_REJECTED` paths.

### Phase 4 — application integration

- FastAPI, CLI, and browser call the same `ApplicationService` and pipeline
  functions.
- The real eight-second file was run independently through HTTP and CLI.
  After removing job IDs, UTC times, and SHA values, result JSON bytes matched;
  every NPZ array matched exactly; ffprobe properties and first/middle/last
  decoded-frame SHA-256 values matched.
- Real-image API/CLI fit metrics and every NPZ array matched exactly.
- Job recovery, upload bounds, typed errors, artifact allowlisting, health,
  media types, and original filenames pass.
- Live browser runs uploaded the official portrait and eight-second speech,
  displayed the resulting overlay/video and artifact links, and produced zero
  console errors after the favicon fix. Evidence is in `output/playwright/`.

### Phase 5 — final hardening

- Reproducible bootstrap and checksum-verified fixture downloader.
- Clean-cache Rhubarb processing tested with the real speech fixture.
- Full application and upstream regressions rerun after the final mux fix.
- Post-fix real jobs retained under `artifacts/verified/` with manifests and
  content hashes.

### Expanded video-performance verification

The opt-in CREMA-D `1001_DFA_ANG_XX.flv` moving-actor fixture was fetched from
official revision `1658cd342dff90010aa843eaeebd53610a08b1dc` and matched
SHA-256 `10dc3fd1f2bc8203657431598bd7dc9312462008f93d08fda786043ae6a8d2f4`.
`docs/TEST_FIXTURES.md` records attribution and license/rights caveats.

The current native MediaPipe video service path produced retained job
`01kxve1hnqqa48xyn6g0xyz0zj` with:

- 67 source frames and 67 detections; performance source PTS and timestamps are
  bit-exact with the native capture (source PTS 27 through 2229);
- dense geometry-calibrated retargeting matching 51/52 MediaPipe channels,
  with 50 active expression channels after quarantining `mouthClose`, and
  calibration hash
  `f7842b3ef9340e6215b4557d8cf87da200b20b04e7913ef7662ce7d36f59e4fe`;
- exact high-confidence blink/contact filter passthrough before neutral calibration and
  93.87% retained temporal variation across the remaining source controls;
- final GNM expression-motion correlation 0.8875 with all three detected
  high-motion expression events retained;
- no strong source blink event, so blink-event retention is correctly reported
  as null/unmeasurable rather than as a perfect score;
- the old control-proxy contact score was -0.2424 because `mouthRollUpper`
  falsely initiated an event at 0.467 s while the lips were geometrically open
  at 0.1192 interocular distance. The v2 performance schema instead measures
  three inner-lip distances directly from tracked image geometry. It finds one
  post-release closure event near 0.566 s and reports positive source/final
  closure correlation 0.4067;
- the released Claire `mouthClose` row is quarantined because it opens this GNM
  character. The geometry event now drives the shared spatial contact solve:
  at frame 17 the source gap is 0.00786, requested GNM target is 0.0030748, and
  final GNM gap is 0.0030726 versus neutral 0.04183. All 14 high-confidence
  contact frames attain their target, and no false timing warning remains;
- initial neutral-reference frames 0 through 6 scored 0.7357 with semantic peak
  0.5517; correction was applied, but the reference remains unvalidated because
  high `browDownLeft`/`browDownRight` values may be tracker bias or held expression;
- 41.97% of non-gaze residual samples fall below that one-sided reference and
  are clipped. This loss is now measured and warned explicitly rather than
  being hidden under a generic tracking-quality score;
- 4.43° head and 4.98° baseline-relative gaze-joint excursions, proving those
  tracks are not static on this fixture;
- 67-frame H.264/AAC browser proxy whose first video PTS is exactly media time
  zero, with 1.233 ms maximum inter-frame timestamp error;
- animated GLB reconstruction rank 12, 0.067 mm mesh p95 and 0.116 mm landmark
  p95 across all frames;
- Khronos glTF Validator: 0 errors, 0 warnings (one informational unused UV
  accessor because this performance GLB has no texture).

Focused post-change command:

```bash
PYTHONPATH=.:src uv run pytest -q \
  tests/test_video_capture.py tests/test_video_pipeline.py tests/test_viewer.py
```

Result: **17 passed**. The HTTP test also resolves the allowlisted viewer and
asserts a video media element, source-proxy clock, exact `AnimationMixer` time,
and end-of-clip clamping. Browser playback QA remains in the final integrated
application pass.

This is implementation evidence, not production approval. CREMA-D provides no
per-frame FACS, phoneme, gaze, head-pose, lip-contact, or GNM ground truth, and
the 52 tracker outputs cannot establish subtle microexpression fidelity. The
fresh native run closes the observed character-space seal defect, but its
source contact is still a landmark heuristic rather than independent phone or
lip-contact annotation. It contains no strong blink event with which to
validate blink retention and still relies on a heuristic rather than labeled
neutral reference. Monocular RGB also remains
unable to recover subject-calibrated depth, tongue behavior, or occluded facial
motion. The fixed path uses gaze only through baseline-relative eye joints,
activates no region bound on the retained clip, preserves 93.87% of source
temporal variation, measures 41.97% one-sided baseline loss, and reports these
unresolved production blockers rather than treating transport success as
animation approval.

## Acceptance ledger

| # | Result | Direct evidence |
|---:|:---:|---|
| 1 | PASS | Real audio creates cues, 383-D controls, moving real GNM geometry, audible MP4 |
| 2 | PASS | 100% cue coverage; offsets 0.000 and 0.876 frame |
| 3 | PASS | Rig block tests; eye/pupil blocks remain exactly zero for speech |
| 4 | PASS | Manual emotion composition test preserves the lower-face viseme delta |
| 5 | PASS | Two real photos produce nonzero identity, neutral OBJ/NPZ, overlay, confidence |
| 6 | PASS | Compact/official landmarks agree to `1e-6` |
| 7 | PASS | Twelve-trial recovery metrics all beat thresholds |
| 8 | PASS | Exact audio/image/emotion caveat strings asserted and retained |
| 9 | PASS | Real API/CLI JSON, NPZ, ffprobe, and decoded-frame parity |
| 10 | PASS | Native dependency and invalid/zero/multiple/extreme input errors are typed |
| 11 | PASS | Final E2Es use downloaded human speech and real photos, not mocks |
| 12 | PASS | 148 application tests plus native Swift and upstream GNM regressions pass after the final source change |

## Defects found and fixed during strict loops

1. A basic autocorrelation pitch estimate octave-doubled some voices. It was
   replaced with a bounded YIN-style detector and rerun on neutral and angry
   speech.
2. A common but incorrect MediaPipe right-jaw mapping caused an orientation/
   residual defect. The mapping was corrected and real-photo NME dropped to the
   retained values above.
3. Fresh setup copied only the Rhubarb executable, omitting PocketSphinx data.
   Bootstrap now installs and validates the complete signed/checksummed bundle;
   runtime health also rejects incomplete bundles immediately.
4. Visemes A-G leaked decoder values into GNM's tongue block. The mask now
   restricts them to lower face; only H may drive tongue.
5. FFmpeg `-shortest` with stream-copied H.264 silently dropped four final
   RAVDESS frames while the container-duration metric still passed. Audio is
   now padded to the video track, and validation asserts the actual video frame
   count plus video-stream duration.
6. Learned audio allowed 0.047-0.060 raw mouth steps while its production
   evaluator rejected exact face-local steps above 0.040. Compiler v8 now uses
   the evaluator's geometry at 0.039; a quality-only two-frame speech hangover
   prevents isolated VAD false negatives from being mislabeled as true silence.
7. Three.js 0.183.2 imports the sibling `three.core.js`, which was initially
   absent from the checksum-pinned viewer bundle. Bootstrap, health, allowlist,
   and tests now include that transitive module.
8. The viewer's strict CSP allowed `img-src blob:` but not `connect-src blob:`.
   Three's `ImageBitmapLoader` fetches embedded GLB images through a blob URL,
   so textured GLBs silently rendered gray. The narrowly scoped policy fix and
   regression assertion restore embedded textures; a clean live textured
   viewer now has zero console errors.
9. Compiler v8 formed a valid spatial lip contact before temporal projection,
   then the contact-oblivious forward limiter reopened one seal. Compiler v9
   retains feasible contacts as hard local anchors and redistributes only the
   bounded approach/release neighborhood. The retained real job attains all
   3/3 inferred contacts; deliberately infeasible anchors remain rejected and
   reported rather than weakening the continuity contract.

## Known limitations and viable upgrades

- **Lipsync accuracy:** Audio2Face is now the preferred motion generator and
  materially improves continuity, but its output has not been scored against
  an independently hand-labeled phone/contact corpus. Rhubarb A-H/X remains a
  coarse diagnostic/fallback, never ground truth. Production approval still
  requires annotated closure/contact timing and blinded animator review.
- **Retarget calibration:** Claire's controls now use a dense geometry-calibrated
  ARKit/tongue solve with an explicit raw-jaw observation and spatially local
  lip-contact correction. GNM still has no jaw joint, physical collision, or
  artist-approved speech-contact targets. Character-specific jaw, dental,
  tongue, lip-seal, asymmetry, and gain calibration remain required.
- **Learned runtime/license:** the macOS runtime is a new third-party Swift/MLX
  port. NVIDIA's model weights use the NVIDIA Open Model License; redistribution
  and product notices require legal review. The official NVIDIA runtime still
  requires CUDA/TensorRT on Windows or Linux.
- **Emotion accuracy:** the deterministic acoustic/lexical heuristic correctly
  labeled the retained RAVDESS anger sample but remains unvalidated and always
  reports confidence below 0.65. Replace it with a licensed, benchmarked SER
  model; use an LLM only for semantic expression intent, never lip timing.
- **Photo likeness:** one image constrains visible 2D geometry, not back-of-head
  shape, depth, texture, hair, or metric identity. The calibrated five-view
  synthetic positive control proves the shared fitter/texture/export core, but
  no small, commercially compatible calibrated real-person fixture was found.
  A guided real capture or a legally reviewed DECA/EMOCA/MICA-style initializer
  mapped into GNM remains the viable accuracy path.
- **RAVDESS licensing:** the optional emotional fixture is CC BY-NC-SA and is
  downloaded only with `AUTOANIM_FETCH_RAVDESS=1`; it is test evidence, not an
  application asset for commercial redistribution.
- **Upstream pyrender on macOS:** Google's nested `gnm_pyrender_test.py` and
  `render_gnm_test.py` are excluded by its top-level discovery and cannot import
  here because `gnm_pyrender.py` hard-codes `PYOPENGL_PLATFORM=osmesa`, while
  macOS has no OSMesa library. Run those two modules on Linux with Mesa. AutoAnim
  does not use that code path; its deterministic OpenCV renderer is covered by
  real-mesh and video E2Es.
- **Form factor/performance:** this release is local-first, single-process, and
  CPU-oriented. A job queue, GPU renderer, auth/storage, and deployment work are
  separate product phases.

## Reproduction

```bash
scripts/bootstrap.sh
scripts/bootstrap_a2f.sh
source .venv/bin/activate
export RHUBARB_BIN="$PWD/.cache/autoanim_gnm/rhubarb/rhubarb"
AUTOANIM_FETCH_RAVDESS=1 scripts/fetch_test_fixtures.sh
pytest -q

autoanim-gnm audio \
  .cache/autoanim_gnm/fixtures/libri-human-speech-8s.wav \
  --out artifacts/jobs --backend auto
autoanim-gnm image \
  .cache/autoanim_gnm/fixtures/official-portrait.jpg \
  --out artifacts/jobs
autoanim-gnm serve --host 127.0.0.1 --port 8000 --artifacts artifacts/jobs
```
