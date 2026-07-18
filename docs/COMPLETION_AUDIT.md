# AutoAnim expanded-objective completion audit

Status: active; completion is not proven
Audit date: 2026-07-18
GNM revision: `3de70dfca5f3244620f44103c24b7cedc0dcb8b6`

This file maps every requested outcome to the evidence that would actually
prove it. Passing a unit test, producing a plausible preview, or failing to
notice a defect is not sufficient evidence for a broad production claim.

## Requirement ledger

| Requested outcome | Required proof | Current authoritative evidence | Audit state |
|---|---|---|---|
| Deep repository-grounded GNM research | Code- and asset-grounded architecture/data-model analysis, with primary external sources where needed | `RESEARCH.md`, `FEASIBILITY.md`, `PRODUCTION_LIPSYNC_RESEARCH.md`, `AUDIO_PRODUCTION_RESEARCH.md`, `EXPANDED_RESEARCH_AND_PLAN.md`, and `VIEWER_RESEARCH.md`; claims reconciled against pinned GNM `3de70dfc...` and current artifacts | **Achieved for research deliverable** |
| Detailed phased application plan | Concrete form-factor decision, contracts, milestones, dependencies, tests, and stop/go gates | Local-first web workspace justified in `SPEC.md`; phases 0–8, milestones, dependencies, tests, and production stop/go gates in `EXPANDED_RESEARCH_AND_PLAN.md` | **Achieved for planning deliverable** |
| Audio in → smooth lips and expressions out | Real learned inference, synchronized GNM motion, finite geometry, emotion/prosody layering, contact-aware dynamics, viewer/media playback | Retained compiler-v9 job `01kxvby11g6gg7qb978njn87t0`: learned dense retarget, exact media clock, 0.03900 mouth max, 0.06559 false-silence p95, 3/3 inferred contacts attained, finite GLB, live playback | Working improved prototype; 32/211 bounded interventions remain, while independent phonetic/contact and human-quality evidence is absent |
| Production-quality audio claim | Independent held-out phone/contact annotations, character seal/jaw calibration, collision checks, approved dynamics distribution, blinded animator/naive MOS, language/noise slices, legal approval | The evaluator fails closed without annotations; no approved human panel or capture-ground-truth GNM performance exists | **Not achieved** |
| Single photo → matching 3D face | Real photo E2E, correct camera convention, visible-landmark fit, confidence/uncertainty, neutral GLB/viewer | Real astronaut and official-portrait jobs plus synthetic recovery and adversarial cases | Conservative visible-geometry fit works; perfect-person claim contradicted by monocular ambiguity |
| Multiple photos → best-accuracy shared identity | Legitimate calibrated real multiview subject, shared bounded identity solve, mixed-person rejection, held-out geometry improvement over single view, camera/occlusion audit | New sidecar/shared-registration branch has five-fit/two-held-out synthetic E2E, leakage, nonzero-distortion, matrix-provenance, accepted-set-stability, and effective-evidence-rank regressions. The older retained five-view artifact predates this branch and proves only the legacy synthetic fitter/texture core. No small ready-download commercial fixture was found. | **Incomplete**: calibrated core is synthetic-tested but no rights-cleared calibrated real-person ground truth exists. Viable routes are a consented local capture or a contractually licensed dataset subset with measured K/D/RT and scan reference. |
| Person texture on the GNM model | Real multiview texture bake, seam-correct GLB, component-wise direct/fill/generic provenance, coverage/seam review | Synthetic positive control: 53.051% observed, 24.885% inpainted, 22.064% generic; per-component provenance; live textured viewer; valid embedded-texture GLB | Core achieved; real-person likeness, de-lighting/intrinsic albedo, BRDF, hidden anatomy, and artist approval remain incomplete |
| Video → expressions, microexpressions, lips, gaze, and head | Real moving-human E2E at source PTS, missing-frame behavior, dense retarget, exact browser media synchronization | True-native CREMA-D job `01kxve1hnqqa48xyn6g0xyz0zj`: 67/67 detections, bit-exact capture timestamps, 0.8875 expression correlation/3 events, geometry-derived closure, 14/14 high-confidence contact frames at the calibrated seal, live synchronized viewer | Working experimental pipeline; the false control-proxy contact event is removed, but source contact remains an unlabeled landmark heuristic, 41.97% of one-sided neutral residuals are clipped, and no strong blink exists to evaluate |
| Production microexpression claim | Per-frame FACS or dense capture ground truth, occlusion/head-pose slices, independent perceptual study, subject-specific high-quality tier | CREMA-D has no per-frame FACS, gaze, contact, or dense face ground truth | **Not achieved** |
| Unified 3D viewer | Static, textured, audio-animated, and video-animated GLBs; orbit/zoom/topology/texture; exact media time; local runtime; error/accessibility/lifecycle handling | Live final browser pass covers all four asset types; embedded textures fixed by allowing Three's blob fetch; local checksum-pinned bundle, CSP, cleanup, errors, and media clock pass | Desktop foundation achieved; mobile matrix, formal accessibility, WebGL context-loss recovery, and memory/load tests remain incomplete |
| CLI/API/browser consistency | Same pipeline/service, matching artifacts/warnings for same inputs, job recovery and allowlists | Shared `ApplicationService`, parity/typed-error/recovery/allowlist tests, retained jobs, post-change live dashboard/viewers | **Achieved for implemented prototype paths** |
| Full post-change verification | Complete app suite, Google upstream suite, Swift suite, Khronos validator for every retained final GLB, real-input E2E, browser QA after final source change | 164 app tests pass; the one explicit Claire-environment skip also passes separately with its asset variable; 278 official GNM; 60 nested fitting; 21 camera/color pass + 3 skip; 7 Swift; four prior retained GLBs 0 validator errors/warnings; dashboard plus four live viewer paths | **Achieved for current source and retained prototype artifacts; new calibrated branch remains synthetic-test evidence rather than a retained real-person artifact** |
| Production release readiness | Approved dependencies/model terms/notices, consent/deletion/audit workflow, licensed validation corpora, human sign-off, operational load/recovery/canary evidence | Three.js license and Claire model card/provenance are now retained; broader product/data/legal/operations approval is absent | **Not achieved** |

## Hard truth about “perfect” and “flawless”

A single unconstrained RGB photo cannot uniquely determine unseen skull, ears,
rear head, teeth, tongue, hair, metric scale, camera intrinsics, or illumination-
free skin appearance. Multiple calibrated views reduce ambiguity but do not make
ordinary photographs equivalent to structured-light/photogrammetry capture.
Likewise, audio alone does not uniquely determine gaze, blinks, head gestures,
or the intended acting performance, and monocular video does not expose all
subtle muscle or contact states.

The viable production alternatives are:

1. a guided calibrated multiview capture with cross-polarized lighting and a
   scale target, followed by dense photometric/depth refinement and artist QA;
2. a subject/character calibration profile with neutral, lip-seal, jaw, dental,
   tongue, eye, and extreme-expression captures;
3. independently annotated held-out speech/video corpora and blinded human
   review;
4. an editable animator layer rather than a promise that audio or one camera
   uniquely recovers the intended performance.

Until the rows marked incomplete or not achieved have direct evidence, the
correct product label is **working, instrumented research prototype**, not
production-approved digital-human cloning.

## Final retained artifact ledger

| Path | What it proves | What it does not prove |
|---|---|---|
| `artifacts/jobs/01kxvby11g6gg7qb978njn87t0` | Real audio through learned inference, compiler-v9 contact-anchor continuity, 3/3 inferred contacts attained, finite synchronized GLB/viewer | Independent phone/contact timing, artist quality, physical collision, broad language/noise robustness |
| `artifacts/jobs/01kxv10qk3wscjr13shq2yp916` | Real single-photo detector/fitter/export/viewer path | Hidden 3D anatomy, metric or perceptual identity |
| `artifacts/verification/synthetic-calibrated-multiview-v1` | Older synthetic five-view identity/texture solve and standards-valid textured GLB | The new sidecar/shared-registration/held-out branch, real-person detector ingress, capture robustness, consented likeness accuracy |
| `artifacts/jobs/01kxv9hr5hf9mx2n7ggbnhky68` | Live viewer rendering of that explicitly labeled synthetic textured positive control | A real multiview reconstruction |
| `artifacts/jobs/01kxve1hnqqa48xyn6g0xyz0zj` | True-native real video detection, source timestamps, dense retarget, geometry-derived GNM contact attainment, proxy/viewer synchronization | Independently labeled blink/contact/microexpression accuracy, bidirectional subject calibration, or subject-specific depth |
