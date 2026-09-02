# Codex review of the ladder execution brief — 2026-09-02, xhigh reasoning

Model: OpenAI Codex CLI 0.149.0, `model_reasoning_effort=xhigh`, read-only sandbox, repository access. Brief: the draft in `docs/LADDER_EXECUTION_PLAN.md` §1's "draft". Verbatim answer, thinking traces and tool calls removed. Session id in `.context/codex-session-id`.

I’m using the engineering-plan review skill because this is an execution-order and gate-validity audit. I’ll restrict repository inspection to the files you named and will not touch the excluded Claude/agent paths.
The skill’s interactive prompts and artifact writes conflict with your explicit “answer the six questions” and repository-only constraints, so I’m using only its review rubric. I’m now tracing the named reports, gates, fitter assumptions, and converter code.
The first hard contradiction is already clear: the proposed hand gate resurrects a comparison the repository explicitly invalidates, and the proposed shape step sits behind known converter/facing defects plus an unresolved identifiability choice. I’m checking exact line-level evidence before ranking.
## 1. Rank and ordering errors

The current order breaks attribution.

- **G cannot be last.** The plan of record calls the owned fixture rank 0 because calibration ownership, absolute accuracy, and fitter identifiability all depend on it ([BODY_LANE_PLAN.md:32](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/BODY_LANE_PLAN.md:32)). Start the hardware work immediately, in parallel.
- **B must precede A.** A’s metric currently mixes body scale, analytic conversion, the clavicle-origin defect, facing, joint conventions, and ground projection. B prices the combined delivery stage before A changes it.
- **A is not “shape” as written.** Nineteen joints can fit MHR’s skeletal scale channels, not its surface identity. MHR’s 45 identity coefficients do not move joints. Calling this a body/mesh fit overclaims it.
- **A must be split.** First facing, then clavicle/root convention, then rest-skeleton schema, then MHR scaling. One report after changing all four attributes nothing.
- **C combines incompatible instruments.** Epipolar spread, residuals against MAMMA’s fitted projections, and MAMMA’s in-sample residual are different axes. Do not collapse them into one “bulk versus bias” verdict.
- **D is stale.** Current code already uses `ToeBase` to solve `LeftFoot`/`RightFoot`; only toe articulation remains inherited from the foot ([commercial_multiview.py:1588](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:1588)). Score the current delivery, not the old torso-welded implementation.
- **E uses a metric explicitly invalidated by the repository.** SMPL-X axis-angle spread and MHR Euler-channel spread are incomparable ([BATTLE1_INCREMENT5_ARTICULATION_VARIANCE.md:14](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/BATTLE1_INCREMENT5_ARTICULATION_VARIANCE.md:14)).
- **F is not a one-part substitution.** It changes detector, landmark cardinality, landmark semantics, target mapping, and MAMMA’s loss. It is uninterpretable without a matched sparse-MAMMA control.
- **“The front end reaches the reference” is too broad.** Only triangulation given MAMMA’s clean 2D is pinned. Association is pinned only with a perfect, count-correct detector; your own detector still lacks a discriminating report.

Execution should be two lanes:

```text
Hardware:     G fixture ───────────────────────────────→ marker/scan validation
Instrument:   B → C → corrected D/E reports → F only if still justified
Delivery:     B → facing → converter → skeleton schema → A
```

## 2. Strongest case against Step A being next

Step A cannot currently distinguish success from cheating.

The proposed score favors SMPL-X because the reference is MAMMA’s SMPL-X `pred_joints`. A model sharing its joint regressor and conventions has a structural advantage over MHR. “Within 10 mm of the SMPL-X arm” therefore measures convention convergence, not equivalent body recovery.

Worse:

- The canonical round-trip control is misstated. It is not 0 mm on the arms; the known converter defect is 36–47 mm ([BODY_LANE_PLAN.md:893](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/BODY_LANE_PLAN.md:893)). Requiring 0 makes the gate unreachable; excluding the arms makes it useless.
- Bone scale and locator offset are unidentifiable from the sparse landmarks. Free offsets leave the skeleton at the mean while scoring beautifully; zero offsets stretch bones to absorb convention error ([FITTER_PLAN.md:345](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/FITTER_PLAN.md:345)).
- Facing and clavicle-origin errors are in the same output stage. Improving A can merely compensate for them.
- Fifteen-joint scoring cannot support the user’s “skeleton with mesh” requirement.
- A 150-frame bootstrap treats correlated frames as independent. With one take and two people, its CI is not a generalization claim. Use sequence-block bootstrap, but even that only describes this take.

Do B first, repair the known delivery defects individually, establish a surface instrument, then build A with pinned locator offsets as an explicitly interim skeleton fit.

## 3. Instrument, blindness, band, and missing degenerates

| Step | Verdict |
|---|---|
| **A — MHR fit** | **Instrument:** same 15-joint axis is legal for reporting agreement, but SMPL-X is not an oracle for MHR and “within 10 mm” is not neutral. Primary gate should be exact synthetic 3D scale recovery; secondary real gate can report MAMMA agreement. **Blind:** surface/girth, absolute accuracy, landmark convention, temporal behavior, facing, skinning. **Band:** 10 mm is arbitrary and post hoc; not preregisterable without a measured MHR representation floor. **Missing degenerates:** global-scale-only fit, shoulder-only fit, subject swap, per-frame scale, offsets absorbing all proportions, half-take-specific skeleton, mirrored minimum, fitter inheriting raw outliers. |
| **B — MAMMA joints into converter** | **Instrument:** valid as a price for the entire delivery/retarget stage, not the converter alone. MAMMA’s shaped body entering a canonical rig mixes shape mismatch with conversion. Bypass or root-normalize ground projection. Verify the 15→19 adapter because the function requires all 19 finite inputs. **Blind:** surface, temporal, unmapped joints, absolute placement when root-relative. **Band:** no product band needed, but harness controls must be fixed. **Missing degenerates:** wrong joint permutation, left/right swap, copying input positions without valid rotations, root alignment hiding facing, unused joints filled with values that accidentally affect torso/head construction. |
| **C — detector reports** | **Instrument:** split into separate reports: epipolar incoherence; residual spread against MAMMA projections; coherent common mode. MAMMA’s own residual is deflated because its fit consumed those landmarks. **Blind:** common-mode calibration versus crop versus detector convention; absolute accuracy; depth. **Band:** only preregister a decision rule for the next experiment, not an accuracy claim. **Missing degenerates:** static camera offsets fitted and scored on the same frames; frozen/generic skeleton projections; subject-specific offsets; joint-semantic mismatch. Fit diagnostic offsets on one half and report the other half, while marking them non-shippable. |
| **D — feet** | **Instrument:** marginal axis or angle distributions are insufficient. Use frame-paired local foot orientation, phase/tracking agreement, and contact transitions on the same valid frames. Separate foot orientation, toe articulation, and contact; they are three outputs. **Blind:** MAMMA accuracy, sole deformation, surface penetration. **Band:** preregisterable only as parity/agreement, not truth. **Missing degenerates:** time-shuffled motion matching the distribution, left/right swap, mirrored forward axis, torso-welded moving foot, constant toe riding a solved foot. Include time-shuffled MAMMA as a control. |
| **E — hands** | **Instrument:** held-out camera is right for view generalization. Temporal gate must use wrist-local fingertip amplitude/jitter/roughness plus translation-only/world-space jitter, not angle SD. **Blind:** absolute 3D accuracy, contact, inherited wrist-anchor error. **Band:** the existing amplitude/jitter bands are preregistered; the draft’s “toward 5.6°” band is invalid. **Missing degenerates:** frozen rest hand, strong low-pass/lag, duplicated frames, prior-dominated solver that never leaves initialization. Warm-start is mandatory; the repository already demonstrated the zero-acceleration rest-pose trap ([BATTLE1_INCREMENT5_ARTICULATION_VARIANCE.md:308](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/BATTLE1_INCREMENT5_ARTICULATION_VARIANCE.md:308)). |
| **F — ours into MAMMA fitter** | **Instrument:** wrong as proposed. Add a control using MAMMA’s own observations reduced to the identical 17 targets and run through the identical modified loss. Otherwise sparsity and loss changes are charged to the detector. **Blind:** dense-surface information, model-prior domination, commercial relevance. **Band:** not preregisterable until the sparse MAMMA control establishes its floor. **Missing degenerates:** fitter ignoring observations, prior returning a plausible mean pose, mapping errors, shuffled detections producing nearly unchanged output. Require input-perturbation sensitivity and shuffled/time-constant controls. |
| **G — owned fixture** | **Instrument:** “checkerboard + flash” is insufficient unless validation is independent. Use held-out calibration views plus a moving wand/marker reconstruction, drift measurement, distortion residual by image region, and handedness/scale checks. **Blind:** soft tissue and anatomical marker definition until the marker protocol exists. **Band:** preregister from the motion/error budget, not from MAMMA. **Missing degenerates:** planar calibration overfit, zero-distortion model, index-based synchronization, copied MAMMA calibration, stationary sync target, correct reprojection with wrong metric scale. |

G only removes MAMMA calibration from the delivery path. It does **not** make any comparison containing MAMMA outputs non-instrumental.

## 4. Missing decomposition, surface instrument, and temporal measurement

### Missing decomposition

Step A currently conflates at least six components:

1. Landmark-to-joint convention.
2. Per-performer skeletal scales.
3. Per-frame pose.
4. Rest-skeleton serialization and consumer propagation.
5. Mesh skin/bind behavior on the new skeleton.
6. Surface identity, which the 19 joints cannot observe.

The schema dependency is substantial: `positions_to_body_track` constructs rotations against the fixed `DETAILED_HUMANOID` rest skeleton ([commercial_multiview.py:1493](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:1493)). A fitted MHR skeleton must propagate through FK, skin matrices, projection, export, sockets, and validation. Otherwise it silently serializes canonical names and downstream consumers reconstruct the canonical body.

### Surface/mesh instrument

Add three distinct instruments:

- **Skeleton truth:** exact synthetic MHR subjects with independently perturbed scale channels; score recovered rest joints and FK positions.
- **Skinning correctness:** render/FK consistency, finite vertices, bind-pose preservation, subject-specific skeleton propagation, and synthetic exact-mesh oracle.
- **Surface agreement:** held-out-camera silhouette boundary distance/IoU using commercially clean or manually owned masks; later, registered body-scan point-to-surface distance on held-out scan points.

Controls must include mean MHR, global-scale-only, shuffled subject mesh, inflated/deflated body, frozen pose, and a camera-facing billboard. Once held-out silhouettes are a gate, masks are no longer “parked because nothing consumes them.”

Do not use MAMMA `pred_vertices` as the shipping surface gate. Different topology, fitted model, and research provenance make it an agreement instrument only.

### Temporal is not unmeasurable

Only real-world absolute temporal accuracy is currently unavailable. Mechanism accuracy is measurable now.

Split rung 5:

- **5a recovery:** synthetic exact trajectories with deliberately injected one-ray slots, correlated camera outages, gaps, and known 3D truth.
- **5b smoothing/outlier rejection:** exact trajectories containing smooth motion, rapid real events, spikes, and gaps. Score 3D error, phase lag, peak attenuation, acceleration error, and recovery latency.

Controls: identity/no smoothing, frozen trajectory, over-smoothed trajectory, time-shifted truth, and an exact-truth oracle. The current synthetic fixture is inadequate because it produces zero-ray failures and never enters the sequence solve ([BODY_LANE_PLAN.md:1334](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/BODY_LANE_PLAN.md:1334)).

## 5. Research before code and sequencing

| Step | Research required first |
|---|---|
| **A** | Official MHR scale semantics, bounds and coupling; momentum parameter normalization; commercially clean regularizer source; locator/scale identifiability; per-performer skeleton schema; re-binding and export behavior. Decide pinned-offset interim versus waiting for markers. |
| **B** | Exact `PAIRS` semantics, root definition, coordinate handedness, unused 19-joint inputs, FK scoring, and ground-projection boundary. No external research is needed. |
| **C** | Detector landmark semantics, resolution/crop transforms, MAMMA in-sample-fit bias, and which figures have independent references. Define the decision rule before writing JSON. |
| **D** | SMPL-X ankle/foot/toe index semantics, angle periodicity, local foot-frame definition, and distinction between foot pose, toe articulation, and contact. Confirm current delivery state first. |
| **E** | Little new research. The repository already selected the parameterization-invariant metrics and exposed the warm-start failure. Define weight selection without MAMMA before coding. |
| **F** | Whether the modified fitter can run identical sparse target sets for both detectors; prior strength and observation sensitivity; cost of the Modal run. |
| **G** | Camera shutter/sync choice, calibration target coverage, independent wand protocol, marker/scan registration, drift tolerances, and performer/data-use releases. |

Parallel work:

- Start **G** immediately; it has external lead time.
- B, C, D, and E report generators can be implemented in parallel, but their `ladder.py` registration should have one owner to prevent registry conflicts.
- A’s external/API research can run alongside G. Its build must wait for B, the facing/converter split, and the skeleton schema.
- Within A, execute strictly: instrument freeze → facing → converter → schema propagation → MHR scaling → mesh binding.
- D’s bar must precede any foot tuning.
- E’s report must precede any temporal change; change one prior/initialization rule per run.
- F waits for C and for a matched sparse-MAMMA control design.
- Absolute surface and temporal claims wait for G’s marker/scan capture.

## 6. Reference leakage into shipped constants

There is already an apparent violation: `THORAX_SMOOTHING_FRAMES = 15` is in the delivery source and the comment says it was “chosen by the oracle arm alone” ([commercial_multiview.py:1675](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:1675)). Its caller is the shipped head solve ([commercial_multiview.py:1854](/Users/abhi_macbook/Projects/apps/AutoAnim/src/autoanim_gnm/commercial_multiview.py:1854)). Replace or justify it from synthetic/owned data; do not grandfather it.

Other leak paths:

- Tuning MHR regularization, locator limits, scale bounds, pose weights, or stopping criteria until the MAMMA 15-joint score improves.
- Selecting `X = 10 mm` after seeing the SMPL-X arm, then iterating A against it.
- Shipping the static per-camera offsets fitted against MAMMA projections.
- Choosing facing, clavicle-anchor, foot-axis, contact, or hand-smoothing constants from MAMMA agreement.
- Selecting `smooth_weight` to approach MAMMA’s amplitude/jitter. Pick it using synthetic truth or commercially clean held-out data; run MAMMA once after freezing.
- Using the SAM 3D 28-dimensional scale PCA. Its checkpoint is not covered by MHR’s Apache licence ([FITTER_PLAN.md:218](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/FITTER_PLAN.md:218)).
- Training the pseudo-label detector on this fixture. “Our triangulated labels” still depend on MAMMA’s `ma_cap` calibration, so MAMMA-derived data enters the weights.
- Letting F’s MAMMA experiments determine shipping loss weights or priors.
- Copying MAMMA-derived foot ranges, hand pose statistics, or SMPL-X betas into MHR initialization/defaults.

Required containment:

- Freeze the delivery commit and configuration hash before any MAMMA evaluation.
- Keep reference reports/configuration physically separate from delivery configuration.
- Record provenance for every runtime constant, weight, calibration, and training example.
- Reject builds whose training manifest contains `artifacts/mamma`, `ma_cap`, SMPL-X outputs, MAMMA-derived offsets, or reports computed from them.
- Use MAMMA only for the final benchmark report; tune on synthetic truth, owned calibration, owned annotations, markers, and scans.

## Final ranked step list

1. **Start G: owned fixture and marker/scan protocol** — rank 0 dependency with real lead time; it unlocks accuracy, calibration ownership, and fitter identifiability.
2. **Run B: clean-input delivery pricing** — establishes the pre-A cost of the whole retarget stage and exposes mapping/root mistakes cheaply.
3. **Split and fix facing, clavicle/root convention, and ground projection** — known delivery defects must be isolated before body scaling can be attributed.
4. **Add per-performer rest-skeleton schema and propagation gate** — without it, an MHR fit silently becomes canonical again downstream.
5. **Land C as separate detector instruments** — makes the detector decision auditable without mixing incoherence, fitted-reference residuals, and common mode.
6. **Add synthetic skeleton, mesh-binding, surface, and temporal instruments** — closes the largest measurement holes before more optimization.
7. **Correct D and score the foot implementation that actually ships** — the plan’s “foot welded to torso” premise is stale.
8. **Correct E and register fingertip amplitude/jitter plus held-out error** — angle SD is invalid; the existing gauge-invariant protocol is the usable gate.
9. **Build A as an interim pinned-locator MHR skeleton fit** — valuable after prerequisites, but it is skeletal scaling, not dense surface recovery.
10. **Run F only with an identical sparse-MAMMA control** — otherwise the modified fitter experiment cannot attribute anything to the detector.

tokens used: 1711979
