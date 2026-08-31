# The fitter — plan of record for the largest structural gap

**Snapshot 2026-08-31.** Written after the user asked why MAMMA's mesh overlays live
action and ours does not. The answer was architectural: **MAMMA fits a body model, we
fit nothing.** This is the plan to close that, with the bands it must pass.

Tags: `[T]` derived from this repo's own measurements · `[R]` recalled, unverified ·
`[D]` a document that must be read.

---

## 0. The reframe: it is a sequence, not a fork

The choice was posed as *build an MHR fitter* **or** *take the HumanIK route*. It is
neither-or; they are stages of one build.

| stage | what it is | status |
|---|---|---|
| 1. schema + gate | mandatory `rest_translations_m` on `BodyTrack`, three-layer gate, the `0.98` hardcode | **designed** (§Step 0d of the plan), days |
| 2. **MVP skeleton fitter** | per-take skeleton + per-frame pose, jointly optimised against landmarks | **the new build** |
| 3. HIK characterisation | `HIKSkeletonGeneratorNode` fed the fitter's skeleton instead of SMPL-X's | consumes stage 2 |
| 4. dense shape | girth, silhouette, mass | waits for the Battle 2 scan |

Stage 1 is the fitter's **delivery mechanism**; stage 3 is its **export target**.
Nothing in the HIK route is throwaway — without a fitter it is a skeleton with no
optimised pose; without stage 1 the fitter has nowhere to put its result.

**⚠ "And the mirror dissolves" — REFUTED before build.** The first draft claimed a
fitted skeleton built from measured positions is correct-handed by construction, so the
mirror never needs fixing. It is false **in the delivery path**, by two mechanisms:
(i) Step 0d's legality constraint stamps offsets as non-negative multiples of the
**mirrored** canonical directions, so a correct-handed fitted skeleton is
*inexpressible* in the very schema stage 1 builds; and (ii) the delivered mesh is
skinned **name-keyed to the mirrored rig**, so driving it with correct-handed data
reproduces the documented pure-180° yaw. The fitter makes the *skeleton*
correct-handed; the *mesh asset* stays mirrored. **The mirror fix is a prerequisite or
an explicit interaction — not dissolved**, and the parent plan's ordering (gate → mirror
→ proportions) stands.
*Unreconciled, and it must be before the schema hardens:* for multi-segment chains the
direction-preserving constraint admits legal-but-grotesque solutions — shrink
`LeftUpperArm`'s pure-X offset to ~7 mm and the 540→355 shoulder width "fixes" while
corrupting clavicle geometry for skinning and for HIK. Single-offset joints like the hip
(90 lateral / 80 down) are genuinely direction-locked. **The constraint and the fitter's
job are not yet reconciled.**

---

## 1. What 19 landmarks can and cannot fit — the question that decides feasibility

`[T]` **They admit a skeleton, not a shape.** MHR carries 45 identity parameters and
68 scale channels; 19 joint centres give roughly the 13 `RIGID_LIMBS` distances
`estimate_limb_lengths_m` already computes, aggregated over a take — plus, from pose
variation across the take, how the non-rigid pseudo-bones vary and some components of
constant landmark-to-joint offsets. Order tens of numbers, all skeleton-adjacent,
never surface. That is not a
shape estimate — it is a **skeleton** estimate, which is exactly where every defect
chased on 2026-08-30/31 lives (shoulder width 540 vs 355, per-performer sizing, joint
placement).

**MAMMA needs 512 dense surface landmarks *because* it solves dense shape.** We have
neither the landmarks nor — per §10's open product question — a demonstrated need for
dense shape before the body scan exists.

**So the MVP is:** fit the **skeleton-affecting** parameters to take-aggregated
landmark constraints, hold dense shape at the model mean, and solve **pose per frame**
against the same objective.
*What it will fix:* limb lengths, joint placement, stance width, stature, and the
facing — all by optimisation rather than hand-correction.
*What it will not fix:* girth, mass distribution, silhouette. **Sizing fixes
proportions; the scan fixes the body.**

`[W]` **VERIFIED by running the shipped asset, and the result is stronger than the
claim.** Perturbing each of MHR's 45 `identity_coeffs` to 3σ moves vertices by up to
**9.65 cm** and moves joints by **exactly 0.000000 cm** — at zero pose and on a posed,
scale-perturbed body. **Surface shape lies in the null space of the joint-centre
likelihood by construction**, so 19 joint centres carry *zero* information about
identity rather than merely too little. The 68 scale channels move joints directly.
The MVP is not a workaround; it is MHR's **native factorisation.**

---

## 2. The band — against truth, never against looks

**Primary instrument: the synthetic fixture.** Its documented blindness is to the
*sequence solve*, which never fires there; **a fitter touches every slot**, so the
fixture prices it against exact 3D truth. Same argument as the constrained
re-estimator entry.

**⚠ THE FIRST DRAFT'S BAND WAS REFUTED BEFORE BUILD — twice over, and the failure is
instructive.** It read: *"must beat 105.4 / 120.8 mm by ~44 mm, the amount oracle
substitution recovered."* Re-running `retarget_cost.py`: oracle substitution recovers
**16.4 mm** and **34.6 mm** on the mandated pooled mean. **Nothing measured recovers
44.** That number traces either to the **withdrawn pooled-median statistic** this
plan's own ninth-instance warning exists to prevent, or to arm C's median *residual*
coincidentally reading 44.0/44.8. And the **denominator was wrong**: 105.4/120.8 is
delivered-vs-*captured* on real footage, while the band applies it to
FK-vs-*truth* on the synthetic fixture — the same-denominator rule broken inside the
plan's own pre-registration. The resulting ~61 mm threshold is also far **too easy**:
the fixture's own observations sit ~12 mm from truth, so a fitter scoring 61 would be
5× worse than the points it consumed and still pass.

**Replacement, to be measured before it is fixed:**
1. Measure **arm-B-on-fixture** — canonical rig plus analytic converter against fixture
   truth. Never measured. The fixture's two subjects already carry non-canonical,
   mutually different rest skeletons, so this is a real baseline.
2. The fitter's FK-vs-truth must land near **the fixture's observation noise plus a
   measured representation floor** — not merely beat the old pipeline.
3. Pin one pixel scale before pre-registering the secondary band: 11.7 px was measured
   at 800×450, the ~4–5 px spread at 1280.

**Secondary, on real footage:** delivered-vs-point-skeleton median must fall from
**11.7 px** toward the point skeleton's own reprojection spread (~4–5 px per §8).

**Five degenerate solutions the band must reject** — the first draft named two:
1. **The mean-body regressor.** A fitter that ignores the observations and returns
   canonical would look entirely plausible. *Test:* feed two synthetic subjects with
   deliberately different proportions; the fitted skeletons must differ **in the right
   direction, by the injected amount**.
2. **The overlay-flatterer.** A fitter optimising reprojection can hide depth error
   along the camera ray. The instrument table bans reprojection as a gate for exactly
   this reason, so **the primary metric stays 3D-against-truth on the fixture — never
   the overlay**, however persuasive the overlay looks.
3. **The mirrored minimum — and it is the sharpest of the five.** A 180°-yawed pose on
   a near-symmetric skeleton fits 19 labelled landmarks at a residual close to the true
   minimum; there are no toe landmarks and the fore-aft asymmetries are small. **An
   FK-positions-vs-truth band plausibly cannot reject the exact defect the user found by
   eye.** Pre-register a facing check beside it — `camera_overlay.py` exists and
   currently reads dot −0.90.
4. **Pose-aliasing into per-take constants.** The non-rigid pseudo-bones (root→neck
   through the articulated spine, neck→shoulder through the clavicle) vary with pose, so
   a per-take "length" for them is a pose-distribution statistic. *Test:* fit the two
   halves of one take independently; the skeletons must agree within band.
5. **The outlier trap.** Trap 1 below says fit against **raw** triangulation — whose
   tail reaches 0.9–1.35 m and is currently absorbed by Savitzky-Golay as the pipeline's
   de facto outlier filter. An L2 fitter inherits that tail and can be **worse than
   today on real footage while acing the fixture**, whose noise is a tame mixture.
   Pre-register a robust loss, and settle smoother-vs-fitter ownership the way trap 3
   settles ground projection.

---

## 3. Three traps, named before writing the objective

1. **Do not fit per-frame bone lengths.** Lengths are per-take constants, solved once
   as the median over directly-triangulated frames — the estimator we already have.
   Per-frame lengths eat detector noise and pulse the skeleton, which is the failure
   `estimate_limb_lengths_m`'s own docstring warns about.
2. **Do not optimise against smoothed positions.** Already pre-registered for Step 4's
   labels, and the same reason applies: the fitter would learn the smoother's lag. Fit
   against **raw** triangulation with the confidence gating already validated.
3. **Do not let the fitter absorb the ground projection.** `project_generated_foot_contacts`
   runs afterwards and shifts the root ~90 mm; if the fitter also solves global
   placement the correction is applied twice. **Decide the ownership boundary — fitter
   owns pose and skeleton, projection owns ground contact — before writing the
   objective.**

---

## 3a. The solver already exists — the build was mispriced

`[W]` **`BODY_LANE_PLAN.md`'s "no published MHR fitting solver exists" is refuted in
the operative sense.** Meta's **momentum** (MIT licence) ships
`momentum/marker_tracking/marker_tracker.h` carrying `CalibrationConfig` — **per-bone
scale calibration** (a `globalScaleOnly` flag proves per-bone is the default),
**locator calibration** (directly the tool for the convention-offset problem), and
floor-contact constraints — plus `TrackingConfig` and, decisively, `CameraKeypointData`:
**multi-camera 2D keypoints with confidences and calibrated cameras, which is exactly
our data shape.** MHR's README declares PyMomentum integration, `compact_v6_1.model` is
a momentum parameterisation, and PyPI's `pymomentum-core` exposes `marker_tracking` and
`solver2` from Python.

**So stage 2 is repriced: from "write an optimiser" to "integrate one and own the
locator/convention layer."** What remains genuinely absent is a published
*image-to-MHR multi-view* system — but stage 2 as scoped here is a calibration and
tracking problem momentum was built for.
**`[W]` DE-RISK RUN 2026-08-31, and it passes.** `pymomentum-cpu` installs from PyPI
into a clean Python 3.12 venv on this machine and imports; `marker_tracking`,
`solver2`, `geometry`, `camera`, `skel_state` all load. The Python surface carries the
whole contract:

| exposed | what it answers |
|---|---|
| `CalibrationConfig.calib_shape`, `.global_scale_only` | per-bone scale calibration, not just a global one |
| `.locators_only` + `convert_locators_to_skinned_locators`, `get_locator_error` | **the convention-offset layer** — the landmark-vs-bone problem, as a first-class object |
| `.loss_alpha` on **both** configs | **a robust loss**, which is degenerate solution 5's required mitigation |
| `.adaptive_floor_contact`, `.enforce_floor_in_first_frame`, `.floor_contact_percentile` | trap 3's ground-projection ownership, settable rather than argued |
| `TrackingConfig.smoothing`, `.smoothing_weights` | trap 2's smoother boundary — can be switched off so the fitter never learns the lag |
| `CameraKeypointData`, `KeypointObservation`, `.projection_weight` | multi-camera 2D keypoints with confidences against calibrated cameras — our exact data shape |
| `calibrate_markers`, `process_markers`, `refine_motion` | the calibrate-then-track pipeline, already factored |

**Every trap and degenerate solution named by either advisor has a corresponding knob
in this API.** That is the strongest evidence yet that stage 2 is an integration rather
than a research build. *Still unverified:* that it accepts **MHR's** `compact_v6_1.model`
specifically, and that results on our data clear the bands in §2 — the API existing is
not the fitter working.

## 4. To verify before building

- `[R]` MHR's skeleton/shape decoupling and its actual parameter surface.
- `[R]` Whether `HIKSkeletonGeneratorNode` accepts arbitrary proportions without
  HumanIK's own retargeting fighting them.
**All four were checked. Results:**

- **`[W]` MHR's decoupling — CONFIRMED, and stronger than claimed.** Loading the
  shipped `mhr_model_lod6.pt` and perturbing each of the 45 `identity_coeffs` to 3σ
  moves vertices up to **9.65 cm** and moves joints by **exactly 0.000000 cm** — at
  zero pose *and* on a posed, scale-perturbed body. Surface shape sits in the **null
  space of the joint-centre likelihood by construction.** So 19 joint centres carry
  *zero* information about identity, not merely too little. The MVP is MHR's **native**
  factorisation. *(SMPLify's "2D joints carry surprising shape information" is about
  SMPL, whose joint regressor depends on betas — there, skeleton is part of shape and
  the rest is population prior. MHR factorises explicitly, so a joints-derived "shape"
  would be pure prior.)*
- **`[W]` HIK holds supplied proportions — CONFIRMED, scoped.** Worst placement error
  **8×10⁻⁶ cm** across 62 slots after `hikCharacterLock`. Scope: the tested case is an
  identity-proportioned source→target, which *is* stage 3's use. **It does not verify
  HIK retargeting across mismatched proportions** — do not quote it as that.
- **`[W]` MHR asset licence — CONFIRMED clean.** Release v1.0.1's `assets.zip` contains
  `assets/LICENSE.txt`, **stock Apache-2.0** (sha256 matches the canonical text),
  covering the mean mesh, corrective blendshapes, `mhr_model.pt` and
  `compact_v6_1.model`. No NOTICE, no rider. **Two hygiene items:** consume from the
  MHR release directly rather than via the SOMA third-party redistribution; and —
  **`⚠ licence trap`** — the convenient 28-d scale PCA (`scale_mean`, `scale_comps` in
  the vendored SAM 3D checkout) comes from **SAM 3D's checkpoint under the SAM licence,
  not the Apache MHR asset. Do not lift it.** Fit the 68 raw scale channels with our own
  regulariser, or re-derive a proportion prior.
- **`[T]` The ~44 mm band — REFUTED.** See §2.

**One caveat that survives everything:** the fitted skeleton is a *landmark* skeleton,
well-posed, but **biased as an anatomical one** — it inherits the detector's convention
offsets, measured at up to 33.9 mm per joint. **The synthetic fixture is structurally
blind to this**, listing joint-definition error as absent by construction, and it is the
term that dominated Battles 0 and 1 on real footage. Stamp detector identity with the
skeleton, and never read fixture success as evidence about convention.


---

## 5. Integration status — 2026-08-31, all gates cleared so far

Run end to end on this machine, in a throwaway venv so nothing was installed into the
project environment.

| gate | result |
|---|---|
| `pymomentum-cpu` installs and imports on macOS | **PASS** — `marker_tracking`, `solver2`, `geometry`, `camera`, `skel_state` all load |
| momentum loads **our** delivered GLB | **PASS** — 55 joints, correct names, mesh intact — but **0 model parameters**, since glTF carries no momentum parameter transform |
| MHR assets are Apache-2.0 | **PASS** — official release `assets.zip`, `LICENSE.txt` is stock Apache-2.0 |
| momentum loads MHR + its model definition | **PASS** — `lod6.fbx` gives 127 joints with mesh; `compact_v6_1.model` adds **204 model parameters: 68 scaling, 136 pose** |

**And the scale parameters are, name for name, the defects measured this week:**

| measured defect | MHR parameter |
|---|---|
| shoulders 540 mm vs 346–363 | `scale_shoulder_width` |
| torso 577 vs 513 between performers | `scale_spine_length`, `scale_neck_length` |
| upper arm −17/−27, forearm −18/−29 | `scale_uparms`, `scale_lowarms` |
| hips 180 vs 207–215 | `scale_hip_width`, `scale_hip_height`, `scale_hip_depth` |
| thigh +28/+30, shin +15/+24 | `scale_uplegs`, `scale_lowlegs` |
| fingers (no landmarks yet) | full per-digit length and offset scales |

**Nothing here required inventing a parameterisation.** The skeleton-affecting knobs
are named, separate from the 45 identity coefficients, and cover every measured error.

### What remains, and it is precisely the layer Fable said we would own

`calibrate_markers(character, identity, marker_data, calibration_config, …,
camera_keypoint_data=[])` returns `(identity, parameter_indices, motion)`. The
character arrives with **zero locators**, and MHR's joints are named `root`,
`l_upleg`, `l_lowleg`, `l_foot`, … — so the remaining work is the **locator layer**:
define named attachment points on MHR corresponding to our 19 landmarks, which is
also where the detector's convention offsets get modelled rather than baked. momentum
exposes `locators_only` calibration and `get_locator_error` for exactly this.

**Do not skip to a fit before that layer is designed** — a marker set matched to the
wrong joints will still converge, and the result will look plausible. That is
degenerate solution 1 wearing a different coat.


---

## 6. First calibration run — 2026-08-31. **It is worse than the mean body, and the reason was predicted.**

17 locators attached to MHR (our landmarks minus the ears SOMA-77 never emits),
**zero offsets**, raw triangulation as markers per the pre-registered trap, robust
loss on, per-bone scale enabled. `calibrate_markers` converged: 150 frames, identity
and motion returned.

| segment | MHR mean | **fitted** | measured (perf A) | mean err | **fit err** |
|---|---:|---:|---:|---:|---:|
| shoulder width | 351.7 | 322.4 | 346.4 | 5.3 | **24.0** |
| pelvis → neck | 518.0 | 501.3 | 576.6 | 58.6 | **75.3** |
| hip width | 164.1 | **193.2** | 207.1 | 43.0 | **13.9** ✓ |
| upper arm | 256.8 | **356.8** | 287.2 | 30.4 | **69.6** ✗ |
| forearm | 270.0 | 296.2 | 268.9 | 1.1 | **27.3** |
| thigh | 419.8 | 427.4 | 399.8 | 20.0 | 27.6 |
| shin | 420.6 | **398.6** | 396.5 | 24.1 | **2.1** ✓ |
| **mean abs error** | | | | **26.1** | **34.2** |

**The fit is 8 mm worse than doing nothing.** Two segments improved sharply — hip
width 43→14, shin 24→2 — and the arms and torso blew up. `scale_uparms` came back at
**+1.0005**, a suspiciously round number that reads as a parameter **saturating at its
bound** while trying to absorb something it cannot represent.

**The cause is the locator layer, and this plan warned about it in §5 before the run:**
every locator offset was left at **zero**, which asserts that our detected landmark
*is* the joint centre. It is not — these are surface landmarks carrying the detector's
convention offsets, measured elsewhere in this lane at up to **33.9 mm per joint**. The
fitter, given no way to express "the marker sits 30 mm lateral to the joint", did the
only thing it could and **stretched the bone instead.** That is precisely the failure
the warning named: *a marker set matched to the wrong joints will still converge, and
the result will look plausible.*

**What this establishes, and it is worth more than a passing number:**
1. The integration works end to end — character, locators, markers, calibration, motion.
2. **The band works.** A fit that looks entirely reasonable, converges cleanly, and
   moves two segments in exactly the right direction is still **worse than the mean**,
   and only measuring against the performer's own dimensions reveals it.
3. **The locator layer is not optional plumbing — it is the substance.** Solving
   landmark-to-joint offsets is the difference between a fitter and a bone-stretcher.

**Next, and pre-registered before running it:** a two-stage calibration — solve locator
offsets first (`locators_only`), then scales — and the acceptance band stays *beat the
26.1 mm mean-body error on this performer*, with the mean-body regressor and mirrored
minimum checks alongside. **Do not report a fit that fails to beat doing nothing.**
