# D4 — the body model in the delivery path

**Date** 2026-09-21/22 · **Branch** `ladder/D4` · **Worktree** `.claude/worktrees/ladder-D4`
**D4 ACCEPTANCE: FAIL.** The gate reads
`D4 ACCEPTANCE: FAIL (O1 1.030 mm > 1 mm; hygiene, reproduction, B1, B2, B1 frozen-pose control,
B1 MAMMA arm unchanged, O1 closure, O1 mean-body must-fail PASS)`. Astra's one merge round
(`docs/reviews/body-model-astra-merge-review-2026-09-22.md`) returned **NO MERGE at `e209eea`**
and was right on the rule: the earlier "O1 a recorded exception" was an override, which the lane
forbids. **The coordinator withdrew it the same day.** What merges is a separate and separately
derived thing — the **opt-in implementation**: `--body mhr` behind an unchanged `rig` default, so
nothing shipped changes, with D4's acceptance still open. O1 is re-registered prospectively as
**D4b**. MHR — Meta's open body model (Apache), fitted by momentum (MIT) to the delivered take's
own smoothed, repaired landmarks with locator offsets pinned — is the body that path delivers.
The card's rule is `hygiene AND the reproduction AND O1 AND B1 on both performers AND B2`, and
the gate now enforces every clause the card requires — nine of them, including the three Astra
showed were printed but not enforced. **Eight pass. O1 fails at 1.030 mm against a 1 mm band, so
acceptance fails**, and the failure is written as a failure everywhere. Why the band was missed is
attributed in §2.1 — it was set without measuring the instrument's floor, which is the lane's
recorded pre-registration error, again — but that is an explanation, not a licence: **the band is
not moved and nothing is re-selected.** A calibrated replacement is D4b's to register in advance.

The headline: the delivered body stopped being a stock MPFB mesh stretched over a scaled rig.
Silhouette IoU against the SAM2 masks goes **0.647 / 0.652 → 0.803 / 0.767**, paired lower CI
bounds **+0.133** and **+0.084**, with MAMMA's own mesh at 0.873 / 0.845 on the identical
rasteriser and masks (that pair is the median of its four camera medians — the committed report
carries no pooled figure — beside our pooled 0.803 / 0.767, which is stated wherever it appears) — bit-identical to its committed value on all eight cells, so the instrument
has not moved under the comparison.

---

## 0. What was done, in order

| stage | what | verdict | commit |
|---|---|---|---|
| 1 | hygiene — `--body rig` rebuilds the D7c delivery byte-identically | **PASS**, 8 of 8 | `6980c71` |
| 2 | the reproduction — `--body mhr`, lod6, the RAW array, through the build script | **PASS**, exactly (0.0 on 600 cells) | `6fb4288` |
| 3 | O1, the exactness oracle, its must-fails and its closure band | **FAIL → the step stopped** | `837fc63`, `5ebdc28` |
| — | coordinator's decision: continue, O1 a recorded exception — **later withdrawn** | — | `4338ada` (main) |
| 4 | the delivery — `--body mhr`, lod2, the smoothed repaired landmarks | built | `5865e06` |
| 5 | the bands — B1, B2, B3, B4, B5 | **B1 PASS, B2 PASS**, B3/B4/B5 reported | `d14224a` |
| 6 | the gate, the tests, the extractor stub, the report frames, this review | — | `fc173d4` |
| 6b | hygiene re-run at HEAD, the gate's B2/B5 derived from checks not verdicts, the unread inputs read | — | `e209eea` |
| — | **Astra's one merge round: NO MERGE at `e209eea`.** The exception is an override | — | review, 2026-09-22 |
| 7 | the repair pass: the exception branch removed, B1's population enforced, the required subclauses wired in, this review | — | `ff6e320`, `aac92c0`, `a177878`, this commit |

### The disposition, decided 2026-09-22

* **D4's ACCEPTANCE is recorded as FAIL on O1** — 1.030 mm against the 1 mm band, attributed in
  §2.1. Nothing about the band moved and nothing was re-selected.
* **The implementation merges as OPT-IN**: `--body mhr`, the default stays `rig`, nothing shipped
  changes. The gate derives that line from the rig rebuild's byte-identity, the `--body` default
  read out of the build script's own source, and `mhr` being reachable only by asking for it.
* **O1 gets a new, explicitly prospective registration as D4b.** No further review rounds on D4.

---

## 1. The clause table

| clause | predicted | measured | verdict |
|---|---|---|---|
| **hygiene** | `--body rig` rebuilds the D7c delivery 8 of 8 byte-identical | 8 of 8 identical, **re-run at HEAD** (`fc173d4`) after stage 2 changed the same file | **PASS** |
| **the reproduction** | `--body mhr` lod6 RAW reproduces 0.789 / 0.730 to 0.001 | 0.7894 / 0.7295; per-cell difference **0.0** on all 600 cells. *Repaired* (one process per performer, §4): 0.7894 / **0.7503** | **PASS** |
| **O1 exactness** | max over six seeds ≤ 1 mm | 0.853 / 0.985 / **1.030** / 0.967 / 0.799 / 0.898 mm | **FAIL — acceptance fails with it** |
| O1 must-fail, the mean body *(required conjunct)* | misses it on every seed, 5–60 mm | 31.8 / 32.8 / 24.7 / 9.6 / 39.8 / 32.2 mm | PASS (rejects by 9×–39×) |
| O1 closure *(required conjunct)* | the delivered GLB's FK = the track, ≤ 1e-4 m | worst 2.9e-6 m over eight files | PASS |
| closure input mutations | a mutated input must fail it | 0.156 / 1.92 / 0.112 m — all three reject | PASS |
| **B1, the band** | fitted MHR − D7c, lower CI bound > 0 on **both** performers, **on the named population** | **+0.1556 [+0.1326, +0.1626]** and **+0.1149 [+0.0842, +0.1355]**; 150 frames × 4 cameras consumed by every arm, 0 cells excluded, **600 of 600 scored** on each pair | **PASS** |
| B1 frozen-pose control below the candidate *(required conjunct)* | required | 8 of 8 cells (0.318–0.403 against 0.650–0.817) | PASS |
| B1 MAMMA arm unchanged *(required conjunct)* | bit-identical to its committed value | 8 of 8 cells identical | PASS |
| **B2 same denominator** | the consumed array is the rig converter's input; the markers re-derive | byte-identical on both; marker delta **0.0 cm** | **PASS** |
| delivery closure | the delivered GLB's FK = the track, ≤ 1e-4 m | 1.84e-6 / 1.96e-6 m | PASS |
| B3 placement *(reported)* | ~20 mm segment error expected (FITTER_PLAN §7) | all-landmark median 19.9 / 18.4 mm; segment mean abs error **11.9 / 9.0 mm** | reported |
| B4 the MAMMA arm *(reported)* | — | absolute median 34.5 / 43.1 mm (bias 50.7 / 34.5, spread 30.5 / 36.0) | reported |
| B5 delivered bytes *(reported)* | times, rest, hierarchy, skin, mesh | all byte clauses PASS on both | reported |
| B5 facing dot *(reported)* | head/eye facing dot > 0 | 150/150 and **145/150** frames, median angle 19.0° / 16.5° | reported, attributed |
| must-fail, free locator offsets | offsets median > 50 mm **and** the identity channels at zero | offsets **131 / 106 mm**, residual 150 / 115 mm; **13 identity channels nonzero on both** | the arm fails as it must; **the prediction's second half is FALSE** |

`artifacts/compare/d4-body/gate.json` carries every one of these with its predicted value, its
measured value and a verdict **derived** from a number in a report — not one literal `PASS` in
the file — plus the input-mutation table below. Its two output lines are

```
D4 ACCEPTANCE: FAIL (O1 1.030 mm > 1 mm; hygiene, reproduction, B1, B2, B1 frozen-pose control,
                     B1 MAMMA arm unchanged, O1 closure, O1 mean-body must-fail PASS)
OPT-IN IMPLEMENTATION: mergeable behind --body rig default (coordinator decision 2026-09-22,
                       acceptance open)
```

and there is no branch in the file that can print MERGE while a required conjunct fails.

### The gate is proven by mutating its inputs

| mutation applied to the gate's INPUTS | conjunct | verdict |
|---|---|---|
| one rebuilt file's sha256 changed | hygiene | PASS → FAIL |
| performer 0's reproduced IoU moved by 0.01 | reproduction | PASS → FAIL |
| O1's worst seed brought under the band | O1 | **FAIL → PASS** |
| **the consumed take truncated to 15 frames** (Astra's attack) | B1 | PASS → FAIL |
| one pair scored on fewer cells than the population names | B1 | PASS → FAIL |
| performer 1's lower CI bound put below zero | B1 | PASS → FAIL |
| performer 0's lower CI bound put **exactly at zero** | B1 | PASS → FAIL |
| **the frozen-pose arm lifted to 1.0** (Astra's mutation) | B1 frozen-pose control | PASS → FAIL |
| MAMMA's arm no longer bit-identical to its committed value | B1 MAMMA arm | PASS → FAIL |
| **O1 closure moved to 0.01 m** (Astra's mutation) | O1 closure | PASS → FAIL |
| **one mean-body seed brought to 0 mm** (Astra's mutation) | O1 mean-body must-fail | PASS → FAIL |
| a re-derived marker check falsified | B2 | PASS → FAIL |
| the consumed array no longer byte-identical to the rig converter's input | B2 | PASS → FAIL |

Every conjunct turns, O1 in the opposite direction from the rest — which is the check that its
FAIL is read from the measurement and not hardcoded. Astra's four demonstrated holes (the
truncated population and the three printed-but-unenforced clauses) are the four bold rows; each
of them now turns acceptance, and the producer refuses a truncated arm outright before the gate
ever sees it (`logs/32-b1-truncation-attack.log`).

---

## 2. The two FAILED predictions, attributed

### 2.1 O1 exactness — 1.030 mm against 1 mm

Robust to how the card's statistic is read. Max over six seeds on the oracle arm: 1.030
(median of per-frame medians), 1.259 (pooled median over joint-frames), 3.924 (worst per-joint
median), 2.245 mm (mean). Every reading exceeds 1 mm.

**Not the instrument, and not the duplicate length channels.** Scored one cell per process, the
simplified 178-parameter character and the full 204-parameter character with
`TrackingConfig.active_params` masking the 26 `*_flexible` channels agree to four decimals.
Astra's finding 1 — `spine_length_flexible` duplicates `scale_spine_length` among the pose — is
real, is handled, and is worth 0.012 mm. What it buys is that the oracle cannot be exact for the
wrong reason.

**Part of it was the fixture, and that part was repaired.** The card calls the donor "natively
representable". It is FK-representable, but measured against `compact_v6_1.model`'s own
`limit … minmax` entries it violates **23 pose parameters on up to 150 frames each** (45 counting
the `*_flexible` channels the fixture already holds at zero): `r_clavicle_rx` has limit [0, 0] and
is nonzero on every frame; `head_lean` has [−0.3, 0.3] and reaches 0.924. momentum's limits are
soft — the pre-card's own tracker produced this motion — but they are a penalty in the solve, so
a truth outside them is not a pose the solver will return and the fixture was charging the fitter
for its own infeasibility. Clamping the donor's pose to those limits (1025 parameter-frames) is a
**fixture parameter**, applied with the fitter's source byte-identical across the repair
(`mhr_delivery.py` sha256 `8ff8d973…`), and the same clauses rerun:

| | pre-repair | post-repair |
|---|---|---|
| oracle, max over six seeds | **1.2484 mm** | **1.0300 mm** |
| exact-identity floor, max over six seeds | 1.1604 mm | 0.7568 mm |

**The pre-repair reading stands on record** (`o1-prerepair.json`), is reproducible with
`--no-clamp` (verified: 1.2484 mm on the worst seed), and the repair is recorded as post hoc.
*What the repair is blind to:* `configured_limits()` parses `limit … minmax` lines; all 198
`limit` lines in the model file are `minmax`, so nothing declared there is missed, but any limit
momentum synthesises rather than reads was not checked against the donor.

**What remains is the fitter's own cost against a floor the band does not clear.** With the
*truth* identity handed in and only the pose re-solved, the tracker still leaves **0.51–0.76 mm**
on exact, noiseless, fully visible data. The 1 mm band therefore allows the identity fit 0.24 mm
on the worst seed, and it costs 0.21–0.35 mm. The excess is two channels:

* **`scale_spine_length` is MEASURED shrunk toward zero on every seed; the mechanism is not
  established.** The sign of the error is the opposite of the sign of the draw in six of six
  cells (+1.082 → −0.189, +1.078 → −0.209, −0.955 → +0.170, −0.750 → +0.153, +0.743 → −0.151,
  −0.886 → +0.149): **14–20 % of the drawn value pulled back to the mean**, with
  `scale_neck_length` compensating in the trunk on the seeds where the spine is stretched
  (+0.019…+0.033) and idle where it is shortened (±0.001). That is the measurement and it stands.
  An earlier draft of this review attributed it to a default-weight soft limit; **Astra refuted
  that** (2026-09-22): `scale_spine_length` is not the only drawn channel whose `limit` line omits
  a trailing weight — `scale_shoulder_width`, `scale_uparms` and `scale_lowarms` omit it too — and
  the installed min/max evaluator applies **zero penalty inside** the configured interval, where
  every draw lies. A missing weight therefore cannot be the mechanism, and no other candidate has
  been tested. The attribution is withdrawn; the number is not. Nothing was tuned either way: the
  solver's settings are the pre-card's, and touching them here would be selecting a constant on
  the oracle that scores it.
* **`scale_foot_length` is unidentifiable from this landmark set** and is recovered at ≈ 0 on
  every seed: the adapter maps no toe and `l_foot`/`r_foot` are the ankles, so nothing in the 17
  landmarks moves with foot length. `scale_hip_height` is a second dead channel for a different
  reason — its *configured* limit is the point [0.0, 0.0], so "uniform within its configured
  limit" draws identically zero. `scale_feet` does not exist in this release and
  `scale_foot_length` stands in for it under the card's own nearest-named-channel rule. All three
  facts are in the O1 report and in §6 below; none of them is in any band.

The locator offsets are not the explanation: at `limit_weight 10` they stay at 0.0000 mm median
and 0.0009 mm max after the full two-stage fit, so the card's "the same offsets the fitter uses"
premise holds exactly. The exact-identity tail is pose-dependent, not noise: the worst frames
(3.4–5.2 mm) are the **same frames on every seed** — 0, 30, 32/33, 52/53, 57/58 — and frame 0 is
among them on four of six, which is the tracker starting from the rest pose on a performer bent
over the ground.

**What O1 did prove.** The arm it exists to reject — the mean body, which beats the rig on B1 by
+0.119 / +0.119 on its own and would pass that band unaided — it rejects by a factor of **9 to
39**. There is no reading of this fixture under which a constant identity passes.

### 2.2 The free-offset must-fail — half the prediction is false

Rerun at the delivery's own settings (lod2, the smoothed array):

| | offsets median | offsets max | residual median | nonzero identity channels |
|---|---|---|---|---|
| pinned (`limit_weight 10`) | 0.0 mm | ~0 mm | 16.7 / 17.0 mm | 13 / 13 |
| **free** | **131.3 / 105.9 mm** | 362 / 439 mm | **149.5 / 115.1 mm** | **13 / 13** |

The arm is pathological and fails as a must-fail must — by offsets two to three times over the
50 mm clause and by a residual seven to nine times the pinned fit's. But the prediction's second
half, "*and* the identity channels at zero", is **false**: 13 channels are nonzero on both
performers and several sit on their configured limits. This is Astra's debt item exactly — the
"identity at zero" claim was carried over from FITTER_PLAN §7's **locator-only stage A** and was
never established for the full two-stage control. Recorded as a failed prediction, with the arm's
rejection intact on the other half.

---

## 3. What ships

`scripts/build_commercial_multiview_comparison.py --body mhr` (the default stays `rig` until this
merges, then flips). After the converter's smoothed landmarks — **the same array the rig
converter is handed on the same build** — the build writes that array to `converter-inputs/` and
calls `tools/fitter/mhr_delivery.py` under `/tmp/momenv/bin/python`, **one process per
performer**, which fits MHR's identity (stage A/B, pinned offsets) and per-frame pose at
**lod2** (10,661 vertices) and writes per performer:

* `subject-XX.glb` — skinned, 30 fps, one mesh, the marker spheres stripped;
* `subject-XX.body-track.json/.npz` — schema **`autoanim.body-track/2.0-mhr`**: MHR's 127 joint
  names, the 68 identity channels, the 136 pose channels per frame, the rest, the FK joints in
  the capture frame, the declared `landmark_to_joint`, and the consumed landmark arrays carried
  so every downstream instrument reads the array this body was fitted to;
* `subject-XX.markers.npz` — exactly what momentum received (B2's second half).

The rig path is untouched and remains selectable; hygiene is the proof. No constant is fitted.
The momentum configs (`calib_frames` 100, `loss_alpha` 2.0, `max_iter` 30, smoothing 0) are the
pre-card's, recorded as engineering settings and not selected. MAMMA enters nowhere: it is a
report arm on B1 and B4 only, and its mesh is bit-identical to its committed value.

Calibrated heights 184.04 / 183.73 cm; 13 identity channels each; locator residual 16.7 / 17.0 mm.

---

## 4. Three defects found in the path the pre-card walked

These are the most important lines in this review, because each of them ships a wrong file
silently.

1. **A momentum `calibrate_markers` call mutates whatever a LATER
   `Character.load_fbx(...).load_model_definition(...)` returns in the same process.** The same
   204 model parameters then put the skeleton somewhere else — skeleton-state sum −53291.87
   against −57778.49, the skinned mesh up to 0.76 m away, the glTF animation 79 channels instead
   of 98 — and the file still imports cleanly at the right 0..149 frame range. A character loaded
   *before* the first calibration is unaffected (its export is byte-identical before and after).
2. **The pre-card fitted both performers in one process**, so performer 1 was fitted on a model
   performer 0's calibration had moved. Repaired (one process per performer), performer 1's
   calibrated height goes 170.73 → 178.58 cm, its locator residual 27.44 → 16.72 mm, and its
   pooled IoU **0.7295 → 0.7503** (+0.0208 [+0.0091, +0.0403] paired, block 15, identical draws)
   at the pre-card's own lod6 and raw array. Performer 0, fitted first and so uncontaminated, is
   unchanged to 0.0 on all 600 cells. **Every MHR figure quoted for performer 1 before 2026-09-22
   carries that**, including the pre-card's 0.730. The pre-card script is left exactly as it fell
   — the guard is not a one-line change there, because it needs a second character loaded before
   any calibration — and this paragraph is the record.
3. **`Character.with_locators([])` does not clear locators** (17 in, 17 out), so the marker
   spheres cannot be stripped that way. The export character is loaded separately, before any
   calibration, and a guard compares its skeleton state against the fitted character's **on every
   frame** and refuses to write otherwise. That guard is what caught (1).

The delivered GLB now reproduces momentum's own `skin_points` to 5.7e-6 / 5.2e-6 m and carries
one mesh, not eighteen.

---

## 5. What each instrument is blind to

* **B1, the silhouette**, scores the MESH's outline and placement — not joints, not depth along
  the viewing ray, and not clothing, which is in the masks and in no mesh. It is not a
  limb-placement gate (D2's lesson). Part-wise numbers are an image-region decomposition built
  from the captured landmarks, identical for every arm; they are not anatomical part labels,
  because the SAM2 masks carry none.
* **Precision and recall are the inflation diagnostic and nothing more.** IoU penalises excess
  area but permits a recall gain to outweigh a precision loss. Measured: torso+legs precision
  0.862/0.822 → 0.882/0.869 and recall 0.730/0.784 → 0.834/0.833; arms precision 0.880/0.769 →
  0.854/0.818 and recall 0.740/0.698 → 0.881/0.812. The single precision fall — performer 0's
  arms, 0.880 → 0.854, where MHR's arm pixels exceed the mask's by 3 % — sits against a recall
  gain of 0.14 and is nowhere near the 3 px dilation control's signature (precision 0.948 →
  0.858 for IoU 0.743 → 0.796). **No precision veto was invented at merge time**, as the card
  fixed before the numbers.
* **The instrument's own `control_mean_body` is SMPL-X's mean shape under MAMMA's POSE.** It says
  nothing about the candidate, and on performer 1 it is above it in **all four cameras**
  (0.8181 / 0.8055 / 0.8047 / 0.6878 against 0.7794 / 0.8052 / 0.7523 / 0.6502 — Astra's
  correction, 2026-09-22; B001 is 0.0003 apart and it still counts). It is reported because the card says to report it.
* **O1 is a model-consistency oracle.** It cannot validate the take's landmark-to-joint
  convention: pinned zero offsets knowingly misstate that relationship, and it is lane H's marker
  session that owns it.
* **B3** scores the skeleton the file carries, not the mesh, and its reference is our own
  triangulation — a common-mode detector error is inside it.
* **B4**: MAMMA's joints are conventions. "MAMMA cannot referee a width" (CLAUDE.md); bias and
  spread are reported separately for exactly that reason.
* **B5's facing clause** is scored against the captured eyes and nose on each frame, so a miss
  there is a statement about the reference as much as about the file. The five misses on
  performer 1 (frames 34, 35, 42–44) have dots of −0.0002 to −0.006 against a typical magnitude
  ten times that, on frames where the capture's own eye-to-head vector is 1.2–1.5× its median
  length.
* **Equal body arrays are not equal information.** B2 proves the MHR route consumes the same
  smoothed, repaired array as the rig converter, byte for byte. It also records that the MHR
  route uses **17 of the 19 body landmarks** (SOMA-77 emits no ears) and **none** of the rig's
  auxiliary feeds — the five skull-rigid head landmarks, the two toe landmarks, and `Spine1`.

---

## 6. What is open

* **D4b — O1 re-registered prospectively, before fresh fixtures are evaluated.** Freeze, in
  writing and in advance: the statistic; a calibrated baseline taken from the measured
  0.7568 mm exact-identity floor rather than a round number; the tolerance; the **identifiable
  channel set** (which excludes `scale_foot_length` and `scale_hip_height` for this landmark
  set); and the mean-body rejection. Raising a threshold around the observed 1.030 is explicitly
  not that. The 1.030 mm FAIL stands on record until D4b reads.
* **The default flip is its own gated integration step**, not part of this one. It has to cover
  the compositor's consumption of the MHR body (`unified_gltf`, the N5.1 assembly),
  schema-aware artifact checks (the close-out's eight-file list assumes `subject-XX.mapping.npz`,
  which the MHR path does not write), instrument compatibility (`delivered_vs_capture`,
  `d3_skeleton_gate`, `retarget_cost`, `head_gate`, `facing_location`,
  `captured_limb_stability`, `delivered_foot` all read `BodyTrack` fields a `2.0-mhr` npz does
  not carry), and an end-to-end rebuild of the actual delivered MHR output. **This review
  carries no promise to flip the default at merge.**
* **`scale_spine_length`'s 14–20 % shrink toward the mean is measured and unexplained.** The
  soft-limit attribution was refuted (§2.1). It is the next thing to measure, and it belongs to
  D4b's registration, not to this step.
* **Two dead identity channels for this input**: `scale_foot_length` (no toe landmark) and
  `scale_hip_height` (configured limit is a point). Any future identity gate over MHR must state
  which of its channels the input can see. Neither is in any band.
* **The head, the toes and `Spine1` are not consumed by this route**, and the pelvis and hip
  conventions and the identifiability of the locator offsets remain lane H's marker session.
* **The finer LODs and the compositor's consumption of the MHR body** (`unified_gltf`, the N5.1
  assembly, which today consume the MPFB body) go to the integration step after this one. D6 is
  absorbed — there is no MPFB mesh to re-skin on this path — and D5's scaling becomes the
  marker-session step.
* **B5's facing clause and B3's neck/root/head groups** (56.6/72.4, 40.5/44.0, 39.8/38.8 mm) are
  convention gaps between MHR's joints and SOMA-77's landmarks, not fit error, and pinned offsets
  refuse to model them by design.
* **Pre-existing, not D4's — confirmed by name at the base commit.** Four tests fail on this
  checkout and fail identically in a throwaway worktree at `803f111` (the commit D4 branched
  from), with `src/` untouched by this step:
  * `tests/test_body_compositor.py::test_unified_preview_is_explicitly_diagnostic_and_uses_one_video_clock`
  * `tests/test_body_export.py::test_export_animated_body_glb_is_one_skin_one_timeline_and_hash_bound`
  * `tests/test_phase4_app.py::test_home_and_health`
  * `tests/test_phase4_app.py::test_api_blank_image_returns_typed_error`

  Full suite at HEAD: **1209 passed, 4 failed, 48 skipped**; the same four, and D4's own
  `tests/test_body_model.py` is **15 passed**
  (`artifacts/compare/d4-body/logs/33-full-suite-at-HEAD.log`).
  `.cache/autoanim_gnm` was wiped between 2026-09-15 and 2026-09-21, so `DEFAULT_BODY_RUN` no
  longer resolves and the rig rebuild needs
  `--body-run artifacts/compare/d1-fix/body-run-regenerated`.
* **The ladder registration is a stub.** `tools/compare/extractors/d4_body_model.py` supplies the
  `x_body_model` function and its `VISUALS`; `ladder.py` owns `RUNGS` and this step does not edit
  it.
* **THE CLOSE-OUT PROTOCOL CANNOT RUN ON THIS DELIVERY, and whoever merges needs to know before
  the `--body` default flips.** The card flips the default to `mhr` at merge. `post_merge.sh`
  then rebuilds `artifacts/commercial-multiview-soma77` as an MHR delivery, and (a) its
  eight-file byte-check names `subject-XX.mapping.npz`, which the MHR path does not write, and
  (b) every rig-schema instrument it reruns reads `BodyTrack` fields an `autoanim.body-track/
  2.0-mhr` npz does not carry — `delivered_vs_capture` (measured: `KeyError`, log 22),
  `d3_skeleton_gate`, `retarget_cost`, `head_gate`, `facing_location`,
  `captured_limb_stability`, `delivered_foot`. This belongs to the integration step alongside
  `unified_gltf` and the N5.1 assembly, but it is a merge-time hazard, not a later one: the flip
  and the close-out cannot both happen until that step has run.

---

## 7. Reproduce

```
# stage 1, hygiene
PYTHONPATH=$PWD/src .venv/bin/python scripts/build_commercial_multiview_comparison.py \
  --videos .cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos \
  --calibration-yaml .cache/mamma/configs/examples/calib/iphones_outdoors.yaml \
  --detector soma77 --body rig --body-run artifacts/compare/d1-fix/body-run-regenerated \
  --output artifacts/compare/d4-body/hygiene

# stage 2, the reproduction (lod6, raw)  /  stage 4, the delivery (lod2, smoothed)
... --body mhr --mhr-lod 6 --mhr-landmarks raw    --output artifacts/compare/d4-body/repro-v2
... --body mhr --mhr-lod 2 --mhr-landmarks smoothed --output artifacts/compare/d4-body/delivery
/Applications/Blender.app/Contents/MacOS/Blender --background --python \
  tools/compare/blender_export_mesh_momentum.py -- OUT.npz DELIVERY_DIR 30

# stage 3, O1
/tmp/momenv/bin/python tools/fitter/d4_o1_exactness.py --drive --out artifacts/compare/d4-body/o1 --lod 2
/tmp/momenv/bin/python tools/fitter/d4_o1_exactness.py --drive --out DIR --lod 2 --no-clamp   # the pre-repair fixture

# stage 5, the bands
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_silhouette_paired.py --out b1-paired.json --arm ... --pair ...
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_silhouette_paired.py --out /dev/null --parts b1-parts-terciles.json --landmarks DELIVERY --arm ...
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/silhouette.py --delivery DELIVERY --work WORK --out silhouette-delivery.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_b2_same_denominator.py --delivery DELIVERY --rig-build HYGIENE --out b2.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_b3_placement.py  --delivery DELIVERY --out b3.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_b4_mamma_arm.py  --delivery DELIVERY --out b4.json
/tmp/momenv/bin/python tools/fitter/mhr_delivery.py --inputs DELIVERY/converter-inputs --out . --lod 2 --dump-reference mhr-reference-lod2.npz
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_b5_delivered_bytes.py --delivery DELIVERY --reference mhr-reference-lod2.npz --mesh MESH.npz --out b5.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_glb_closure.py --pair NAME=GLB,TRACK --out closure.json

# stage 6
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_body_gate.py --out artifacts/compare/d4-body/gate.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_report_frames.py --out artifacts/compare/d4-body/report
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_body_model.py
```

Logs, every line, under `artifacts/compare/d4-body/logs/`; reports beside them under
`artifacts/compare/d4-body/`. Report frames and the mp4 under `artifacts/compare/d4-body/report/`
(50 JPEGs at 640 px q42, every 6th frame, cameras A001 and D001, the rig beside MHR with the SAM2
mask outline under both; aqua is the rig, blue is MHR, and orange is absent because orange is
MAMMA's in this lane and MAMMA is not drawn here).
