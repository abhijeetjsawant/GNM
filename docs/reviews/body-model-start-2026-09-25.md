# D4c: the calibration's start. The measurement (2026-09-25)

> **Disposition (coordinator, 2026-09-25, Astra's merge round):** FAIL (L). This review and its records are merged to main
> ALONE. The tooling it cites (`tools/fitter/d4c_*.py`, `tools/compare/d4c_start_gate*.py`, the hash-bound
> `d4_glb_closure.py`, `tests/test_d4c_start.py`, the extractor stub) and the landmark-start change to
> `tools/fitter/mhr_delivery.py` are NOT on main. They are pinned at commit `1a89cc73b2ced93543e9cbfb191d79208aef423b`
> (tag `ladder/D4c-fail-1a89cc7`, branch `ladder/D4c`). To reproduce any figure below, check out that commit. The tooling
> merge the card named is DEFERRED debt, not done.


**VERDICT: FAIL (L). D4 stays open. O1 is not superseded. The `fit_one` change stays on the branch.**

The gate prints one line:

    VERDICT: FAIL (failed: L)

Every conjunct but one holds. Precondition 0 holds: the limit-aware rule draws eight channels, the spine and the
shoulder width among them, so the trunk is scored. Stage 0b holds on 18 of 18. The rest hold too: validity, closure,
must-fails i–iv, B1, B2 and hygiene.

**L fails on one fixture of twelve.** On 20261106 donor 0 the fitted trunk is 2.13 mm off against a paired
tolerance of 1.94 mm, 1.10× the band. That fixture draws the shortest spine in the acceptance set, −0.911 units. The
other eleven fixtures pass every scored segment. The worst limb segment on any fixture sits at 0.47× its tolerance.

The card's consequence applies unchanged:

* tooling and records merge;
* the `fit_one` change does not merge, so nothing opt-in changes on a failed oracle;
* D4 stays open.

The falsifier's case does not arise on the failing fixture (§5): the start was not near the truth there.

| stage | commit | what |
|---|---|---|
| 1 | `86f1a5a` | Precondition 0: the limit-aware drawn set, frozen before any fit. **Eight drawn**, the spine among them. No STOP |
| 2 | `6f261c5` | The ONE code change. The tripwire: the zero start through the new code reproduces D4's delivery and all 18 retained O1 cells. `--body rig` rebuilds 8/8. Momentum's calibration frames were read (0..149) |
| 3 | `85058cc` | Development on 18 burned fixtures. Stage 0b holds 18/18. The trunk statistic was frozen at **p90**. The development JSON was committed before any acceptance fixture existed |
| 4 | `9f664b3` | Acceptance: 12 fixtures × 6 arms. Closure on every candidate cell. The manifest names the cells by content |
| 5 | `efb5d66` | The real take: `--body mhr` rebuilt with the new start. B1 and B2 re-run |
| 6 | `46790f8` | ONE verdict. The fuzz turns every conjunct from its inputs: 33/33, and the leaf walk leaves 0 leaves unjustified |
| 7 | this commit | These tests, this review, the extractor stub and the report frames |

The records are in `docs/reviews/body-model-start-records/`. The logs are in `artifacts/compare/d4c-start/logs/`
(01–26). Nothing under `src/` changed, and neither did the build script.

---

## 1. The pre-registration, verbatim

The card is the `D4c the calibration's start` row of `docs/LADDER_EXECUTION_PLAN.md` §2. It is copied below
unchanged. `docs/reviews/body-model-start-card-2026-09-25.md` is byte-identical to that row (checked with `cmp`).

| **D4c the calibration's start, O1 re-registered with the trunk a precondition** | **Why.** D4's O1 is still open. D4b STOPPED at stage 3 (`docs/reviews/body-model-o1-2026-09-24.md`): its limit-blind drawn-set rule left the trunk unscored, so the band could not reject the displaced-spine control. Its post-stop probes, now development evidence and never acceptance evidence, point at WHERE the calibration starts. Started at the truth identity (WARM), the fit holds the spine within 0.005 and the trunk within 0.45 mm. Started at zero, it stops 0.149–0.209 short. `max_iter` 300 from zero still drifts one burned body to 0.445 short (trunk 44 mm), so more iterations are not the lever and may find a wrong basin. **The change, ONE part: the calibration's starting identity.** `tools/fitter/mhr_delivery.py` `fit_one` hands both `calibrate_markers` calls (stage A, locators only; stage B, identity) a landmark-derived starting identity instead of `zero.copy()`. Nothing else in the fitter changes: `max_iter` stays 30 (the compute constraint, unchanged, so the attribution is clean); `calib_frames` 100, `loss_alpha` 2.0, pinned offsets (`limit_weight` 10), and tracking are all unchanged. **The starting identity, frozen, landmarks only:** for each drawn channel c and its one segment s (D4b's `segment_rest_length_change_mm_by_channel` shows every channel moves one segment kind, a bilateral channel moving both sides equally; that table records the largest ABSOLUTE change at either end, so the signed slopes are computed afresh here): start_c = (ℓ_s − ℓ⁰_s) / k_c, clipped to c's configured limit. ℓ_s is the median over the calibration frames of that segment's length in the CONSUMED landmark array (bilateral segments are the mean of the two sides' medians). ℓ⁰_s is the rest length at zero identity. k_c is c's SIGNED rest-length change per unit, computed afresh (the change at the upper configured limit over that limit; about 100 mm per unit, 200 for shoulder width and hip width; linearity checked at the lower end and reported). Every undrawn channel starts at 0. It reads landmarks as joints, the same pinned-zero-offset convention as O1 (lane H's question, stated). **Development phase, declared (18 burned fixtures: D4's six + D4b's twelve):** the start is run there first. One adjustment is allowed, and only one: the per-segment statistic for the NON-RIGID trunk, chosen from {median, p90, p95} of the frames' root→c_neck length (flexion shortens the chord, so the median is biased short, the same direction as the shrink). The rule: the smallest worst-case POST-CALIBRATION trunk L-ratio over the 18 fixtures; ties go to the median, then p90, then p95. Frozen before any development reading: the calibration-frame indices (the fitter's own `calib_frames` selection), percentile interpolation (numpy `linear`) and the tie order. It is recorded and frozen in a JSON before any acceptance fixture is generated; nothing else may change after that. **Acceptance population, untouched:** new seeds 20261101–20261106 × donors 0 and 1 (12 fixtures), the identity draw seeded by (seed, donor), the drawn set by the rule below, poses clamped to the configured limits. Only two donor motions exist, so the pose population is D4b's: the held-out dimension is the identity, and the per-fixture paired floor carries the pose dependence. Stated. **Real take (the fitter change reaches the opt-in delivery):** `--body mhr` is rebuilt at lod2 on the smoothed array with the new start, and D4's own B1 is re-run on it. **What D4c's verdict does:** PASS → D4's O1 is superseded by D4c's fitter with B1 re-shown on that fitter, and D4 and D4c close together. FAIL → tooling and records merge; the `fit_one` change stays on the branch (nothing opt-in changes on a failed oracle); D4 stays open; and the consequence registered now applies. If the trunk is still beyond tolerance while init-only sits near the truth, the leading hypothesis is the objective's basin (stopping and stage coupling remain alternatives) and D4d takes it. If the precondition STOPS, the trunk failed the screening rule, and the `Spine1` feed (integration step) or lane H's markers are the route. Window 0. | 7 | **Precondition 0, the drawn set by a LIMIT-AWARE rule, computed before any fit.** For each of D4's ten named channels at each end of its configured limit, per donor frame: the least-squares pose correction δ within the configured limits (`scipy.optimize.lsq_linear`, bounds = limits − donor pose, flexible channels excluded) minimising ‖J_pose·δ − d_c‖ over the 17 mapped joints. The channel is drawn iff the residual's median over frames exceeds **2 mm** on both donors. Frozen with it: the evaluation identity (zero), the endpoint aggregation (max over the 17 joints), the finite-difference step, the 0.1 mm segment-movement cut, and rounding (none before comparison). The aggregation is D4b's in full: the maximum joint norm, then the maximum over the two limit ends, then the median over frames. Also frozen: the Jacobian and displacement sign convention, the pose-column membership (flexible channels excluded), bounds for parameters with no configured limit (unbounded) and fixed parameters (zero width), and a solver failure on any frame (STOP, never a silent drop). The unbounded projection is REPORTED beside it. Predicted: eight drawn, including `scale_spine_length` and `scale_shoulder_width` (D4b measured their absorbing steps at 17–65 against limits ≤ 1.5). **If `scale_spine_length` is not drawn, the step STOPS here: no fit runs and no acceptance fixture is generated.** This is a screening rule, so a STOP says the trunk failed THIS rule (a linearised bound can still misjudge a large angular step), not that the trunk is unidentifiable from the landmarks. The trunk being scored is a precondition, never a clause. **Stage 0b, on the development set before acceptance:** the spine-displaced control must FAIL L AS SCORED at the trunk on all 18; if it passes on any, STOP. **THE band, L (D4b's, unchanged):** the rest-segment error of every segment a drawn channel moves, the trunk and shoulder width included, each within the sum of its two endpoints' paired per-fixture floors (the `exact_identity` arm, same fixture), on all twelve acceptance fixtures. **Validity:** the exact arm's pooled statistic ≤ 1.0 mm on every fixture, else INVALID; INVALID leaves D4 open and never licenses replacing a fixture. **Must-fails, each on every acceptance fixture, through the actual scorer as scored:** (i) the mean body misses L; (ii) the spine displaced 0.149 units toward zero, held, pose re-solved, misses L at the trunk; (iii) the exact identity reads L = 0 and PASSES (a positive implementation control); (iv) INIT-ONLY, the starting identity held with no calibration and the pose re-solved, must MISS L. Init-only bypasses BOTH calibration stages (locators and identity). If it passes on any fixture, the calibration is unnecessary for passing that fixture, the oracle cannot then score the calibration, and the step STOPS. That is conservative by design: it says nothing against the start, and a start-only delivery would need its own registration. Predicted: it misses at the trunk (a chord is not a rest length). **Required beside L:** closure, the delivered GLB's FK = the track ≤ 1e-4 m on every oracle cell, the closure report carrying and the gate verifying the sha256 of the GLB and track it measured (D4b's path-substitution hole). **B1 on the real take (D4's band, re-run on the D4c fitter):** whole-body IoU against the SAM2 masks at 960×540, whole take, four cameras pooled, per performer, paired per frame-camera cell, moving-block bootstrap (block 15, 2000 resamples, identical draws). Band: D4c-fitted minus the D7c rig with the LOWER CI above zero on BOTH performers. D4's other required B1 conjuncts are inherited: the frozen-pose control below the candidate, and MAMMA's mesh bit-identical to its committed value. The silhouette inputs are bound by sha256 to the rebuilt delivery. D4c-fitted minus D4-fitted is REPORTED with its CI; it is NOT a non-regression band, and a regression against D4 that still beats D7c on both performers is a registered PASS. The population is named and refused if short (150 frames × 4 cameras × 2 performers). **B2 same denominator (D4's clause, re-run). Hygiene:** `--body rig` rebuilds the D7c delivery 8 of 8 byte-identical, and the tripwire (the zero start forced through the new code) reproduces D4's `--body mhr` delivery and D4's retained O1 cells byte-identical. Attribution also needs the source diff: `fit_one` changes ONLY in the two starting identities, each receiving its own copy (the WARM hook's precedent). **Reported, never banded:** the legacy zero-start fitter on the acceptance fixtures (the before arm, predicted to miss the trunk as in D4b); J per joint; the per-channel start and recovered error; the calibration's cap-exhaustion count, per stage (A and B) and for the inner solver separately (D4b's "973 of 1056" is stage B only); the development phase's four readings (zero start, the chosen start, each trunk statistic); and what a PASS does not establish (the full parameter vector, the eye channels, the landmark-to-joint convention). **Prediction, fixed before acceptance numbers:** PASS. The trunk within its paired tolerance on most or all fixtures (WARM ≤ 0.45 mm; a landmark start is noisier than the truth); limbs PASS; init-only misses at the trunk; the legacy arm misses the trunk; B1 holds (D4 read +0.156 / +0.115), with D4c − D4 within ±0.01. **Falsifier:** the trunk still beyond tolerance with the start near the truth points to the objective's basin, not the start. That is a HYPOTHESIS: finite stopping and stage-A/B coupling remain alternatives (and D4b's CONVERGED probe changed the tracking budget too, since `max_iter` is shared). **Population, named and refused if short:** 12 fixtures × {candidate, exact_identity, mean_body, spine_displaced, init_only, legacy} × 150 frames × 17 joints; a missing, short or non-finite cell is FAIL; every set is checked by identity; the gate VERIFIES constructions, not labels: it regenerates each seeded draw, recomputes the init-only identity from the consumed landmarks and the frozen rule, and binds each closure report's measured hashes to the actual files. Reported: whether the fixtures' allowable rest-length intervals have an empty intersection on any scored segment, which rules out a common constant body; provenance by sha256 (the fitter, both donors, `compact_v6_1.model`, `lod2.fbx`, the pymomentum version, the frozen development JSON). Card reviewed by Astra GPT6 once, at medium effort (`docs/reviews/body-model-start-astra-review-2026-09-25.md`): dispatchable, no blockers; every finding adopted into this text. **ONE verdict, no split disposition line:** PASS iff precondition 0 ∧ stage 0b ∧ validity ∧ L ∧ closure ∧ must-fails i–iv ∧ B1 ∧ B2 ∧ hygiene. Otherwise FAIL, INVALID or STOP, and D4's disposition is derived from that one verdict. The gate is `tools/compare/d4c_start_gate.py`, with a per-conjunct input-mutation fuzz (`d7c_gate_fuzz.py` pattern). **Merge rule, fixed before numbers:** PASS → everything merges, `--body mhr` carries the new start, and the default stays `rig` (the flip is the integration step). Otherwise tooling and records merge and `fit_one` does not. |

---

## 2. The clause table (`artifacts/compare/d4c-start/gate.json`; a copy is in the records)

| clause | predicted (the card) | measured | verdict |
|---|---|---|---|
| **precondition 0** | eight drawn, spine and shoulder width among them | **Eight drawn**: spine, neck, shoulder width, upper arms, forearms, hip width, thighs, shins. The bounded residual is 52.5/55.3 mm for the spine and 17.6/12.9 mm for shoulder width. `foot_length` is UNSEEN and `hip_height` UNDRAWABLE. 0 solver failures in 6000 solves | HOLDS (prediction held) |
| **stage 0b** (development) | the spine control fails L as scored on all 18 | 18 of 18. The trunk is scored | HOLDS (held) |
| validity | exact pooled ≤ 1.0 mm on every fixture | 0.555–0.848 mm, 12/12 | PASS (held) |
| **L** | PASS: the trunk within tolerance on most or all fixtures; limbs PASS | **11 of 12**. The trunk misses on **20261106 d0: 2.13 mm against 1.94 mm (1.10×)**. On the other eleven the trunk reads 0.02–0.81×. Every limb and width segment is within tolerance on all twelve (worst 0.47×) | **FAIL** (prediction FAILED; its "most" half held) |
| closure | ≤ 1e-4 m on every candidate cell, bound by content | 1.9e-6 to 3.4e-6 m, 127 joints × 150 frames. Each row carries and matches the measured GLB and track sha256 | PASS (held) |
| must-fail (i), mean body | misses L everywhere | 12/12. Its closest fixture is 32× tolerance | PASS (held) |
| must-fail (ii), spine −0.149 | misses L at the scored trunk everywhere | 12/12. Trunk 14.87–14.88 mm against 1.8–2.4 mm; the closest fixture is 6.1× | PASS (held) |
| must-fail (iii), exact identity | L = 0 and PASS | 0.000000 mm, 12/12 | PASS (held) |
| must-fail (iv), init-only | misses L; predicted **at the trunk** | Misses L on 12/12, so no STOP. **At the trunk on only 1/12** (20261106 d0, 1.67×). It misses through **shoulder width** everywhere (3.1–23× on its worst segment) | PASS (the "at the trunk" prediction FAILED) |
| B1, D4c − D7c | lower CI > 0 on both performers | **+0.1555 [+0.1332, +0.1628]** and **+0.1178 [+0.0889, +0.1395]**; 600/600 cells each. The frozen-pose control is below the candidate in 8/8 cells. MAMMA is bit-identical in 8/8. The scored mesh is bound by sha256 to the rebuilt GLBs | PASS (held) |
| B1, D4c − D4 *(reported)* | within ±0.01 | −0.0000 [−0.0003, +0.0009] and +0.0029 [−0.0015, +0.0115] | reported (held) |
| B2 | the consumed array is the rig converter's input; the markers re-derive | Both performers pass every numbered check against the stage-2 hygiene rig build. The consumed arrays are byte-identical to D4's | PASS (held) |
| hygiene | `--body rig` 8/8; the tripwire; source diff = the two starts | 8/8 byte-identical. Tripwire: 62/62 files byte-identical and the JSONs equal after normalising named path fields. `fit_one`'s AST equals the base's once the start line and the two `start.copy()` are undone | PASS (held) |
| the legacy arm *(reported)* | misses the trunk | It misses the trunk on 11/12 (up to 30×) and misses L on 11/12 | reported (held) |
| **overall** | **PASS** | **FAIL (L)** | **FAIL** |

The allowable rest-length intervals have an **empty intersection on every scored segment**. That is reported, and
it rules out one constant body passing all twelve fixtures.

## 3. The development record (the 18 burned fixtures; `development.json`, committed at `85058cc`)

Frozen before any development reading, and committed at stage 2 in `mhr_delivery.py`:

* **The calibration frames.** Momentum's own selection: `computeSampleStride`, greedy 0, frames 0..149. Read from
  `calibrate_markers`' own return value and equal to `calibration_frame_indices` (`calibration-frames.json`).
* **The percentiles.** numpy `linear`.
* **The tie order.** median, then p90, then p95.
* **The numerator.** The post-calibration trunk |error| divided by its paired tolerance.

Stage 0b holds on 18 of 18. The worst-case trunk L-ratio over the 18 fixtures:

| arm | worst trunk ratio | trunk within | L passes |
|---|---|---|---|
| legacy (zero start) | 11.93 | 3/18 | 3/18 |
| landmark start, median | 4.77 | 6/18 | 6/18 |
| **landmark start, p90 (chosen)** | **1.27** | **15/18** | **15/18** |
| landmark start, p95 | 1.56 | 15/18 | 15/18 |

**Development already showed the acceptance failure mode.** Under p90, the three development misses (1.27, 1.02 and
1.15×) are the three D4 fixtures with **negative** spine draws (−0.955, −0.750, −0.885). There the landmark start
reads 0.030–0.033 units long, and 30 iterations of calibration recover only a quarter of that. The one acceptance
miss is the only acceptance fixture with a large negative draw (−0.911; start −0.878, fit −0.889).

The card's prediction ("most or all") left room for that. The band did not: it is "on all twelve".

## 4. The FAILED predictions, attributed

**(a) "Overall PASS", through L.** The trunk misses on one fixture of twelve, by 0.19 mm against a 1.94 mm tolerance.

* **The start.** The p90 chord statistic reads a strongly shortened spine long on donor 0's motion: 0.033 units,
  3.3 mm of trunk. On the positive draws the same statistic is 0.007–0.017 long. On donor 1 it is exact.
* **The calibration.** Thirty iterations move that start 0.011 toward the truth and stop 0.021 short.
* **Cap exhaustion is reported, not attributed.** Stage B leaves 605–607 of 1056 solves at the configured cap and
  371–413 above it, on every candidate cell.

The start and the stop share this miss. No single cause is measured.

**(b) "Init-only misses at the trunk".** It does so on 1/12 fixtures only. On donor 1 the p90 trunk chord equals the
rest length to within 0.0005 units, so the held start is already exact at the trunk (0.00–0.02×). Init-only still
misses L on every fixture, through **shoulder width**, and on donor 0 through more segments. The clavicle pose
changes the shoulder chord, so its median over frames is not the rest width. Must-fail (iv) therefore holds as a
must-fail (the oracle can score the calibration), but not for the predicted reason.

That is also why calibration is necessary here. It is the shoulder width that the start cannot supply, not the
trunk.

**(c) B1's reported "D4c − D4 within ±0.01" held.** The fitter change moved the real take's IoU by nothing
measurable on performer 0 and by +0.003 on performer 1.

## 5. The falsifier's reading

The card: *"the trunk still beyond tolerance with the start near the truth points to the objective's basin."*

**On the failing fixture the start was not near the truth.** Init-only reads the trunk at 1.67×, and calibration
improved it to 1.10×. Nothing here licenses the basin hypothesis for this failure. The start's error and the finite
stop are the two terms, and neither is excluded.

**The falsifier's signature does appear, within tolerance, on donor 1.** There the landmark start is EXACT for the
spine (start − truth ≤ 0.0005 on all six fixtures), and the calibration moves it away by −0.005 to −0.019 units
(trunk 0.23–0.81× tolerance). Started at the truth, the fit leaves it.

This differs from D4b's WARM probe, which started EVERY channel at the truth and held the spine within 0.005. Here
only the spine starts exact. Shoulder width starts −0.05 to −0.13 off (the non-rigid chord), and the recovered spine
error has the sign of a compensation. This is a hypothesis, **stage coupling through a wrong companion channel**. The
objective's basin and the finite stop remain alternatives. It is D4d's evidence, not a finding.

## 6. The real take (REPORTED; the take has no truth)

| performer | spine: D4 / D4c start / D4c fit | neck: D4 / start / fit | calibrated height |
|---|---|---|---|
| 0 | 0.971 / 0.707 / 0.959 | 0.446 / **−0.395** / 0.446 | 183.95 cm |
| 1 | 0.675 / 0.157 / **0.420** | 0.599 / **−0.400** / 0.531 | 180.28 cm |

The neck start sits at its lower limit on both performers. The captured `neck`→`nose` landmark distance is about
60 mm against MHR's 99 mm `c_neck`→`c_head` rest. That is the landmark-to-joint convention gap (lane H), not a fit
error, and the start reads landmarks as joints by design (the card states it).

On performer 1 the recovered spine moved 0.675 → 0.420 with the start. The silhouette cannot tell which is right: B1
reads +0.003. The take has no truth.

## 7. What each instrument is blind to

* **L scores lengths, not the parameter vector.** A compensating pair can hold every scored length with a wrong
  identity. On donor 1 the recovered spine and shoulder width both move off the truth while every length stays
  within tolerance.
* **The tolerance is an engineering rule, not a bound.** The sum of two marginal medians does not bound a median
  length error.
* **Every fixture is exact, noiseless, with pinned zero offsets.** The landmark start on real landmarks carries the
  convention gap of §6, which no fixture here has.
* **The start is chord-based.** A segment whose landmark chord is non-rigid biases its start: the trunk under
  flexion, shoulder width under clavicle pose. Only the trunk was given a development choice.
* **Only two donor motions exist.** The held-out dimension is the identity. The failing pose-by-draw combination
  (donor 0, a strongly shortened spine) was visible in development, as §3 says.
* **The precondition is a screening rule.** It is a linearised bound within the limits, so it can misjudge a large
  angular step in either direction.
* **B1** scores the mesh's outline against clothed masks. It cannot see a 0.25-unit spine change on performer 1.
* **The fuzz** proves that what the gate reads, it depends on, not that it reads the right things. The measured
  verdict is FAIL, which would make FAIL-direction mutations vacuous. So every conjunct is turned from a REPAIRED
  baseline, with the one failing trunk brought onto the truth, and that baseline reads PASS. Each targeted conjunct
  is also shown to fall on the measured inputs. The leaf walk is bounded to one fixture's six cells, its closure row
  and the B1/B2 rows.

## 8. What is open

* **D4 stays open; O1 is not superseded.** Fixtures 20261101–06 × donors 0/1 are now burned (acceptance, read). A
  further registration needs new seeds.
* **D4d, from this evidence.** Two terms, neither excluded.
  * The start's non-rigid chord statistic. It fails on donor 0's short spines, and it is inexact for shoulder width
    everywhere.
  * The calibration's drift from an exact spine start (donor 1), with a wrong companion channel as the leading
    hypothesis.

  A start whose trunk and shoulder width are not chord statistics (for example the `Spine1` feed, or a
  pose-compensated length) is one candidate. A calibration that holds a correct start is the other. Neither is
  measured here.
* **The detection cache.** The SOMA-77 model is absent from `.cache/autoanim_gnm/gem-x` on 2026-09-25. Both builds
  reused the shipped delivery's cached detections, byte-identical to D4's (log 05a, 05b). A fresh detection is not
  reproducible on this machine until the model is restored.
* **Merge hazard.** Under FAIL the card says tooling and records merge and the `fit_one` change does not. On this
  branch the two cannot be cleanly separated.
  * `d4c_fixture.py` calls `mhr_delivery.landmark_start` and reads `TRUNK_STATISTIC`, and it passes
    `start_identity=` to `fit_one`.
  * The gate's hygiene conjunct and `trunk_statistic_in_source` read the fitter's source.
  * `test_the_source_diff_accepts_only_the_two_starts` asserts that the current fitter differs from `3136befb` in
    exactly the two starts.

  Reverting `mhr_delivery.py` to `3136befb` and merging the rest therefore breaks the tooling and that test. Two
  clean options, for the coordinator to decide (not decided here):
  * (a) merge `mhr_delivery.py` with `main()`'s default switched back to the zero start. The `fit_one` keyword stays,
    and it is inert when None: the tripwire proves that path byte-identical. Record it as the coordinator's decision.
  * (b) keep the whole branch unmerged.
* **Instrument debt.** The gate's hygiene conjunct re-derives the 62 tripwire file hashes from bytes. The two
  normalised-JSON comparisons it reads only as the boolean `all_byte_identical_or_equal_after_normalising` that the
  fixture wrote. The fuzz turns that boolean, so it is enforced, but it is a label, not a re-derivation (Astra's
  finding-6 class).

## 9. Decisions an executor made

1. **The build script is unchanged.** `mhr_delivery.py`'s `main()` defaults to the landmark start, and
   `--zero-start` is the flag the card asks for, so `--body mhr` selects the new start without editing the build
   script. The start used is written as `subject-XX.calibration-start.json`, never into the fit report or the track.
   That is what keeps the zero-start tripwire byte-identical.
2. **The landmark start is computed on the export character** (loaded before any calibration, same parameter layout)
   and passed to `fit_one`. In the fixture it is computed on the 178-parameter character the fixture fits with.
3. **The tripwire JSONs.** A body-track JSON carries a worktree-rooted assets path, and D4's O1 cell JSONs carry
   paths and a fitter sha. These were compared field by field after normalising exactly those named fields. Every
   GLB, npz and fit report is compared byte for byte.
4. **The cap-exhaustion split** is by each solve's last logged iteration index: below `max_iter − 1`, at it, or above
   it (a solver inside `calibrate_markers` with its own, larger cap, which reaches index 49).
5. **The detection cache was reused** (open item above). Hygiene still reads 8/8 against the shipped files.
6. **The tests were committed at stage 4, with the gate's source-diff repair.** The source diff first missed the
   `start` line inside an `else` branch. It was repaired before any gate verdict was read, and now also requires the
   start expression's exact text.

7. **A second tripwire reading, on the fixture path.** The development `legacy` arm ran on the stage-2 fitter over
   D4's six seeds. Its pooled statistics equal D4's retained oracle cells to four decimals: 0.8533, 0.9851,
   1.0300, 0.9669, 0.7986 and 0.8977 (log 09).

## 10. Reproduce

```
/tmp/momenv/bin/python tools/fitter/d4c_identifiability.py --out artifacts/compare/d4c-start/drawn-set.json \
    --frozen-copy docs/reviews/body-model-start-records/drawn-set.json
for s in 0 1; do /tmp/momenv/bin/python tools/fitter/mhr_delivery.py --inputs artifacts/compare/d4-body/delivery/converter-inputs \
    --out artifacts/compare/d4c-start/tripwire/delivery --lod 2 --landmarks smoothed --subject $s --zero-start; done
/tmp/momenv/bin/python tools/fitter/d4_o1_exactness.py --drive --out artifacts/compare/d4c-start/tripwire/o1 --lod 2
/tmp/momenv/bin/python tools/fitter/d4c_fixture.py --tripwire-compare --out docs/reviews/body-model-start-records/tripwire.json
/tmp/momenv/bin/python tools/fitter/d4c_fixture.py --calibration-frames --out docs/reviews/body-model-start-records/calibration-frames.json
PYTHONPATH=$PWD/src .venv/bin/python scripts/build_commercial_multiview_comparison.py ... --body rig \
    --body-run artifacts/compare/d1-fix/body-run-regenerated --output artifacts/compare/d4c-start/hygiene
/tmp/momenv/bin/python tools/fitter/d4c_fixture.py --drive --population d4  --out artifacts/compare/d4c-start/development
/tmp/momenv/bin/python tools/fitter/d4c_fixture.py --drive --population d4b --out artifacts/compare/d4c-start/development
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4c_start_gate.py --development --out artifacts/compare/d4c-start/development.json
/tmp/momenv/bin/python tools/fitter/d4c_fixture.py --drive --population acceptance --out artifacts/compare/d4c-start/acceptance
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_glb_closure.py --pair SEED_dD=GLB,TRACK ... --out artifacts/compare/d4c-start/closure.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4c_start_gate.py --manifest --out docs/reviews/body-model-start-records/acceptance-manifest.json
PYTHONPATH=$PWD/src .venv/bin/python scripts/build_commercial_multiview_comparison.py ... --body mhr --mhr-lod 2 \
    --mhr-landmarks smoothed --output artifacts/compare/d4c-start/delivery
Blender --background --python tools/compare/blender_export_mesh_momentum.py -- \
    artifacts/compare/d4c-start/work-delivery/delivered-mesh.npz artifacts/compare/d4c-start/delivery 30
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/silhouette.py --delivery artifacts/compare/d4c-start/delivery \
    --work artifacts/compare/d4c-start/work-silhouette --out artifacts/compare/d4c-start/silhouette-delivery.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_silhouette_paired.py --out artifacts/compare/d4c-start/b1-paired.json \
    --arm baseline_D7c_rig=artifacts/compare/i6/delivered-mesh.npz \
    --arm D4_fitted_MHR_lod2=artifacts/compare/d4-body/work-delivery/delivered-mesh.npz \
    --arm D4c_fitted_MHR_lod2=artifacts/compare/d4c-start/work-delivery/delivered-mesh.npz \
    --pair D4c_fitted_MHR_lod2,baseline_D7c_rig --pair D4c_fitted_MHR_lod2,D4_fitted_MHR_lod2
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_b2_same_denominator.py --delivery artifacts/compare/d4c-start/delivery \
    --rig-build artifacts/compare/d4c-start/hygiene --out artifacts/compare/d4c-start/b2-same-denominator.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4c_start_gate.py --out artifacts/compare/d4c-start/gate.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4c_start_gate_fuzz.py --out artifacts/compare/d4c-start/fuzz.json
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_d4c_start.py
```

`tests/test_d4c_start.py` passes 17 tests, and `tests/test_d4b_o1.py` 18. The full suite (log 26) reads 1249 passed,
43 skipped and 4 failed. The four failures are the same four pre-existing ones D4 recorded at its base
(`test_body_compositor` unified preview, `test_body_export` GLB hash-bound, and two `test_phase4_app`). `src/` is
untouched.

The report frames are in `artifacts/compare/d4c-start/report/`: 25 JPEGs, 480 px wide, q40, every 6th frame, camera
A001. The two panels are STACKED, not side by side: the D4 fit (aqua) sits above the D4c fit (blue) over the SAM2 mask outline, and the full-rate
`d4c-start-A001.mp4` is beside them. They are rendered by D4's own `d4_report_frames.py`, imported and re-pointed
(log 23). The extractor stub is `tools/compare/extractors/d4c_start.py`, and the registry is not edited.
