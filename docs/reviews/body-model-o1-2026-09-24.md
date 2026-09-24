# D4b: O1 re-registered prospectively. The measurement (2026-09-24)

**STOPPED at stage 3 under the registered (ii). D4 stays open. O1 is not superseded.**

The first merge round (Astra GPT6) ruled, and the coordinator adopted after checking the source, that must-fail
(ii) is **reading B**. The card requires the displaced spine to *"FAIL L at the trunk"*. L is defined over
SCORED segments, and *"a trunk that cannot be scored and rejected is not a demonstrated trunk failure."*

The drawn-set rule was frozen before any fit. It left `scale_spine_length` (and `scale_shoulder_width`) out, so
the trunk is not scored. The band as scored therefore **accepts** the displaced spine on **6 of 6** of D4's
burned fixtures. That is the registered STOP: *"If (ii) passes there, STOP before fresh fixtures."*

The gate now reads:

    BURNED VERDICT: FAIL   STOP: True    (the registered (ii) fails on D4's own cells)
    VERDICT: FAIL   STOP: True   D4: STAYS OPEN, O1 NOT SUPERSEDED  (fresh, POST-STOP EXPLORATORY)

D4's 1.030 mm FAIL stays on record.

**What happened after the STOP.** The executor implemented (ii) as reading A instead: the trunk's error against
its tolerance, whether or not the trunk is scored. Under that reading (ii) passed on D4's cells, so stages 4–6
ran. **Those fresh measurements are POST-STOP EXPLORATORY.** They are not registered acceptance evidence, and
fixtures 20261001–20261006 × donors 0/1 are now burned. They stay on disk and in this review, labelled as such.

The executor's reading-A computation read the conjunction as PASS with D4 STAYS OPEN. It is kept below (§2,
the second table) as a superseded computation, never a verdict.

**Read first, for the merge review.**

* **The registered (ii) (reading B)** is now what the gate implements. (ii) holds on a fixture only if the trunk
  is scored AND its error exceeds its tolerance there. An unscored trunk fails (ii) and raises STOP, on
  `--burned` as on the fresh population. The reading-A count is kept as a REPORTED quantity
  (`SUPERSEDED_reading_A_…`).
* **The cause is the drawn-set rule itself** (§4a). The card's first-order column-space test, implemented
  literally at the card's own 1e-6 rank convention, finds the along-spine displacement inside the pose Jacobian's
  span. It gets there through absorbing poses of 17–65 parameter units (±12 000 under plain `lstsq`) against
  limits of at most 1.5. A least-squares fit bounded by the configured limits was considered and rejected,
  because it would have changed a frozen rule after seeing its reading. Both readings are on record.
* **The executor's two superseded choices:** reading A for (ii), and printing the conjunction's PASS beside a
  STAYS OPEN disposition. §7 keeps both as they were decided, marked superseded.

Nothing under `src/` changed. `tools/fitter/mhr_delivery.py` is byte-identical at `3136befb…` at every commit.
The momentum settings in every banded arm are the pre-card's.

| stage | commit | what |
|---|---|---|
| 1 | `d6eb9b9` | `/tmp/momenv` rebuilt (pymomentum-cpu 0.1.114.post0). Provenance recorded. The environment reproduces D4's retained seed-20260922 cells byte for byte |
| 2 | `3ae1956` | the drawn set, frozen before any fit: **six**, not eight |
| 3 | `344bdc3` | BURNED: L and (ii) on D4's six retained cells. **Under the registered reading B, (ii) FAILS 6/6: STOP here.** The executor read it as A (no STOP) and went on |
| 4 | `1d9fe69` | POST-STOP EXPLORATORY: the fresh population, 12 fixtures × 6 arms, with closure on every oracle cell |
| 5 | `173d901` | the gate and its fuzz (reading A at that commit) |
| 6 | `3ad35d8`, `a39d64b` | the tests, this review and the extractor stub (reading A at those commits) |
| merge round | the reading-B commit | (ii) re-implemented as reading B, STOP raised on burned; fresh relabelled POST-STOP EXPLORATORY; test and fuzz updated; no new fits |

The committed records are in `docs/reviews/body-model-o1-records/`: `provenance.json`, `drawn-set.json`, `tripwire.json`,
`burned.json`, `fresh-manifest.json`, `gate.json` and `fuzz.json`. The logs are in `artifacts/compare/d4b-o1/logs/`.

---

## 1. The pre-registration, verbatim

The card is the `D4b O1 re-registered, prospectively` row of `docs/LADDER_EXECUTION_PLAN.md` §2. It is copied
unchanged below; the file `docs/reviews/body-model-o1-card-2026-09-24.md` is identical to that row.

| **D4b O1 re-registered, prospectively** | **Why.** D4's O1 read 1.030 mm against a 1 mm band written without the tracker's floor: FAIL, on record, never moved, never excepted. The smallest legitimate path (Astra, 2026-09-22) is a NEW registration frozen before fresh fixtures are evaluated. **What D4's readings showed (burned: they may inform the rule, never evidence it; `docs/reviews/body-model-2026-09-21.md` §2.1):** (1) the pooled statistic (median over frames of the median over the 17 joints) buries the one defect D4 measured. With the truth identity handed in, the tracker reads 0.18–1.64 mm per joint; the fitted identity adds 1.6–2.7 mm at `root` and 0.3–2.6 mm at `c_neck` while wrists, ankles and eyes stay within 0.35 mm of their floor. (2) `scale_spine_length` comes back 0.149–0.209 units short toward zero on six of six seeds (16.8–20.4 % of the draw); cause not established (the soft-limit attribution was refuted). (3) On this oracle a joint's distance to truth IS the fitter's own data-term residual (landmarks at pinned zero offsets sit on the joints), so a joint band is one the candidate optimises directly, and the pose absorbs identity error. The identity is the quantity the fitter never sees, so the band goes on it. **Scope: registration and measurement only.** `tools/fitter/mhr_delivery.py` byte-identical at sha256 `3136befb…` (the merged file at 285643c; D4's O1 cells record `8ff8d973…`, the O1-stage source before the reference-dump CLI was added, `fit_one` unchanged between them); the momentum settings stay the pre-card's (`calib_frames` 100, `loss_alpha` 2.0, `max_iter` 30, offsets pinned at `limit_weight` 10, the 26 `*_flexible` channels frozen). Nothing is tuned on the oracle that scores it, so hygiene, B1 and B2 do not reopen. A fitter change the probes point to is **D4c**, carded separately. `/tmp/momenv` must be rebuilt first (`tools/fitter/README.md`); the pymomentum version goes in the report. **The fixture: fresh, held out, every parameter declared here.** TWO donors, the pre-card's tracked MHR pose for performer 0 AND performer 1 (`artifacts/compare/d4-body/fitted/motion_{0,1}.npz`, sha256 recorded), each clamped to `compact_v6_1.model`'s configured `limit … minmax` entries (a declared fixture rule now, not a repair). Six NEW seeds, 20261001–20261006, per donor: twelve fixtures, each with its OWN identity draw (the generator is seeded by the pair (seed, donor index), so no two fixtures share a draw). Identity drawn uniform within the configured limits on the drawn set, every other identity channel zero. Truth landmarks are the 17 mapped locators at pinned zero offsets; the fitter receives landmarks only; one process per cell (the `calibrate_markers` corruption). **The drawn set, by a frozen rule computed before any fit:** take each of D4's ten named channels to either end of its configured limit on the truth body, pose held, at every donor frame, and read the largest displacement of any of the 17 mapped joints (a) raw and (b) POSE-ORTHOGONAL, i.e. the least-squares residual after projecting that displacement onto the pose Jacobian's column space at that frame (flexible channels excluded). A channel is drawn iff (b)'s median over frames exceeds **2 mm** on both donors, above every per-joint pose floor D4 measured (max 1.64 mm). Predicted: eight drawn. `scale_foot_length` is excluded as UNSEEN (no toe landmark; `l_foot`/`r_foot` are the ankles) and `scale_hip_height` as UNDRAWABLE (configured limit is the point [0, 0]); two reasons, both in the report, neither in a band. The drawn set's column rank and condition number are reported (spine/neck is the suspect pair) as a local diagnostic, not a threshold: individually surviving columns need not be jointly identifiable. The evaluation identity (the zero identity), the endpoint aggregation (the max over the 17 joints) and the rank convention (singular values above 1e-6 of the largest) are frozen in the drawn-set JSON before any fresh evaluation. **If `scale_spine_length` falls below the rule, that is a finding about the INPUT (only `root` and `c_neck` bound the trunk; `Spine1` is not fed), not a pass for the fitter:** it is reported as such and handed to the integration step's `Spine1` feed, AND exclusion never skips must-fail (ii), which is unconditional. A trunk that cannot be scored and rejected is not a demonstrated trunk failure, and such a run cannot close D4. **What D4b's verdict does:** D4's 1.030 mm FAIL stays on record whatever D4b reads. PASS → D4's acceptance closes as "O1 superseded by D4b: PASS" and D4 is done. FAIL → D4 stays open, D4c is carded from the probes, and the default flip is not dispatched while D4's acceptance is open. Window 0. | 7 | **THE band, L: identity in physical units.** On each fixture, pose the truth body and the fitted body at MHR's zero pose with their identity channels RETAINED (forward kinematics with the identity set and every pose parameter zero; never the track's `rest_positions_z_up_m`, which `write_track` computes with ALL parameters zero and would read the mean body on every arm). For each segment between mapped joints that a drawn channel moves, read the rest length and its error \|fitted − truth\| in mm. The segments: trunk `root`→`c_neck`, `c_neck`→`c_head`, shoulder width `l_uparm`↔`r_uparm`, upper arms `*_uparm`→`*_lowarm` and forearms `*_lowarm`→`*_wrist` (×2), hip width `l_upleg`↔`r_upleg`, thighs `*_upleg`→`*_lowleg` and shins `*_lowleg`→`*_foot` (×2); a segment whose only channel is undrawn is reported, not scored. **The tolerance is a rule, not a number, from one principle: the identity fit may cost no more than the pose solve already costs.** Each segment's tolerance is the sum of its two endpoints' PAIRED floors, the per-joint median over frames of the `exact_identity` arm (truth identity handed in, pose re-solved) on the SAME fixture, because two endpoints each off by their floor can already misstate the length by that sum. This is a registered ENGINEERING tolerance motivated by the triangle inequality, which bounds an instantaneous length error by the instantaneous endpoint errors; the sum of two marginal medians is not a derived statistical bound. PASS iff every scored segment on all twelve fixtures is within its tolerance. The baseline is per fixture and paired, never D4's constant 0.7568: the floor is pose-dependent (the same worst frames on every seed), and a constant from D4's fixture on fresh ones is a cross-population denominator. **Validity, a precondition and never a pass:** the `exact_identity` arm's pooled statistic ≤ 1.0 mm on every fixture (D4 read 0.51–0.76). A degraded tracker inflates every tolerance, so a floor above this reads INVALID. INVALID leaves D4 open and never licenses dropping or replacing a fixture. The pooled ceiling does not bound every segment's tolerance; the spine control limits that at the trunk only, which is stated. **Must-fails, each required on every fixture:** (i) the mean body (identity held at zero, pose re-solved) misses L. (ii) **The band's resolution at D4's own finding:** the truth identity with `scale_spine_length` displaced 0.149 units toward zero (the rounded minimum of D4's measured shrink, 0.148777, in the channel's own units; through zero if the draw is smaller), held, pose re-solved, must FAIL L at the trunk. If it passes on any fixture, the band is blind to the defect it exists to see and the step STOPS. (iii) The `exact_identity` arm reads L = 0 and the gate must PASS it: a positive implementation control through the actual scorer (identity extraction, units, correspondence, polarity), which does not validate the tolerance itself. **Required beside L:** closure, the delivered GLB's FK = the track ≤ 1e-4 m (D3's band), on every oracle cell. **Reported, never banded:** J, per joint oracle − paired floor for all 17 joints (the fitter's own residual on this oracle; the pose absorbs identity, so J cannot carry a band); D4's pooled statistic, for continuity; the per-channel parameter error on the drawn set and the drift of every undrawn identity channel, listed by name (the complement of the drawn set among the 68). What a PASS does NOT establish: full identity recovery. Compensating channels can hold the scored lengths with a wrong parameter vector, undrawn channels (the eye-location channels among them) change anatomy outside the scored segments, and the locator offsets are a soft penalty, not an exact pin (D4 measured them at 0.0009 mm max). **The mechanism, measured and reported, never selecting (D4c's evidence):** WARM, the calibration started at the truth identity. Drift toward the same shrink is evidence that the objective pulls away from the truth (a prior, pose absorbing length, or the limits); holding is evidence the shrink lives in where the solve starts or stops. Neither is a causal verdict. CONVERGED, `max_iter` 300; 300 iterations do not prove convergence, and the iteration count reached is reported. Reading it is allowed; adopting it here would select a constant on the oracle that scores it. Plus sensitivity (b) above. **Burned first:** before any fresh fixture runs, L and must-fail (ii) read D4's six retained cells (`artifacts/compare/d4-body/o1/`), recorded as BURNED, not evidence: they prove the instrument on known data. If (ii) passes there, STOP before fresh fixtures. **Prediction, fixed before fresh numbers:** FAIL. The trunk segment fails on most fixtures (D4's shrink, carried by an unchanged fitter); limb segments PASS; every must-fail behaves as stated; WARM decides whether D4c changes the objective or the start. **Population, named and refused if short:** 12 fixtures × {oracle, exact_identity, mean_body, spine_displaced, warm, converged} × 150 frames × 17 joints. A missing, short or non-finite cell is FAIL. Every set (seeds, donors, arms, segments, channels) is checked by identity. Provenance is the sha256 of the fitter, both donors, the model file `compact_v6_1.model`, the loaded `lod2.fbx`, and the pymomentum version, never a path prefix (D4's `o1.json` points into the ladder-D4 worktree). Card reviewed by Astra GPT6 once, at medium effort (`docs/reviews/body-model-o1-astra-review-2026-09-24.md`): dispatchable, no blockers; every finding adopted into this text. **Gate wired, not printed:** `tools/compare/d4b_o1_gate.py` computes the verdict as the conjunction (validity AND L AND closure AND must-fails i–iii) and prints INVALID, PASS or FAIL. A mutation fuzz in the `d7c_gate_fuzz.py` pattern turns each conjunct by mutating its INPUTS. **Merge rule, fixed before numbers:** D4b merges tooling and reports (src byte-identical) on a completed, valid run WHATEVER it reads, and its verdict is D4's O1 verdict. |

---

## 2. The clause tables

**The registered reading, where the step STOPS.** These are D4's six burned cells (`burned.json`), which prove
the instrument and are never evidence:

| clause | predicted (the card) | measured | verdict |
|---|---|---|---|
| (ii) the displaced spine fails L at the trunk (reading B: the trunk scored and beyond tolerance) | fails on every fixture; STOP if it passes | the trunk is **not scored** (drawn-set rule), so the band as scored accepts the displaced spine on **6/6**. The trunk's own error, 14.87–14.89 mm against 1.64–2.11 mm, cannot fail an unscored L | **FAIL, STOP** |
| validity / (i) / (iii) | as the card states | validity 0.506–0.757 mm; (i) misses 6/6; (iii) reads 0 mm, 6/6 | PASS |
| L | FAIL | 3/6 pass on the scored set | FAIL |
| overall | FAIL | FAIL, STOP | FAIL |

**POST-STOP EXPLORATORY: the fresh population.** It ran only because of the executor's superseded reading A. It
is not registered acceptance evidence, and these fixtures are now burned. There are 12 fixtures (seeds
20261001–20261006 × donors 0, 1) × 6 arms × 150 frames × 17 joints. Every cell is present, finite and bound by
identity (`gate.json`, `fresh-manifest.json`).

Under reading B the fresh gate reads **VERDICT: FAIL, STOP: True**: (ii) fails on 12/12 because the trunk is not
scored. Every other conjunct reads as in the table below.

The table below is the **executor's superseded reading-A computation**, kept as it was computed. Its (ii) and
overall rows are **not** the registered verdict:

| clause | predicted (the card) | measured | verdict | prediction |
|---|---|---|---|---|
| drawn set | eight drawn; foot_length UNSEEN, hip_height UNDRAWABLE | **six**: neck_length, uparms, lowarms, hip_width, uplegs, lowlegs. foot_length UNSEEN and hip_height UNDRAWABLE as predicted. **spine_length and shoulder_width below the rule** | rule applied, frozen | **FAILED** |
| validity | exact_identity pooled ≤ 1.0 mm on every fixture | 0.545–0.813 mm, 12/12 | PASS | held |
| L (scored: neck_head, 4 arm, hip width, 4 leg segments) | FAIL overall: the trunk fails on most fixtures, the limbs pass | every scored segment within its paired tolerance, 12/12. The worst ratio is 0.55× tolerance; the tightest margins are 0.43–0.49 mm (r_shin, r_forearm) | **PASS** | **FAILED** (see §4) |
| L, the trunk | the trunk FAILS on most fixtures | reported, not scored: 1.15–14.94 mm against 1.84–2.38 mm; beyond tolerance on **9/12** | not scored | held (as a reported reading) |
| L, the limbs | PASS | PASS 12/12 | PASS | held |
| closure (the GLB's FK equals the track, ≤ 1e-4 m) | within band on every oracle cell | 1.87e-06 to 3.01e-06 m; 150 frames and 127 joints on each of 12 | PASS | held |
| (i) the mean body misses L | misses on every fixture | misses on 12/12; its closest fixture is 42× tolerance | PASS | held |
| (ii) (SUPERSEDED reading A: the trunk read whether or not it is scored) | fails on every fixture; STOP if it passes | 14.88 mm against 1.84–2.38 mm, beyond on 12/12. The band *as scored* passes it on 12/12, **so under the registered reading B: FAIL, STOP** | (reading A: PASS) **registered: FAIL** | FAILED |
| (iii) exact_identity reads L = 0 and PASSES | L = 0 and PASS | 0.000000 mm on every segment, PASS 12/12 | PASS | held |
| **overall** | **FAIL** | reading A: the conjunction read PASS, `d4_disposition` STAYS OPEN. **Registered (reading B): FAIL, STOP** | **FAIL** | held |

## 3. The burned reading (D4's six retained cells: not evidence)

The retained files were resolved by content, not by path. D4's `o1.json` paths point into the ladder-D4 worktree,
so each file was found by its basename and its sha256 recorded. The regenerated truth motion equals the retained
one. D4's pooled statistic reproduces exactly from the retained arrays, and a plain rerun of the seed-20260922
oracle equals the retained cell to 0.0.

* **L** on D4's oracle, over the frozen scored set: **3 of 6 pass**. The scored misses are neck_head (2.67 and
  3.31 mm against 2.28 and 2.49) and the upper arms, on the seeds where the spine is stretched. This is the
  neck and arm compensation D4 recorded. The **trunk**, reported: **14.8–20.9 mm against 1.64–2.11 mm, beyond on 6 of 6.**
* **(ii), registered (reading B): FAILS. STOP.** The trunk is not in the scored set, and the band as scored passes
  the displaced spine on 6/6. The trunk's own error is 14.87–14.89 mm against 1.64–2.11 mm. The executor's
  superseded reading A counted that error as a failure at the trunk, read no STOP, and went on to stages 4–6.
* (i) misses 6/6. (iii) reads 0.000000 mm and passes 6/6. Validity reads 0.506–0.757 mm.

## 4. The FAILED predictions, attributed

**(a) "Eight drawn": six.** The rule was applied exactly as the card states it (`d4b_identifiability.py`):

* the clamped donor pose with the zero identity;
* each of D4's ten channels taken to either end of its configured limit;
* the 17 mapped joints, raw and pose-orthogonal;
* the pose Jacobian by central differences over the 110 non-flexible pose parameters;
* the rank convention the card froze (singular values above 1e-6 of the largest);
* drawn iff the median over frames exceeds 2 mm on both donors.

Two channels the landmarks see fell below it: `scale_spine_length` (raw 108–110 mm) and `scale_shoulder_width`
(raw 20 mm). Both read **0.00 mm pose-orthogonal**.

**The card's pre-assigned cause for a spine exclusion ("only `root` and `c_neck` bound the trunk; `Spine1` is not
fed") is not what was measured.** The measured mechanism is that the card's test is *first-order* and blind to
step size:

* The 51 × 110 pose Jacobian has rank 36, with a clean spectral gap. 36 singular values sit at or above ~2e-5 of
  the largest. Below them are values of 1e-9 to 1e-11, which scale with the finite-difference step squared (the
  truncation error of genuinely null directions), then 1e-14 round-off.
* The weakest *genuine* directions (2e-5 to 1e-4) are the neck_twist/head_twist pair, with clavicle_rx and
  uparm_rz. They absorb the along-spine and shoulder-width displacements at first order.
* The pose steps that absorption needs are 17–65 parameter units (median over frames). The configured limits are
  at most 1.5.

A column-space membership test cannot see how far the pose would have to move. At coarser cuts (REPORTED, never
selecting) the spine is drawn: at 1e-3, eight channels are drawn. That cut is not the rule.

*Decided before the freeze and before any fit, and recorded in the drawn-set JSON:* the projection is U Uᵀ onto
the left singular vectors above the card's rank convention. A first run used `numpy.linalg.lstsq(rcond=None)`.
That call cuts at machine epsilon, so it also projects onto the truncation directions (the absorbing coefficients
reached ±12 000). It read the same six and is kept as `REPORTED_…_ARTEFACT`.

**(b) "Every must-fail behaves as stated": (ii) does not**, under the registered reading B, on the burned cells
and on the post-stop fresh cells alike. The attribution is (a): the trunk left the scored set.

*(Superseded, post-stop exploratory.)* Under reading A the fresh "L: FAIL" and "overall: FAIL" predictions also
failed, because the conjunction read PASS on a scored set without the trunk. The prediction's own reason, *"the
trunk segment fails on most fixtures"*, held as a reported reading: 9 of 12.

## 5. The fresh reading, beyond the table (POST-STOP EXPLORATORY: not registered evidence)

* **Pooled statistic** (D4's, for continuity): oracle 0.589–0.920; exact_identity 0.545–0.813; mean_body
  11.4–37.8; spine_displaced 0.88–2.34; WARM 0.54–0.78; CONVERGED 0.019–0.105 mm.
* **Per-channel parameter error on the drawn set** (oracle, max |error| over 12): neck_length 0.0137,
  uplegs 0.0082, uparms 0.0076, lowlegs 0.0024, hip_width 0.0011, lowarms 0.0003.
* **The spine is not held at zero.** The truth spine is 0 on every fresh fixture (undrawn), and the oracle moves it
  anyway, to −0.116 … +0.150. That is the trunk's 1.1–14.9 mm error. D4's defect on its own drawn spines was a
  pull *toward* zero. Here, with the truth *at* zero, the fit wanders *away* from it. Both readings fit a solve
  that stops before the spine is determined.
* **Drift of the undrawn channels** (oracle, max |value| over 12; the other 55 of the 62 undrawn channels read
  exactly 0): spine_length 0.1497, shoulder_width 0.0037, hip_depth 0.0033, hip_height 0.0012, knee_knock 0.0006,
  l_hands 0.0004, r_hands 0.0000.
* **J** (per-joint oracle median minus paired floor; the fitter's own residual; reported, never banded):
  `root` −0.02…+2.48 mm and `c_neck` −0.08…+0.87 mm. These are the trunk again. Every other joint is within
  −0.36…+0.47 mm.

## 6. The mechanism probes (D4c's evidence; reported, never selecting; the fresh column is POST-STOP EXPLORATORY)

| probe | fresh (12; spine truth 0) | burned (D4's six; spine drawn ±0.74–1.08) |
|---|---|---|
| D4's oracle | spine −0.116…+0.150; trunk beyond tolerance 9/12 | spine 0.149–0.209 short (16.8–20.4 %); trunk 14.8–20.9 mm |
| **WARM** (calibration started at the truth identity) | spine error −0.0010…+0.0034; trunk ≤ 0.34 mm; L passes 12/12 | spine error −0.0031…+0.0045; trunk ≤ 0.45 mm; L passes 6/6 |
| **CONVERGED** (max_iter 300) | spine error −0.037…+0.039; trunk 0.09–3.89 mm (beyond tolerance 5/12); L passes 12/12 | 0.006–0.047 on five seeds; **20260923 drifts to 0.445 short (trunk 44.4 mm, pooled 2.19 mm)** |
| calibration solves stopped at the cap | at 300: 424–469 of 1056 | at 30 (tripwire, 20260922): **973 of 1056**; at 300: 77–472 |

**The reading, not a causal verdict.** Started at the truth, the calibration **holds** it, to 0.001–0.005 units
on every fixture of both populations. Within 30 iterations of the truth, the objective does not visibly pull away
from it. That, together with the cap exhaustion below, justifies **investigating where the solve starts and where
it stops**. It does not exclude effects of the objective itself: a prior, pose absorbing length, or the limits
could still act on a longer or differently started solve. At the pre-card's max_iter 30, 92 % of calibration solves stop at the cap on the one fixture where it was logged (the tripwire, burned 20260922).
Ten times the iterations recovers most of the spine, but not monotonically: one burned seed walks further away.
And at 300, 40–44 % of solves still stop at the cap. 300 iterations do not prove convergence. The iteration count
reached is in `gate.json` (`CONVERGED_iterations`). Reading this is allowed. Adopting max_iter 300 here would
select a constant on the oracle that scores it, and it is not adopted.

## 7. Decisions an executor made, for the merge review (item one first)

1. **The drawn-set rule's quantity.** The card's "least-squares residual after projecting onto the pose
   Jacobian's column space" was implemented literally, at the card's own rank convention. The literal object
   admits unphysical absorbing poses (§4a), and it excluded the one channel the step exists to see. Replacing it
   with, for example, a least-squares fit bounded by the configured limits would have changed a frozen rule after
   seeing its reading, and that was not done. Both the reading and its mechanism are on record. Any different
   rule is a new registration.
2. **(ii) with the trunk unscored. SUPERSEDED: the merge round ruled reading B.** The executor chose (A), the
   trunk's error read against its tolerance whether or not the trunk is scored, arguing from *"exclusion never
   skips (ii)"* and *"such a run cannot close D4"*. The ruling is that L is defined over scored segments, so an
   unscored trunk cannot fail L. (ii) therefore fails, and the step STOPPED at stage 3. The gate now implements B;
   the reading-A count is reported only.
3. **Verdict against disposition. SUPERSEDED.** Under reading A the conjunction read PASS beside a STAYS OPEN
   disposition. Under reading B the conjunction itself reads FAIL with STOP, so that contradiction no longer
   arises on this run.
4. **Beyond the card's letter, BURNED and REPORTED.** WARM and CONVERGED were also run on D4's six truths. The
   spine is drawn only there, so only there can the probes see the shrink. On the fresh fixtures (spine truth 0)
   they can only see the wander.
5. **The undrawn spine's displacement direction.** With the truth at 0 there is no "toward zero". (ii) displaces
   it by −0.149, a direction declared in `d4b_identifiability.displaced_spine` before any fit.

## 8. What each instrument is blind to

* **L scores lengths, not the parameter vector.** Compensating channels can hold the scored lengths with a wrong
  identity; the eye channels change anatomy outside every segment. The locator offsets are a soft penalty (the
  maximum is reported per cell). A PASS on L is not full identity recovery.
* **L's scored set is exactly what the drawn-set rule lets it see.** Here that excludes the trunk and the shoulder
  width. The band as scored passes the displaced spine on 12/12. That is the blindness the card's disposition
  clause exists for.
* **The tolerance is an engineering rule, not a bound.** A sum of two marginal medians does not bound a median
  length error (Astra, finding 1). The tightest scored margins are 0.43–0.49 mm.
* **The validity ceiling does not bound every segment's tolerance.** The pooled 1 mm is a median over 17 joints.
  The spine control limits this at the trunk only.
* **The drawn-set rule is first-order** (§4a). It reads span membership, not reachability within the limits.
* **The fuzz proves that what the gate reads, it depends on.** It does not prove that the gate reads the right
  things. It is bounded: long numeric lists are sampled, two fixtures are walked member by member, and the other
  ten are walked at their top-level keys.
* **Every fixture is exact and noiseless**, with pinned zero offsets. It says nothing about the take's
  landmark-to-joint convention (lane H).

## 9. What is open

* **D4 stays open; O1 is not superseded.** D4b STOPPED at stage 3 under the registered (ii), and D4's 1.030 mm FAIL
  stands. The default flip is not dispatched while D4's acceptance is open. Fixtures 20261001–20261006 × donors
  0/1 are burned by the post-stop exploratory run, so a future registration needs new seeds.
* **Instrument debt: the closure binding by basename and hash.** The gate binds each closure pair to its cell by
  the GLB and track *basenames* and their sha256 over the files in the cell directory. It does not recompute the
  closure itself. Astra independently recomputed all 12 closures. The binding is recorded debt, not a gap in
  this step's reading.
* **D4c, carded separately, from these probes.** WARM holds the truth, the 30-iteration calibration stops at the
  cap on 92 % of solves (the one fixture logged), and 300 iterations mostly recovers the spine but not monotonically. That justifies
  investigating the solve's start and stop first; it does not exclude effects of the objective. A fitter change is D4c's to card and gate. It is not measured
  on this oracle and adopted here.
* **A new registration of the drawn set, or a trunk the fitter can score.** Either the rule's quantity changes
  prospectively (reachability within the configured limits, which at first order the column-space test cannot
  express), or the integration step's `Spine1` feed gives the trunk a landmark between `root` and `c_neck`. Then
  the trunk can be scored and rejected. The card hands the spine exclusion to that `Spine1` feed. The measured
  mechanism, weak pose directions, suggests the feed would help, but that is not shown.
* **`scale_shoulder_width`** fell below the rule by the same mechanism, which the card did not foresee. Reported,
  its segment error is ≤ 0.74 mm against 1.54–2.36 mm.
* **Merge-review items 1–3 of §7.**

## 10. Reproduce

```
uv venv /tmp/momenv --python 3.12 && uv pip install --python /tmp/momenv/bin/python pymomentum-cpu
/tmp/momenv/bin/python tools/fitter/d4b_identifiability.py --out artifacts/compare/d4b-o1/drawn-set.json
/tmp/momenv/bin/python tools/fitter/d4b_o1_fixture.py --burned-read --out artifacts/compare/d4b-o1/burned
/tmp/momenv/bin/python tools/fitter/d4b_o1_fixture.py --drive --burned --only 20260922:0 ... 20260927:0 \
    --arms spine_displaced warm converged --out artifacts/compare/d4b-o1/burned
/tmp/momenv/bin/python tools/fitter/d4b_o1_fixture.py --drive --burned --only 20260922:0 \
    --arms oracle tripwire_zero_start tripwire_debug --out artifacts/compare/d4b-o1/tripwire
/tmp/momenv/bin/python tools/fitter/d4b_o1_fixture.py --drive --out artifacts/compare/d4b-o1/fresh
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_glb_closure.py --pair SEED_dD=GLB,TRACK ... \
    --out artifacts/compare/d4b-o1/closure.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4b_o1_gate.py --burned --cells artifacts/compare/d4b-o1/burned \
    --out artifacts/compare/d4b-o1/burned.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4b_o1_gate.py --cells artifacts/compare/d4b-o1/fresh \
    --closure artifacts/compare/d4b-o1/closure.json --out artifacts/compare/d4b-o1/gate.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4b_o1_gate_fuzz.py --out artifacts/compare/d4b-o1/fuzz.json
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_d4b_o1.py
```

`tests/test_d4b_o1.py` is 18 passed, including an unscored trunk that must read (ii) FAIL and STOP. Its subjects are the projection and the rank convention on synthetic
matrices, the rule, the spine displacement, the tolerance arithmetic, and the gate on a synthetic 72-cell
population. On that population it covers PASS, INVALID, FAIL on a missing, short, non-finite or stray cell, STOP,
(iii), provenance and a shared draw, plus the fuzz's targeted cases. The extractor stub is
`tools/compare/extractors/d4b_o1.py`; the registry is not edited.
