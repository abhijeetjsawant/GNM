| **D4d a second calibration pass on the landmark start, decided by where the truth-started fit settles** | **Why.** D4's O1 is still open. D4c (`docs/reviews/body-model-start-2026-09-25.md`, FAIL on L) started the calibration from the landmark-derived identity and passed 11 of 12 fresh bodies. The miss was a strongly shortened spine (draw −0.911): the p90 chord started it 0.032 units LONG, and 30 iterations removed about a third (0.011), leaving the trunk at 1.10× its tolerance. All four misses across D4c (3 development, 1 acceptance) were strongly negative spine draws, a class under-sampled at every stage: D4b's 12 fixtures never drew the spine, and D4's 6 used a different channel set. Recovery from far starts removed about 80 % (the zero start), and from near starts only about a third, a geometric pattern that more effort from the same start should close, UNLESS the objective's minimum is off the truth there. D4b's CONVERGED probe found one wrong basin at 300 iterations, from the zero start and with tracking's budget raised too. **Base.** D4c's change is NOT on main. `ladder/D4d` branches from main and merges tag `ladder/D4c-fail-1a89cc7` as its first commit, so the candidate is D4c's landmark start PLUS this step's change. Attribution is to the implementation only: two passes do not separate extra iterations from the restart and stage effects, and a PASS does not show that cap exhaustion caused D4c's miss. On PASS both merge together and D4c's deferred tooling merge is discharged. On any non-PASS, RECORDS ONLY go to main and the branch is pinned by tag (registered now, per the D4c rule in CLAUDE.md). **Phase 1, the diagnostic, on burned fixtures only.** Primarily D4c's 12 (the current 8-channel draw); D4's 6 and D4b's 12 are reported. Every arm uses the frozen D4c start (p90 trunk), and stages A and B are read separately:
- **WARM:** calibration started at the truth identity. Truth-seeded: it DECIDES the fork below and never selects a value.
- **TWO-PASS:** the full calibration (stage A then B) run a second time, started from the first pass's identity, each pass at `max_iter` 30. `tracking.max_iter` stays 30 and is untouched by the second pass, so the shared cap is not raised. The pass count is fixed at 2 and never swept.
- **SW*:** only the shoulder-width start set to the truth. Truth-seeded, a report on the coupling hypothesis D4c recorded.

For each, the start error and the recovered error on every drawn channel are reported against the drawn spine value, with a spine-tercile breakdown. **The decision rule, frozen:**
- **STOP, and D4e takes the objective,** if WARM leaves the trunk beyond its paired tolerance on ANY of D4c's 12. That shows drift under this procedure; it does not prove the objective's minimum is off the truth, or that no other start or budget could succeed, so "the basin" is D4e's HYPOTHESIS, not a finding.
- **The candidate is TWO-PASS** iff WARM holds on all 12 AND two-pass brings every scored segment within tolerance on all 12.
- **Otherwise STOP, reported:** WARM holds but two-pass does not close the gap. SW* is then the lead for a landmark-derived shoulder-width start, which would need its own registration.

No option is invented after reading. **Phase 2, acceptance, only if the rule selects TWO-PASS:** untouched seeds 20261201–20261206 × donors 0 and 1 (12 fixtures). Draws are uniform as before; the draw is not re-stratified, which would change the fixture distribution, but the results are REPORTED by spine tercile so the failing class is visible. The real take is rebuilt through `--body mhr` with the combined fitter. **What D4d's verdict does:** PASS → D4's O1 is superseded by the combined fitter, with B1 re-shown on it; D4, D4c and D4d close. Otherwise D4 stays open. A WARM-drift STOP hands the objective to D4e; the alternatives are lane H's markers, a `Spine1` feed (an input change needing its own scope) or a licensed body. **Stated blindness:** the start is a pose-dependent estimator and only two donor motions exist, so "held-out" means the identity, never the pose (lane H). **Instrument debt repaired here, because the tooling is touched anyway:** the fuzz separates CRASH from ENFORCED (D4c counted a crash on a mutated input as enforcement), and the gate re-derives the tripwire's normalised equality and B2's checks from the files instead of reading the producers' booleans. Window 0. | 7 | **Everything from D4c, carried verbatim:**
- **Precondition 0:** D4c's frozen limit-aware drawn set, reused by sha256; the trunk must be drawn, else STOP.
- **Stage 0b:** the spine control fails L as scored on every Phase-1 fixture, else STOP.
- **THE band, L:** every scored segment, the trunk and shoulder width included, within the sum of its two endpoints' paired per-fixture floors on ALL 12 acceptance fixtures. The band and the tolerance rule are unchanged; 1.10× on one body of twelve stays a FAIL.
- **Validity:** the exact arm ≤ 1.0 mm on every fixture, else INVALID.
- **Must-fails through the actual scorer, as scored, on every fixture:**
  - (i) the mean body misses L;
  - (ii) the spine displaced 0.149 units misses L at the trunk;
  - (iii) the exact identity reads 0 and passes;
  - (iv) init-only (the D4c start held, no calibration) misses L. If it passes on any fixture, STOP.
- **Closure:** ≤ 1e-4 m, bound by sha256 to the files measured.
- **B1 on the rebuilt `--body mhr` (D4's band):** combined-fitter minus the D7c rig, the lower CI above zero on both performers. D4's frozen-pose and MAMMA-unchanged conjuncts are inherited, the inputs are sha-bound, and the population is named and refused if short. Combined minus D4c-fitted is REPORTED.
- **B2 same denominator.**
- **Hygiene:** `--body rig` rebuilds D7c 8/8. The tripwire: passes = 1 through the new code reproduces D4c's pinned delivery and D4c's acceptance cells byte-identical. The source diff: `fit_one` changes ONLY by the second calibration pass, which gets its own copy of the first pass's identity.

**Reported, never banded:**
- the legacy zero start and the one-pass D4c start on the acceptance fixtures;
- cap exhaustion per stage and per pass;
- J per joint;
- whether the acceptance fixtures' allowable rest-length intervals have an empty intersection on any scored segment (no common constant body).

**Prediction, fixed before numbers:**
- Phase 1: WARM holds on 12/12 (D4b's WARM held on 18 at ≤ 0.45 mm), and two-pass brings D4c's 1.10× below 1 with every segment within tolerance on all 12, so TWO-PASS is selected.
- Phase 2: PASS, the worst trunk on a strongly negative spine.
- B1 unchanged within ±0.01 of D4c.

**Falsifier:** WARM drifts on 20261106/d0 → the basin → STOP.

**Population, named and refused if short:**
- Phase 1: 30 burned fixtures × {D4c start, WARM, TWO-PASS, SW*} × 150 × 17, PLUS each fixture's supporting cells: the exact-identity floor (the paired tolerances) and the displaced-spine control (stage 0b). Retained cells are bound by content hash and fixture identity; none is omitted silently.
- Phase 2: 12 × {candidate, exact_identity, mean_body, spine_displaced, init_only, one-pass, legacy} × 150 × 17.

A missing, short or non-finite cell is FAIL. Every set is checked by identity, and the gate VERIFIES constructions: it regenerates the seeded draws, recomputes init-only, and binds hashes to files. The Phase-1 decision JSON is committed before any Phase-2 fixture exists, and the gate checks the order. The Phase-2 generator REFUSES to run without the committed decision JSON and writes its hash into every cell (commit ancestry alone proves only record order). Provenance: the fitter, both donors, `compact_v6_1.model`, `lod2.fbx`, the pymomentum version, the D4c drawn-set and development JSONs, and the Phase-1 decision JSON.

Card reviewed by Astra GPT6 once, at medium effort (`docs/reviews/body-model-twopass-astra-review-2026-09-25.md`): dispatchable, no blockers; every finding adopted into this text. **ONE verdict:** PASS iff precondition 0 ∧ stage 0b ∧ the rule selects TWO-PASS ∧ validity ∧ L ∧ closure ∧ must-fails i–iv ∧ B1 ∧ B2 ∧ hygiene; otherwise FAIL, INVALID or STOP. The gate is `tools/compare/d4d_twopass_gate.py`, with a per-conjunct input-mutation fuzz.

**Merge rule, fixed before numbers:** PASS → everything merges (D4c's pinned change and tooling with this step's), `--body mhr` carries the combined fitter, and the default stays `rig`. Otherwise records only on main, and the branch is pinned by tag. |
