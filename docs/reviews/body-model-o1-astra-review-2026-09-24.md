# Astra GPT6 card review of D4b (2026-09-24), one round at MEDIUM effort (the user's choice). Verdict: DISPATCHABLE, no blockers

Verified against the source before adoption: `mhr_delivery.py` hashes to `3136befb…` at 285643c and today, while D4's O1 cells
record `8ff8d973…`; `write_track` computes `rest` with ALL parameters zero (`mhr_delivery.py`, `rest = ...np.zeros(len(names))`);
the spine shrink is 17.46 / 19.39 / 17.75 / 20.41 / 20.29 / 16.80 % of the draw; `load_character` reads `lod{N}.fbx` as well as
`compact_v6_1.model`. Every finding was debt-level and adopted into the card's text:

| # | finding | change |
|---|---|---|
| 1 | a sum of marginal medians is not a derived bound | tolerance restated as a registered engineering tolerance motivated by the triangle inequality |
| 2 | L does not establish full identity recovery (compensating channels, eye channels, soft offsets) | "what a PASS does NOT establish" added |
| 3 | spine exclusion must not skip the spine control | (ii) unconditional; an unscored trunk cannot close D4; rank/aggregation/evaluation identity frozen in the drawn-set JSON |
| 4 | `rest_positions_z_up_m` is the mean body on every arm | L reads FK with identity retained and pose zeroed |
| 5 | INVALID must not license dropping fixtures; pooled ceiling does not bound every segment | stated |
| 6 | same seed on two donors = same draw; hash the FBX; stale fitter hash | draws seeded by (seed, donor); lod2.fbx hashed; `3136befb…` |
| 7 | shrink 16.8–20.4 %, 0.149 is the rounded minimum 0.148777, "58" should be the named complement; probes are diagnostic, not causal | corrected |

---

**Dispatchable. No card-level blockers.** This authorizes the registered measurement, not D4 acceptance or the default flip.

The deciding clause is **“Must-fails, each required on every fixture”**, including rejection of the displaced spine **at the trunk**, followed by the explicit conjunction of validity, L, closure and all controls. Spine exclusion cannot waive that conjunct. An unscored trunk is not a demonstrated trunk failure. [Card, line 158](/Users/abhi_macbook/Projects/apps/autoanim/docs/LADDER_EXECUTION_PLAN.md:158)

Everything below is **instrument debt**, with implementation compliance reserved for the one merge review.

1. **Prospectiveness: legitimate principle, with an overstated mathematical justification.** Learning the rule from burned D4 observations is compatible with testing it prospectively on untouched fixtures. The endpoint-floor rule is not numerically equivalent to raising 1.030’s threshold: it changes the quantity, pairs the baseline and requires rejection of the known defect.

   An exactly recovered identity reads **L = 0 on every segment**, regardless of the pose tracker’s residual. It passes L; overall acceptance still requires validity, closure and controls. However, the triangle inequality bounds instantaneous length error by instantaneous endpoint errors. **The sum of two marginal medians is not generally a bound on median length error.** Treat this as a registered engineering tolerance, not a derived statistical guarantee. [Tolerance and controls](/Users/abhi_macbook/Projects/apps/autoanim/docs/LADDER_EXECUTION_PLAN.md:158)

2. **Quantity: sound for the named rest lengths; insufficient to establish full identity recovery.** Removing pose from the measurement prevents pose compensation from directly reducing L. Being an optimized residual would not itself disqualify J; its demonstrated insensitivity to identity is the stronger reason to replace it.

   Compensating channels can preserve measured lengths without recovering the parameter vector. Undrawn channels can change anatomy outside those lengths: eye-location channels are an explicit example. Locator offsets cannot directly repair an identity-only joint-length score, but can still affect calibration. “Pinned” means a soft penalty, not an exact constraint; D4 measured offsets as small as stated. These limitations narrow what PASS establishes. [Eye channels](/Users/abhi_macbook/Projects/apps/autoanim/.cache/mhr/assets/compact_v6_1.model:284), [locator construction](/Users/abhi_macbook/Projects/apps/autoanim/tools/fitter/mhr_delivery.py:90)

3. **Drawn set: no requirement to force an unidentifiable spine into the recovery band.** The input-finding sentence alone would not prevent an exclusion-assisted PASS. The unconditional spine control does: exclusion must not become “skip control.” If the trunk cannot be scored and rejected, the run cannot close D4.

   The sensitivity calculation remains a local identifiability diagnostic. Individually surviving columns need not be jointly identifiable. Freeze the evaluation identity, endpoint aggregation and numerical rank convention before fresh evaluation; report rank and conditioning without inventing another acceptance threshold. [Drawn-set rule and unconditional control](/Users/abhi_macbook/Projects/apps/autoanim/docs/LADDER_EXECUTION_PLAN.md:158)

4. **Controls: (ii) is a real resolution test; (iii) is a positive implementation control.** A known identity displacement tests whether L and its paired tolerance can see the target defect. Re-solving pose does not change its identity-only L. The burned-first STOP and fresh-fixture STOP are correctly required **in the card**; no implemented D4b gate is present here to certify their wiring.

   Exact identity yielding zero is mathematically tautological, but passing its output through the actual scorer tests identity extraction, units, correspondence and gate polarity. It does not independently validate the tolerance.

   **Concrete implementation trap:** saved `rest_positions_z_up_m` is computed with *all parameters zero*, including identity. Using that field for each arm would give the same mean-body rest skeleton. Recompute FK with identity retained and pose zeroed. [Writer](/Users/abhi_macbook/Projects/apps/autoanim/tools/fitter/mhr_delivery.py:249)

5. **Validity: useful, but the pooled ceiling does not bound every segment tolerance.** Several endpoint floors can grow while the median over 17 joints remains below 1 mm. The spine control limits that vulnerability at the trunk; it does not calibrate every limb segment’s resolution.

   The 1 mm ceiling precedes the fresh floors, but follows measured burned floors. That is legitimate prospective registration, not an empirically established universal tracker limit. Above it, INVALID must leave D4 open; it cannot permit dropping or replacing fixtures. [Validity and population clauses](/Users/abhi_macbook/Projects/apps/autoanim/docs/LADDER_EXECUTION_PLAN.md:158)

6. **Constants, population and provenance.** The mandatory mean-body rejection prevents that constant from earning overall PASS. It does **not** prove rejection of every possible nonzero constant identity. Nor do the six seeds on two donors necessarily provide twelve distinct identity draws.

   The named population, identity checks and input mutations address D4’s documented truncation and unwired-control holes at specification level. Their implementation remains unverified. Hash the actual FBX asset too: loading depends on it as well as `compact_v6_1.model`. [Model loading](/Users/abhi_macbook/Projects/apps/autoanim/tools/fitter/mhr_delivery.py:81)

   The card’s fitter hash is stale. **`8ff8d973…` identifies the retained O1-stage source; both `285643c` and today’s file hash to `3136befb…`.** The intervening change adds the reference-dump CLI path; `fit_one` is unchanged. Correct the provenance label rather than reverting the merged file. [Retained hash](/Users/abhi_macbook/Projects/apps/autoanim/artifacts/compare/d4-body/o1/o1.json:288)

7. **Numerical reproduction and attribution.** I reproduced all four reported statistics and all per-joint medians for **all 18 retained cells**, each finite and shaped 150 × 17. The quoted floor and trunk-excess ranges agree after rounding.

   Two corrections:
   
   - Spine shrink is **16.80–20.41% of the draw**, not 14–20%.
   - The smallest recorded shrink is **0.148777**, rounded to the registered **0.149**. Call it the rounded minimum. [Retained minimum](/Users/abhi_macbook/Projects/apps/autoanim/artifacts/compare/d4-body/o1/o1.json:1980)

   “58 undrawn” is also a stale count: with eight drawn out of 68, the complement is **60**; use the actual named complement.

   I have **not** reproduced L, the displaced-spine control, sensitivity, rank or the eight-channel prediction. `/tmp/momenv` is absent; those measurements remain dispatch work. Finally, WARM drifting does not prove where the objective’s minimum lies, and 300 iterations does not prove convergence. Both are diagnostic evidence for D4c, not established causal verdicts. [Probe claims](/Users/abhi_macbook/Projects/apps/autoanim/docs/LADDER_EXECUTION_PLAN.md:158)