# Astra GPT6 card review of D4c (2026-09-25), one round at MEDIUM effort. Verdict: DISPATCHABLE, no blockers

Verified against the source before adoption:
- `d4b_identifiability.py` records the largest ABSOLUTE rest-length change at either limit end;
- hip width reads 100 mm at its 0.5 limit (200 mm per unit), and the two-sided channels move both sides;
- `fit_one` shares `max_iter` between calibration and tracking (`tracking.max_iter = max_iter`);
- both `calibrate_markers` calls start from `zero.copy()`.

Every finding was debt-level and adopted into the card's text:

| # | finding | change |
|---|---|---|
| 1 | development selection legitimate; freeze its algorithm fully | the numerator is post-calibration; calib-frame indices, numpy `linear` percentiles and the tie order median > p90 > p95 are frozen before any reading |
| 2 | bounded linear LSQ suffices as a screening rule, not as an identifiability conclusion | D4b's full aggregation stated; sign, pose-column, bounds and solver-failure (STOP) conventions frozen; a STOP reads "failed this rule", not "unidentifiable" |
| 3 | init-only bypasses both stages; a pass means calibration is unnecessary there, not inert | reworded; the STOP is kept as a conservative consequence, and a start-only delivery would need its own registration |
| 4 | B1 must inherit D4's frozen-pose and MAMMA-unchanged conjuncts and bind to the rebuilt delivery; D4c − D4 is not a non-regression band | adopted, and stated |
| 5 | the tripwire is regression evidence, not attribution | the source diff is required: only the two starting identities change, with independent copies |
| 6 | the gate must verify constructions, not labels | the gate regenerates the seeded draws, recomputes init-only from the landmarks and binds closure hashes to the files; an empty-intersection constant-body check is reported |
| 7 | the cap figures are stage B only; the retained table is absolute; hip width is 200 mm per unit; the falsifier is a hypothesis; `max_iter` is shared with tracking | corrected |

---

**Dispatchable. No dispatch blockers.** The deciding clauses are “If `scale_spine_length` is not drawn, the step STOPS,” the single conjunction including B1/B2, and “Otherwise … `fit_one` does not [merge].” These prevent D4b’s unscored-trunk PASS and prevent a failed candidate from changing the opt-in delivery. [Card, line 1](/Users/abhi_macbook/Projects/apps/autoanim/docs/reviews/body-model-start-card-2026-09-25.md:1)

Everything below is **instrument debt, not an additional dispatch condition**.

1. **Development selection is legitimate; its algorithm needs a complete freeze.** Selecting among three declared statistics on 18 explicitly burned fixtures, then evaluating once on untouched identities, is separately declared development evidence. The small candidate set helps; separation from acceptance is what makes it legitimate. This meets D4b item 5’s requirement. [Deciding lines 48–53](/Users/abhi_macbook/Projects/apps/autoanim/docs/reviews/body-model-o1-astra-merge-review-2026-09-24.md:48)

   Record whether the selection numerator is **post-calibration** trunk error—I read it that way—the calibration-frame indices, percentile interpolation, and complete tie ordering. “Ties to median” does not resolve a p90/p95 tie when both beat median. Freeze these implementation conventions before their readings can influence the choice. Sharing the same two donor motions limits generalisation to new identities on those motions; it is not an untouched-pose experiment.

2. **Bounded linear least squares is sufficient for this registered screening rule, not for a nonlinear identifiability conclusion.** I would not require the nonlinear tracker as the eligibility test: its optimisation failures would introduce another confound. Large, box-feasible angular steps can nevertheless have fictitious first-order effects. Thus even a correctly implemented bounded rule could drop the spine. STOP remains sound; “the trunk is not identifiable from these 17 landmarks” does not follow.

   Preserve D4b’s full aggregation: maximum joint norm, then maximum over the two limit ends, then median over frames. The new card names only the joint maximum explicitly. Also freeze the Jacobian/displacement sign convention, pose-column membership, treatment of missing or fixed bounds, and solver failures. Flexible spine channels must remain excluded: the source explicitly documents their duplication of identity. [Aggregation](/Users/abhi_macbook/Projects/apps/autoanim/tools/fitter/d4b_identifiability.py:9), [flexible-channel mechanism](/Users/abhi_macbook/Projects/apps/autoanim/tools/fitter/mhr_delivery.py:74)

3. **Init-only is a real ablation, but its consequence is deliberately stricter than product correctness.** Candidate PASS plus init-only MISS demonstrates that calibration is necessary to cross this band on each fixture. It does not isolate identity calibration from stage A’s locator adjustment: init-only bypasses both calls. [Calibration branches](/Users/abhi_macbook/Projects/apps/autoanim/tools/fitter/mhr_delivery.py:157)

   Conversely, init-only PASS means calibration is **unnecessary for passing that fixture’s L**, not “inert.” Calibration may still change or improve the result. Requiring MISS on every fixture can STOP an excellent start because one fixture is easy. That is an acceptable conservative consequence for this calibration claim. It is not evidence that the start is defective; a start-only delivery would need its own registration.

4. **B1 and the merge rule are sound. There is no textual route to closing D4 without photographs seeing the closing fitter.** B1 explicitly scores the rebuilt D4c delivery, and non-PASS excludes the fitter change. The reported D4c−D4 CI is not a non-regression band: a substantial regression against D4 can still PASS if D4c beats D7c on both performers. That is the registered product choice. [Deciding card clauses](/Users/abhi_macbook/Projects/apps/autoanim/docs/reviews/body-model-start-card-2026-09-25.md:1)

   At merge review, verify that the silhouette inputs actually bind to that rebuilt delivery. Inherit D4’s required frozen-pose and unchanged-MAMMA checks when implementing “D4’s own B1”; the prior record explicitly calls both required conjuncts. [D4 record, lines 67–70](/Users/abhi_macbook/Projects/apps/autoanim/docs/reviews/body-model-2026-09-21.md:67)

5. **The zero-start tripwire is strong regression evidence, not sufficient attribution by itself.** It proves equivalence on the retained zero-start executions. A change conditional on a nonzero start could escape it. Attribution also needs the source diff showing that only the two starting identities changed, with copies passed independently. The existing WARM hook supplies exactly that precedent. [Both zero starts](/Users/abhi_macbook/Projects/apps/autoanim/tools/fitter/mhr_delivery.py:170), [WARM substitution](/Users/abhi_macbook/Projects/apps/autoanim/tools/fitter/d4b_o1_fixture.py:118)

6. **The population and closure clauses repair the known holes; they do not make the instrument constant-proof.** Rejecting the mean body rejects that constant, not every possible constant identity. A useful reported check is whether the fixtures’ allowable rest-length intervals have an empty intersection for any scored segment; that would rule out a common constant body.

   The gate must verify constructions, not just labels: regenerate seeded draws, recompute init-only identity from consumed landmarks and the frozen rule, and bind measured closure hashes to actual files. D4b checked generator labels, limits and distinctness without regenerating the random draw. Hash agreement likewise proves consistency, not that a claimed computation occurred. These are implementation-review obligations under the card’s existing identity/provenance clauses. [Existing draw checks](/Users/abhi_macbook/Projects/apps/autoanim/tools/compare/d4b_o1_gate.py:324)

7. **Retained numbers support the experiment, with corrections to interpretation.** I recomputed from retained cells:

   - WARM maximum trunk errors: **0.445353 mm burned**, **0.341382 mm D4b**.
   - CONVERGED worst burned spine error: **0.444691 units**, trunk **44.422874 mm**.
   - The quoted cap ranges reproduce **stage B only**. The 30-iteration tripwire also records stage A separately and maximum logged indices of 49, so “973/1056 calibration solves hit the cap” needs stage and inner-solver qualification. [Retained tripwire](/Users/abhi_macbook/Projects/apps/autoanim/artifacts/compare/d4b-o1/tripwire/cell-20260922-d0-tripwire_debug.json:1)

   The segment premise needs correction: bilateral channels move two named segments; **hip width also changes 200 mm/unit**—100 mm at its 0.5 limit. Moreover, the retained table contains maximum **absolute** changes over both ends, not signed upper-end slopes or a linearity check. Compute those afresh as carded. [Hip-width measurement](/Users/abhi_macbook/Projects/apps/autoanim/artifacts/compare/d4b-o1/drawn-set.json:729), [table construction](/Users/abhi_macbook/Projects/apps/autoanim/tools/fitter/d4b_identifiability.py:193)

   Finally, “near-truth start then failure ⇒ objective’s basin” is a hypothesis, not an identified cause. Finite stopping and stage coupling remain alternatives; CONVERGED also changed tracking’s iteration budget. [Shared iteration setting](/Users/abhi_macbook/Projects/apps/autoanim/tools/fitter/mhr_delivery.py:174)

I inspected source and recomputed retained-cell statistics; I did not rerun fitting, the identifiability sweep or B1’s bootstrap, and generated no acceptance fixtures.