# Astra GPT6 merge review of D4b (2026-09-24), one round at MEDIUM effort. Verdict: NOT MERGEABLE AS PROPOSED; one blocker, adopted

Verified against the source before adoption: `tools/compare/d4b_o1_gate.py`'s `spine_fails_trunk` tests `error_mm > tolerance_mm`
on the trunk whether or not the trunk is in the scored set (reading A); the card's (ii) says "must FAIL L at the trunk", L is
defined over scored segments, and the card adds "A trunk that cannot be scored and rejected is not a demonstrated trunk failure"
(reading B). The closure match is by basename plus the local file's hash, with the reported residual not bound to those hashes.

| # | finding | change |
|---|---|---|
| 1 | reading A substitutes a different predicate for the registered (ii); under B the scored band accepts the displaced spine 6/6 burned, so the registered disposition is STOP at stage 3 | ADOPTED (blocker). The gate is rewired to B, the tests and fuzz follow, the extractor and review lead with "STOPPED at stage 3", and stages 4–6 are kept as POST-STOP EXPLORATORY (applied in 322111a; the coordinator reran both gate lines: burned FAIL + STOP, fresh FAIL + STOP, byte-identical to the committed reports; 18 tests pass) (not acceptance evidence; those fixtures are now burned). The executor's reading-A PASS stays in the record as superseded |
| 2 | D4 staying open is not an override; a stopped run cannot close D4 either | recorded: D4 stays open, O1 not superseded |
| 3 | the identifiability rule was implemented faithfully but is not a feasibility test (limits); a 1e-3 cut may not be substituted retrospectively | debt, carried to D4c: a limit-aware identifiability rule registered prospectively, with trunk eligibility a prerequisite and STOP if absent |
| 4 | closure provenance: basename match, the residual not bound to content hashes (a path mutation still read PASS) | debt: Astra recomputed all 12 closures from the artifacts, every row exact; future closure reports carry and the gate verifies content hashes |
| 5 | what D4c must carry: all D4/D4b fixtures burned, a new untouched population, one frozen candidate (initialisation from landmarks only), frozen stopping and cap behaviour chosen from compute or declared development evidence, limit-aware identifiability, one conjunction, no tuning after unblinding | recorded for D4c's card; the review's "not in a prior" softened |

---

**Not mergeable as proposed. One blocker: the record claims a completed, card-compliant conjunction PASS after the registered spine control should have STOPPED the step.** D4 stays open; O1 is not superseded.

1. **The blocker is reading A of must-fail (ii). I rule B.** The deciding text is “must **FAIL L at the trunk**,” followed by “If it passes on any fixture … the step STOPS,” and “Burned first … If (ii) passes there, STOP before fresh fixtures.” L is explicitly defined over scored segments. The card does not register a second, report-only rejection predicate for this control. [Frozen card, line 158](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4b/docs/LADDER_EXECUTION_PLAN.md:158)

   The implementation substitutes `error > tolerance` without requiring the trunk to be scored. That is where the registered control becomes a different test. [Gate, line 434](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4b/tools/compare/d4b_o1_gate.py:434) My prior review explicitly said: “An unscored trunk is not a demonstrated trunk failure.” [Prior review, line 22](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4b/docs/reviews/body-model-o1-astra-review-2026-09-24.md:22)

   The scored band accepts the displaced spine on **6/6 burned fixtures**. Consequently, the registered disposition is **STOP at stage 3**. “Exclusion never skips (ii)” requires running and honoring that control; it does not authorize changing its predicate.

   **Required correction:** wire that interpretation into the gate and tests, and correct the reports/extractor. Strike stages 4–6 **from the registered acceptance evidence**, not from the historical record. Preserve their measurements as explicitly post-STOP, exploratory results; those fixtures are now burned. The current A-based PASS may remain documented as the executor’s superseded computation. Tooling and measurements can then be retained without claiming successful completion under the original merge rule. No new fit is needed to establish this blocker.

2. **D4 staying open is not an override.** The specific “such a run cannot close D4” clause governs the general PASS consequence. Closing D4 would override that prohibition. But the prohibition does **not** presuppose completion: a stopped run also cannot close D4. It therefore cannot justify continuing past the burned STOP.

   Under B, the apparent contradiction largely disappears: the scored-length subtest passes, but the registered control fails, so the conjunction does not pass. The pre-registration ambiguity is real debt, shared with my dispatch review; it does not turn the implemented predicate into the registered one.

Everything below is **instrument debt, not an additional merge blocker**.

3. **Identifiability: faithful to the frozen mathematical rule, inadequate as a feasibility test.** The central differences, zero evaluation identity, endpoint aggregation and SVD projection implement the frozen convention. Replacing `lstsq(rcond=None)` with the `1e-6` truncated column space before freeze is consistent with that convention; the implementation and frozen record entered at stage 2 and the implementation is unchanged thereafter. [Projection](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4b/tools/fitter/d4b_identifiability.py:94), [freeze record construction](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4b/tools/fitter/d4b_identifiability.py:340)

   The six-channel result is not permission to substitute `1e-3` retrospectively. Conversely, membership in an unconstrained tangent space does not establish attainable compensation within configured limits. The added `0.1 mm` segment-movement convention and rounding before selection deserve explicit prospective treatment; the recorded values show no boundary-sensitive outcome here.

4. **The implementation repairs population truncation and most conjunct wiring, but retains a closure-provenance hole.** Exact array dimensions, named populations and finite values are checked; rest FK retains identity. The closure gate checks both frame counts, so the closure calculator’s internal `min(frames)` cannot conceal a shortened recorded cell. [Population checks](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4b/tools/compare/d4b_o1_gate.py:208), [rest FK](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4b/tools/fitter/d4b_o1_fixture.py:287)

   However, closure report paths are matched only by basename. Local artifact hashes match the cell, but nothing binds the report’s measured residual to those hashes. I changed all closure paths to nonexistent directories while retaining basenames: **PASS remained**. A missing cell folder failing does not test this substitution. [Deciding comparison, line 411](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4b/tools/compare/d4b_o1_gate.py:411)

   This is debt for this run because I independently recomputed **all 12 closures from the actual artifacts**, reproducing every report row exactly, all within band. Future closure reports should carry—and the gate should verify—the measured files’ content hashes.

   The fuzz proves dependence on its chosen inputs, not semantic correctness. Its bounded sampling and “some mutation changes the verdict” classification cannot certify reading A or catch every same-basename substitution. [Fuzz classification](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4b/tools/compare/d4b_o1_gate_fuzz.py:250)

5. **D4c must register selection separately from evaluation.** Its card should carry:

   - All inspected D4/D4b fixtures declared development/burned; a new, untouched acceptance population, fixed before evaluation.
   - One frozen candidate implementation: initialization from inference-available landmarks only; any multistart generation and selection rule specified without truth identity or oracle length error.
   - Frozen stopping criteria, iteration budget, tolerances and cap-exhaustion behavior. Choose these from computational constraints or separately declared development evidence—not by choosing whichever setting passes the acceptance oracle.
   - A prospectively defined limit-aware identifiability calculation, including bounds relative to each pose, numerical conventions and nonlinear validation limits. **Trunk eligibility must be a prerequisite**, with STOP if absent; it must never silently remove the target defect from acceptance.
   - One unambiguous conjunction, paired-baseline policy, unconditional trunk rejection through the actual scorer, burned-first STOP, full population/provenance checks and closure.
   - No tuning after unblinding. Any resulting change requires another registration and untouched evaluation set.

   WARM and cap exhaustion justify investigating initialization/stopping; they do not exclude objective effects. The report’s “so … not in a prior” inference exceeds those probes. [Report, line 153](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4b/docs/reviews/body-model-o1-2026-09-24.md:153)

Verification: **17 tests passed**, gate output reproduced byte-identically, and all 12 closure rows independently reproduced. I did not rerun calibration or the full identifiability sweep.