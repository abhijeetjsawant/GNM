# Astra GPT6 merge review of D4 — 2026-09-22, one round (the 2026-09-15 rule). Verdict: NO MERGE at e209eea

Verified: `d4_body_gate.py:291` returns MERGE when O1 alone fails (the coordinator's "recorded exception" — an override
under CLAUDE.md's rule, and D3's precedent supplies no pre-registered alternative rule); `d4_silhouette_paired.py:42/50`
stop at the available frames and drop nonfinite pairs, so a 15-frame truncation still reads B1 PASS; the frozen-pose
control, the closure and the mean-body must-fail are printed, not wired into the conjunction (`:165`, `:291`);
the shrinkage attribution overreaches (several drawn channels lack an explicit limit weight; the evaluator has zero penalty
inside the interval); the SMPL-X/MAMMA-posed control beats the candidate on performer 1 in four cameras.

| # | finding | change |
|---|---|---|
| 1 | O1's "recorded exception" is an override | withdrawn. D4's ACCEPTANCE is recorded as FAIL on O1 (1.030 vs 1 mm); the implementation merges as OPT-IN (`--body mhr`, default `rig` unchanged) with acceptance open; a NEW prospective registration (D4b) freezes O1's statistic, calibrated baseline, tolerance, identifiable channels and mean-body rejection before fresh fixtures |
| 2 | the gate accepts a truncated population and unwired controls | the agent enforces the named population (150 frames × 4 cameras × 2 performers) and wires the frozen control, the closure and the mean-body must-fail into the conjunction; the gate reads NO MERGE on D4 acceptance |
| debt | the default flip is its own gated integration step (compositor, schema-aware checks, instrument compatibility, an end-to-end rebuild of the MHR output); the shrinkage attribution reworded as measured, cause not established; "four cameras" | recorded |

---

**NO MERGE at e209eea under the adopted D4 card. Two blockers.**

1. **O1 fails a required conjunct; the exception is an override.** I reproduced **1.0300 mm > 1 mm** from the retained truth/fitted arrays. The adopted rule explicitly requires O1, while the gate explicitly returns merge when O1 alone fails. Keeping the numerical band unchanged does not preserve the rule when its failure stops blocking acceptance. [Adopted conjunction](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/docs/reviews/body-model-astra-review-2026-09-21.md:18), [override branch](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/tools/compare/d4_body_gate.py:291), [“never merge on an override”](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/CLAUDE.md:252).

   D3 establishes that an exception happened historically; it does not supply D4 with a pre-registered alternative acceptance rule. Nor does this measurement establish an impossible band: the reported truth-identity baseline tops out at **0.7568 mm**, below 1 mm. Recording the decision before B1 was measured does not make it prospective with respect to the already-failed O1. [Decision record](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/ladder-status.json:850).

2. **The gate accepts inputs that violate mandatory card clauses.** My B1 attack truncated the consumed candidate and reference mesh arrays from **150 to 15 frames**, in memory, then reran the actual rasteriser and bootstrap:

   | Performer | Cells scored | Difference | 95% CI |
   |---|---:|---:|---:|
   | 0 | 60 | +0.162613 | [+0.133736, +0.227601] |
   | 1 | 60 | +0.070144 | [+0.052252, +0.080116] |

   Feeding those newly computed results into the gate still produces **B1 PASS; MERGE with O1 a recorded exception**. The card requires the whole take, four cameras pooled. The producer silently stops at the available frames, drops nonfinite pairs, and the gate checks only the CI lower bounds. Enforce the named frame-camera population. [Truncation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/tools/compare/d4_silhouette_paired.py:42), [pair filtering](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/tools/compare/d4_silhouette_paired.py:50), [acceptance](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/tools/compare/d4_body_gate.py:177).

   The same incomplete conjunction omits required controls. Mutating the consumed frozen-control IoU **0.403360 → 1.0** changes its clause to FAIL but leaves merge permitted. Likewise, changing O1 closure to **0.01 m**, or one mean-body seed to **0 mm**, makes the respective required clause fail without blocking merge. Wire these required subclauses into acceptance; printing their failures is insufficient. [Control predicates](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/tools/compare/d4_body_gate.py:165), [final conjunction](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/tools/compare/d4_body_gate.py:291).

The smallest legitimate path is to preserve this D4 result as **FAIL**, repair those demonstrated gate holes, and establish any replacement O1 criterion through a **new, explicitly prospective registration**. Freeze its statistic, calibrated baseline, tolerance, identifiable channels and mean-body rejection before evaluating fresh held-out fixtures. Raising the threshold around the observed 1.030 is not that. An explicitly rescoped merge of opt-in implementation could be recorded separately, with D4 acceptance still open; retaining `rig` alone does not waive O1.

Everything else is **instrument debt or the follow-up disposition**, not another merge veto:

- **Keep `--body rig` as default.** The source already does. Remove the record’s promise to flip automatically at merge. Record the opt-in schema and artifact location, O1’s standing failure, and a separately gated integration/default-change step covering compositor consumption, schema-aware artifact checks, instrument compatibility and an end-to-end rebuild of the actual delivered output. The current close-out invokes the default builder and assumes eight rig files; its MHR readback failure is real. [Default](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/scripts/build_commercial_multiview_comparison.py:345), [close-out assumptions](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/tools/compare/post_merge.sh:18), [observed failure](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/artifacts/compare/d4-body/logs/22-b3-dvc-attempt.log:14).

- **The shrinkage is measured; its stated cause is not established.** The claim that spine length is the only drawn channel without an explicit limit weight is false: shoulder width, arm lengths and others also omit it. Moreover, the installed min/max evaluator has zero penalty inside the configured interval. Missing weight alone therefore does not prove the asserted shrink-to-zero mechanism. This weakens the attribution, without changing O1’s verdict. [Claim](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/docs/reviews/body-model-2026-09-21.md:122), [configured limits](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/.cache/mhr/assets/compact_v6_1.model:594), [installed evaluator](/tmp/momenv/lib/python3.12/site-packages/pymomentum/torch/parameter_limits.py:366).

- **Card interpretation:** B3–B5 remain report-only; the five facing misses add no veto. Neither does precision loss, the failed free-offset zero-identity prediction, or fitted MHR failing to beat its own mean body on performer 1. My card round explicitly retained those distinctions. One numerical correction: the SMPL-X/MAMMA-posed mean control exceeds the delivered candidate on performer 1 in **four cameras**, not three. [Card review](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/docs/reviews/body-model-astra-review-2026-09-21.md:38), [camera readings](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4/artifacts/compare/d4-body/logs/20-silhouette-delivery.log:23).

I independently reproduced all six B1 bootstrap intervals, B2–B5 subject results, both closure reports, all saved O1 seed readings, hygiene hashes, the gate and its mutation table. MAMMA’s entire recorded arm matches the committed report. These checks used retained meshes/GLBs/NPZs; I did not rerun fresh calibration/export or the four reported baseline test failures. No files were changed.