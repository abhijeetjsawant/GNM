# Astra GPT6 merge review of D4d (2026-09-25), one round at MEDIUM effort. Verdict: MERGEABLE, PASS, no blockers

Verified against the source before adoption:
- `d4c_start_gate.py` reads the fitter source and its hash from the working tree (around line 549);
- `fit_one` defaults to `passes=1` while the CLI forwards a default of 2;
- the tripwire's acceptance compares file counts;
- the fuzz leaf walk marks ENFORCED when either mutation rejects.

| # | finding | change |
|---|---|---|
| 1 | "supersedes D4's O1" is honest only as the registered disposition; fresh acceptance cannot discriminate the candidate from D4c | Astra's permitted-claims text is written into the review, with the list of what may NOT be claimed |
| 2 | the real-take regressions (spine 1.102 past the 1.1 limit, residual +1 mm, B1 vs D4c CI below zero on performer 0) are debt, not blockers; opt-in shipping is acceptable | recorded; the real-take spine is stated as not evidence of accuracy |
| 3 | re-pin the two D4c tests to the tag's fitter (source AND sha), do not retire them | done in 09a81ed (a `history_ref` override was also needed, so the freeze order is read from the tag's own history); the help-text edit changed the fitter hash, so the D4d gate now accepts the cells' fitter only if it is code-identical once docstrings and help strings are removed, and a test proves a real code change is still rejected. The coordinator reran the gate (identical, PASS), both test files (34 pass) and the targeted fuzz against the new gate (54/54 turn, 0 crash) |
| 4 | the differing pass defaults are a caller hazard; `--zero-start` alone no longer reproduces D4; the pass count belongs in the fit report and track | the help text is fixed; the metadata move is recorded as debt (it would break the one-pass byte identity with D4c) |
| 5 | the tripwire checks only the file count; the leaf walk counts ENFORCED on either mutation (291 mixed leaves) | recorded as instrument debt |
| — | hygiene wording: 106 byte-identical + 110 equal after named normalisations | corrected in the review |

---

**Mergeable under the frozen card. Registered verdict: PASS. Merge blockers: none found. No INVALID or STOP condition is established.** The deciding rule is the conjunction at [card line 45](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4d/docs/reviews/body-model-twopass-card-2026-09-25.md:45); line 47 licenses merging the combined fitter behind `--body mhr`, with `rig` remaining default.

I independently reproduced `gate.json` exactly and reran the 54 targeted fuzz cases: all produced their expected verdicts, with zero crashes. The frozen card matches `fedd834`; `src/` and `scripts/` are unchanged. I relied on the coordinator’s 16-test result and inspected, rather than reran, the full leaf walk.

Everything below is **nonblocking debt or a limit on what the record may claim**.

1. **“Supersedes D4’s O1” is honest only as the registered disposition.** The card expressly registers uniform sampling and that consequence at [line 11](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4d/docs/reviews/body-model-twopass-card-2026-09-25.md:11). The record may say:

   > The combined fitter passes the registered acceptance band on twelve fresh identities under two retained donor motions, with B1 re-shown against D7c. It closes the four observed one-pass misses on burned fixtures. Fresh acceptance does not establish improvement over D4c or validate repair of its shortened-spine failure class.

   It may not claim fresh confirmation of that repair, superiority over D4c, general pose robustness, full identity recovery, or convergence. D4c’s historical FAIL remains a FAIL. The review’s opening caveat correctly preserves this distinction.

2. **The real-take regressions are debt, including the soft-limit excursion.** Neither `1.102 > 1.1`, increased landmark residual, nor a negative D4c-relative CI is a registered rejection condition. The deciding [B1 clause, line 22](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4d/docs/reviews/body-model-twopass-card-2026-09-25.md:22) bands improvement over **D7c** and explicitly reports the D4c comparison. Shipping opt-in is therefore acceptable under this card. Do not describe the second pass as non-regressing or the recovered real-take spine as more accurate; that take has no identity truth.

3. **Re-pin the two D4c tests; do not retire their historical checks.** Read the fitter from `ladder/D4c-fail-1a89cc7` at its pinned commit and verify its hash. For gate replay, supply both that source **and its SHA**, because [the loader currently reads both from the working tree](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4d/tools/compare/d4c_start_gate.py:549). Preserve the historical FAIL and its conjuncts. D4d’s tests should guard today’s fitter. Deleting the old checks would discard useful reproducibility; leaving them indefinitely red would obscure future failures.

4. **The differing defaults are a caller hazard, not a demonstrated delivery error.** Direct callers omitting `passes` retain one pass; the CLI explicitly forwards its default of two. See [function default](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4d/tools/fitter/mhr_delivery.py:252) and [CLI invocation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4d/tools/fitter/mhr_delivery.py:490). Existing historical fixture callers benefit from compatibility. New callers wanting the combined fitter must specify `passes=2` and the landmark start. Also, `--zero-start` alone no longer reproduces D4: it needs `--passes 1`, despite its current help text.

   Record requested and effective calibration passes in fit-report settings and track metadata, including zero effective passes for held-identity arms. The [current sidecar](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4d/tools/fitter/mhr_delivery.py:499) is easy to detach. Embedding metadata improves traceability; it does not independently prove execution.

5. **No observed population or ordering defect overturns this run, but enforcement has limits.** Supporting Phase-1 cells are content-bound; Phase 2 checks named arms, regenerated draws, decision hashes and manifest order. The generator invokes its committed-decision guard [before building truth](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4d/tools/fitter/d4d_fixture.py:265). This supports the prescribed workflow; it cannot prove that no alternate process previously generated those identities.

   Two concrete instrument debts remain:
   - The tripwire calls its sets “by identity,” but acceptance checks [only the file count](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4d/tools/compare/d4d_twopass_gate.py:136). Matching substitutions on both sides could evade that check. No such substitution was established here.
   - The leaf walk labels a leaf ENFORCED when **either** tested mutation rejects it: [line 594](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4d/tools/compare/d4d_twopass_gate_fuzz.py:594). There are 291 ENFORCED leaves with mixed PASS/FAIL outcomes. Report mutation-level outcomes; “3420 enforced leaves” does not mean every mutation was rejected. Zero recorded crashes applies to the bounded walk, not every malformed file or loader path.

Finally, retain the precise hygiene wording: **106 files byte-identical, 110 equal after named normalizations**, not “216 byte-identical.” No files were changed or merged during this review.