# Merge review brief for Astra GPT6: D4d, a second calibration pass (2026-09-25). ONE ROUND.

You are the reviewer of record for the AutoAnim body-capture lane. The rule is one merge review per step. Findings about an
instrument that do not reach a card-banded verdict count as instrument debt. Answer adversarially, cite the deciding line, and
list ONLY what blocks the merge, then the rest as debt.

## Where to look
Worktree: /Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4d, branch ladder/D4d. Its base fedd834 holds
the card you reviewed (docs/LADDER_EXECUTION_PLAN.md §2, the D4d row; your review is
docs/reviews/body-model-twopass-astra-review-2026-09-25.md). The agent's commits are f04bd9b (the merge of tag
ladder/D4c-fail-1a89cc7) through d9cc161. Read:
- the review docs/reviews/body-model-twopass-2026-09-25.md, with records under docs/reviews/body-model-twopass-records/
  (including phase1-decision.json);
- tools/fitter/mhr_delivery.py (the second pass; the CLI now defaults to --passes 2);
- tools/fitter/d4d_fixture.py, tools/compare/d4d_twopass_gate.py, d4d_twopass_gate_fuzz.py, tests/test_d4d_twopass.py;
- the reports under artifacts/compare/d4d-twopass/.

## What the coordinator verified
Rerunning d4d_twopass_gate.py reproduces gate.json identically: VERDICT PASS (every conjunct holds). tests/test_d4d_twopass.py
passes 16. src/ and scripts/ are unchanged across the branch. Two D4c tests fail on this branch by design:
test_the_source_diff_accepts_only_the_two_starts and test_the_gate_reproduces_its_committed_verdict_from_the_artifacts both
pin D4c's fitter. The agent was not allowed to edit them.

## The result, in brief
- Phase 1 (the fork, on D4c's 12): WARM holds 12/12 (worst 0.214×; the failing body 20261106/d0 reads 0.214×). TWO-PASS
  closes 12/12 (worst 0.726×; the failing body goes 1.103 → 0.676×), so the frozen rule selected TWO-PASS.
- Phase 2 (12 untouched bodies): L 12/12 (worst trunk 0.726×); all must-fails hold; B1 +0.152 / +0.116 over the D7c rig.
  The overall verdict is PASS.
- **Caveat the agent put first in its review:** the uniform draw gave donor 0 no spine below −0.40, so the acceptance set
  does not contain D4c's failure class. The REPORTED one-pass arm (D4c's own fitter) also passes 12/12 (worst 0.883×). The
  acceptance population therefore cannot discriminate the candidate from D4c; the evidence that the second pass fixes the
  failing class is Phase 1 only (burned fixtures).
- **Reported regressions of the second pass:**
  - trunk ratios worse on 2 of 12 acceptance bodies (0.039 → 0.188×, 0.644 → 0.706×);
  - on the real take, the recovered spine moves 0.959 → 1.102 for performer 0 (past the configured limit of 1.1) and
    0.420 → 0.823 for performer 1;
  - the landmark residual rises from 16.7 to 17.7 / 17.1 mm;
  - B1 combined − D4c reads −0.0039 [−0.0058, −0.0004] on performer 0, a CI entirely below zero; the card REPORTS this and
    does not band it.

## The card's merge rule
PASS → everything merges (D4c's pinned change and tooling with this step's), `--body mhr` carries the combined fitter, and
the default stays `rig`.

## Questions
1. Mergeable under the card? Is PASS the correct registered verdict, with no INVALID or STOP?
2. The acceptance set cannot discriminate the candidate from D4c's fitter. Does PASS still "supersede D4's O1" honestly?
   What exactly may the record claim, and what may it not?
3. The real take's spine moving past its configured soft limit (1.102 > 1.1) and a CI-negative B1 against D4c on performer 0:
   are these debt, or does any card clause make them blocking? Is shipping the combined fitter behind the opt-in
   `--body mhr` acceptable given them?
4. The two D4c tests that pin D4c's fitter: re-pin them to the tag's fitter source (read via git), or retire them as
   superseded? Which keeps the record honest?
5. The CLI default `--passes 2` while `fit_one` defaults to 1: a hazard for any caller? Should the pass count be in the fit
   report or the track, rather than in a sidecar `calibration-passes.json`?
6. Any hole of the D4, D4b or D4c kind: population, provenance, the Phase-1 ordering, or crashes in the fuzz.
