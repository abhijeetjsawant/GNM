# Merge review brief for Astra GPT6: D4c, the calibration's start (2026-09-25). ONE ROUND.

You are the reviewer of record for the AutoAnim body-capture lane. The rule is one merge review per step. Findings about an
instrument that do not reach a card-banded verdict count as instrument debt. Answer adversarially, cite the deciding line, and
list ONLY what blocks, then the rest as debt.

## Where to look
Worktree: /Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4c, branch ladder/D4c. Its base a58e447 holds
the card you reviewed (docs/LADDER_EXECUTION_PLAN.md §2, the D4c row; your review is
docs/reviews/body-model-start-astra-review-2026-09-25.md). The agent's commits are 86f1a5a..1a89cc7. Read:
- the review docs/reviews/body-model-start-2026-09-25.md, with records under docs/reviews/body-model-start-records/;
- the code: tools/fitter/mhr_delivery.py (the one change), d4c_identifiability.py, d4c_fixture.py,
  tools/compare/d4c_start_gate.py, d4c_start_gate_fuzz.py, d4_glb_closure.py (now hash-bound), tests/test_d4c_start.py;
- the reports under artifacts/compare/d4c-start/: gate.json, fuzz.json, logs/.

## What the coordinator verified
Rerunning d4c_start_gate.py reproduces gate.json identically: VERDICT FAIL (failed: L). Every other conjunct reads HOLDS or
PASS. tests/test_d4c_start.py passes 17. src/ and scripts/ are unchanged across the branch.

## The result, in brief
- precondition 0: eight channels drawn, spine included (bounded residual 52.5 / 55.3 mm); stage 0b holds 18/18.
- Development: p90 chosen, worst trunk ratio 1.27, trunk within tolerance 15/18.
- L fails on 1 of 12 acceptance fixtures: the trunk on 20261106 donor 0 reads 2.13 mm against 1.94 mm (1.10×); the other
  fixtures' trunk ratios are 0.02–0.81×.
- All four must-fails hold. Init-only misses L 12/12, but misses at the trunk on only 1/12 (it misses through shoulder
  width).
- B1: +0.1555 / +0.1178 against the D7c rig, CIs clear; D4c − D4 is about 0. B2 and hygiene pass.
- The prediction was PASS. It FAILED on L.

## The merge question the card did not foresee
The card: "Otherwise tooling and records merge and `fit_one` does not." But the tooling IMPORTS the new `fit_one`
(`landmark_start`, `TRUNK_STATISTIC`; the gate reads the fitter source; one test checks the source diff), so the tooling
cannot merge without the fitter change. The options:
- (a) Merge everything with the default start switched back to zero. The new keyword stays in `fit_one` and is byte-inert
  when unused, as the tripwire shows.
- (b) Keep the whole branch unmerged.
- (c) The coordinator's proposal: merge RECORDS ONLY (the review, the records folder, the card-review record) into main,
  keep ladder/D4c as a pinned branch (tag `ladder/D4c-fail-1a89cc7`) whose tooling the records cite by commit hash, and
  make nothing in tools/ or tests/ land on main.

## Questions
1. Is FAIL (L) the correct registered verdict, with no STOP? Is anything in the implementation a reason to call it
   INVALID instead?
2. Which merge option honours "Otherwise tooling and records merge and `fit_one` does not" without an override? Is (a) an
   override of the letter, given that the delivered behaviour is unchanged?
3. The attribution: is the p90 chord's short read on a large negative spine draw, plus 30-iteration under-recovery, a
   fair account of the one miss? Is the agent's "stage coupling through shoulder width" hypothesis properly held as a
   hypothesis?
4. What must D4d carry? The agent proposes either a start that does not rely on non-rigid chord lengths for the trunk and
   shoulder width, or a calibration that holds a correct start.
5. Any hole of the D4/D4b kind: population, provenance, the development-before-acceptance ordering, the hash binding.
