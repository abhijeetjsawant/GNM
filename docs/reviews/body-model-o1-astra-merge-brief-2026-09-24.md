# Merge review brief for Astra GPT6: D4b, O1 re-registered (2026-09-24). ONE ROUND.

You are the reviewer of record for the AutoAnim body-capture lane. The rule is one merge review per step. Findings about an
instrument that do not reach a card-banded verdict count as instrument debt. Answer adversarially, cite the deciding line, and
list ONLY what blocks the merge, then the rest as debt.

## Where to look
Worktree: /Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4b, branch ladder/D4b. Its base f91444b holds
the card you reviewed (docs/LADDER_EXECUTION_PLAN.md §2, the D4b row; your review is
docs/reviews/body-model-o1-astra-review-2026-09-24.md). The agent's commits are d6eb9b9..a39d64b. Read:
- the review docs/reviews/body-model-o1-2026-09-24.md, with records under docs/reviews/body-model-o1-records/;
- tools/fitter/d4b_o1_fixture.py and tools/fitter/d4b_identifiability.py;
- tools/compare/d4b_o1_gate.py and tools/compare/d4b_o1_gate_fuzz.py, plus tests/test_d4b_o1.py;
- the reports under artifacts/compare/d4b-o1/: drawn-set.json, burned.json, gate.json, fuzz.json, fresh/cell-*.json and
  closure.json.

## What the coordinator verified
src/ and tools/fitter/mhr_delivery.py are unchanged across the branch (sha256 3136befb…). Rerunning the gate on
fresh/ + closure.json reproduces gate.json byte-identically. Pointing it at the wrong folder reads FAIL on the missing
population, as it should.

## The agent's result, in brief
- The drawn-set rule, frozen before any fit, drew SIX channels, not the predicted eight: scale_spine_length and
  scale_shoulder_width fell below it. Their raw displacement is 108–110 mm and 20 mm, but their pose-orthogonal residual is
  ~0 at the frozen 1e-6 rank convention. The pose Jacobian reproduces them through weak neck_twist/head_twist directions
  (2e-5 to 1e-4 of the largest singular value), needing pose steps of 17–65 units against configured limits of at most 1.5.
  At a 1e-3 cut, which was reported and never used to select, all eight would be drawn.
- The conjunction (validity AND L on the scored set AND closure AND must-fails i–iii) reads PASS: L's worst reading is 0.55×
  tolerance, and must-fail (ii) at the trunk fails 12/12.
- The trunk, reported and not scored, is beyond its paired tolerance on 9/12 fixtures (1.15–14.94 mm against 1.84–2.38 mm).
  The gate prints `D4: STAYS OPEN` under the card's clause "a trunk that cannot be scored and rejected ... cannot close D4".
- Burned: (ii) fails at the trunk 6/6, so there was no STOP; L passes 3/6.
- WARM (the calibration started at the truth) holds the spine within 0.005. At max_iter 30, 973 of 1056 calibration solves
  stopped at the cap. CONVERGED (300) still drifts on one burned seed (0.445 short).

## The coordinator's proposed disposition
Merge D4b as a measurement, with tooling and reports only. Its record reads: the conjunction PASS on the scored set; D4
STAYS OPEN; O1 is NOT superseded. The card contradicts itself ("PASS → D4 closes" against "cannot close D4"). That is a
pre-registration error: the drawn-set rule was a first-order column-space test blind to the configured limits, and it
removed the very channel the step exists to score. The specific clause governs. What follows: D4c, carded from WARM and the
iteration-cap reading as a convergence/start change to the fitter, selected on no oracle, with the trunk back in the band
under a limit-aware identifiability rule registered prospectively.

## Questions
1. Mergeable as proposed? If not, list ONLY what blocks.
2. Is "the specific clause governs → D4 stays open" the correct reading of a self-contradicting card, or is any closure
   (or any non-closure) of D4 on this run an override?
3. The agent read must-fail (ii) as "the trunk's error against the trunk's tolerance, scored or not" (reading A), not "the
   band as scored must reject the displaced spine" (reading B, under which the step would have STOPPED at stage 3). Which
   does the card's text support? If B, are stages 4–6 an overrun that must be struck from the record?
4. Is the identifiability implementation faithful to the frozen rule (finite differences, rank convention, the lstsq change
   made before the freeze)? Does anything in the fixture, gate or fuzz reproduce D4's holes (truncation, unwired conjuncts,
   path provenance)?
5. What must D4c's card carry so that it does not select a constant (max_iter, the starting identity) on the oracle that
   scores it?
