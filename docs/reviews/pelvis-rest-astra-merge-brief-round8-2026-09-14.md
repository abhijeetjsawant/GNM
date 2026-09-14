# Merge review round 8 for Astra GPT6 — D7c at ladder/D7c 17dc09e — 2026-09-14

Rounds 1–7 are recorded at `docs/reviews/pelvis-rest-astra-merge-review-2026-09-14.md`. The agent answered round 7 in five
commits (e06e708 the four leaves; e557429, 880fc1b, 17dc09e the inverted coverage audit and the generated inventory;
9823596 the correction). `git diff 9dda9ac -- src/` empty. Worktree
`/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c`. Read-only, adversarial, cite the line.

## What the agent reports
- Your four round-7 mutations now read NO MERGE (B2 derived from 2 performers × 2 builds; P1 preservation from
  `frames_that_differ == 0` and the per-side counts; every arm's population checked at the single read point `body_metric`;
  the calibration's medians derived from `per_body_mm` for every evaluation — a moved stored median with the amended file's
  copy moved too is caught on the band at 9.028 mm).
- `coverage_audit`: every leaf no clause reads is classified LABEL / PROVENANCE / DIAGNOSTIC / MEASUREMENT with its report;
  every MEASUREMENT under a report a clause reads is a GAP unless a named family justifies it; a justification matching
  nothing fails. COVERED: 0 gaps, 0 dead patterns, 4,165 leaves read by a clause, 7,904 unread measurements in 86 named
  families. A saved `verdict` / `status` string counts as a MEASUREMENT — thirteen surfaced and were derived.
- The saved-value inventory is generated from the `Reader`'s record: 1,416 reads cross-checked, 106 trusted families, 0
  unjustified; the old hand-written claim withdrawn as false. Closing the audit meant READING twelve more families, among
  them: every build report's `resolved_module` under this worktree; `src_default == E_rig_rest_kabsch` leaf for leaf on
  every seed; the reread pinned to the exact accepted σ; five instruments' landmark byte-identity required to agree; the
  candidate's eight files required to DIFFER from D9b's; the three immutable files itemised into 22 families with the
  amended admissibility rule recomputed.
- Two self-found defects recorded: a circular cross-check (root_translation_m's flag against itself) that hid the one
  genuinely trusted boolean; a stop check a deletion satisfied (Missing also produced FAIL, so deleting selector.json's
  follower table read MERGE with the stop gone) — a stop must now fail ON ITS BAND; 368 leaves moved to enforced.
- Fuzz: 18,172 visited (13,763 + 4,409); 4,918 enforced (from 2,336), 0 gaps, 148 report, 40 diagnostics, 0 preserved-STOP,
  13,066 unread (710 label, 127 provenance, 1,459 diagnostic, 2,898 containers, 7,872 justified measurements); 0 monotone
  failures. 49 clauses, thirteen conjuncts (the oracle's own premises added), 35 PASS / 12 REPORT / 2 FAIL (the two stops,
  on their bands). The :93 attribution withdrawn.
- Coordinator note from the agent: `built_here` ties every build report to the worktree root, so the post-merge rebuild in
  the main checkout must regenerate the artifacts or hygiene FAILs.

## Questions
1. MERGE at 17dc09e, or not? If not, ONLY what blocks.
2. One more attack of the round-7 kind if you can find one; and one on the coverage audit's justifications (a family that
   justifies a measurement the card actually bands).
3. `built_here` / `resolved_module` tied to the worktree root: right as a check, or will it make the post-merge rerun in the
   main checkout fail for a reason that is not a defect — and if so what should the check be tied to?
4. Anything that misreads round 7, and any number you cannot reproduce.
