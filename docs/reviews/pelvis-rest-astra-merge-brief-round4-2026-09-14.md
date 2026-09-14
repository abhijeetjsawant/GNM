# Merge review round 4 for Astra GPT6 — D7c at ladder/D7c 1673e6b — 2026-09-14

Rounds 1–3 are recorded at `docs/reviews/pelvis-rest-astra-merge-review-2026-09-14.md`. The agent answered round 3 in three
commits (dcc8ace the gate, aed10e7 the inversion test, 1673e6b corrections). `git diff 9dda9ac -- src/` empty. Worktree
`/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c`. Read-only, adversarial, cite the deciding line.

## What the agent reports
- Gate: your five counter-examples now read NO MERGE — (i) the six (a)/(b) cells recomputed from the medians and the tie
  rule, SPLIT derived, the implied winner cross-checked against the shipped mode; (ii) the calibration's three conditions
  re-derived from the evaluations, including the accepted sample's own statistic within 0.05 mm and inside the bracket;
  (iii) exactly the eight named files present and equal; (iv) the follower ratio recomputed from the two numbers on six
  required bodies; (v) both performers and both landmark arrays present and equal. Ten coverage mutations added (missing
  seed ×4 clauses, missing performer ×3, missing delivered file, evaluations emptied, hygiene report absent). 31/31 detected.
- Inversion test: now the signed volume of a carried tetrahedron (the deformation gradient's determinant), vertex-order
  invariant; four tests pin it, including your diag(1, −0.2, −0.2) → uninverted and a reflection → inverted. Counts as
  per-frame ranges: 279–328 → 274–326 (11.3–13.5 %) performer 0, 46–349 → 45–344 (1.9–14.4 %) performer 1; always-inverted
  187 → 172 and 0 → 2; dominated by Left/RightUpperLeg (~38 % each) and Hips (~20 %). "Deep hip crease", "a handful" and
  "not a systematic tear" withdrawn.
- Rotational closure reconstructed from `_canonical_arm_bind_alignment` on the asset's rest matrices: median 3e-6°, max
  1.1–1.4e-5°. "Only one of four cells" corrected. Clause table updated. Tests 36 passed; the two pre-existing failures.

## Questions
1. MERGE at 1673e6b, or not? If not, ONLY what blocks.
2. Try at least two input mutations you have not tried before, and say what happens.
3. Is the carried-tetrahedron determinant a sound inversion classifier for LBS output, and do the per-frame ranges support the (withdrawn-to-neutral) B6 text as now written?
4. Anything that misreads round 3, and any number you cannot reproduce.
