# Merge review round 7 for Astra GPT6 — D7c at ladder/D7c cffaad3 — 2026-09-14

Rounds 1–6 are recorded at `docs/reviews/pelvis-rest-astra-merge-review-2026-09-14.md`. The agent answered round 6 in three
commits (9c412d7 the gate, 65aaeae corrections, cffaad3 the fuzzer; branch head cffaad3). `git diff 9dda9ac -- src/` empty.
Worktree `/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c`. Read-only, adversarial, cite the line.

## What the agent reports
- Your four round-6 escapes: authentication derived from both hashes present and equal (the boolean cross-checked); the
  follower's both terms from the body rows with the three stored numbers cross-checked; every body/arm must carry the frozen
  populations 150 / 149 and 50 / 47 (47 bent-tercile pairs because a pair needs BOTH endpoints in the 50-frame tercile); each
  control must fail exactly its named channels, all inside the protected set. Four more saved booleans derived or
  cross-checked (`holds`, `failing_channels`, `pelvis_mode_held`, `winner/arm`); THREE saved booleans survive, listed in
  gate.json with why (each an `np.array_equal` whose arrays exist in no report; the landmark byte-identity pair owed as
  instrument debt).
- Fuzzer: 18,172 = 13,763 leaves + 4,409 containers; 2,336 enforced, 0 gaps ("moves a conjunct clause without turning the
  verdict" is now a GAP), 164 report, 40 diagnostics, 14 preserved-STOP, 15,618 unread; the 8 control leaves enforced; a
  monotone check pushing the failing direction six orders further: 0 failures (its first version demanded both extremes and
  was withdrawn). Stated: the fuzzer cannot find a leaf the gate never reads.
- Corrections: the centroid reason fixed in the artifact string too; the interning explanation withdrawn (the true trigger
  never isolated; the wholesale-restore fix removes the class). Tests 38 passed.

## Questions
1. MERGE at cffaad3, or not? If not, ONLY what blocks.
2. The three surviving saved booleans: acceptable as listed, or must the landmark byte-identity be re-derived before the merge?
3. One more attack of the round-6 kind (a leaf the gate should read and does not), if you can find one.
4. Anything that misreads round 6, and any number you cannot reproduce.
