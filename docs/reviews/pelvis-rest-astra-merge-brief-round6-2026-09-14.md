# Merge review round 6 for Astra GPT6 — D7c at ladder/D7c db3066c — 2026-09-14

Rounds 1–5 are recorded at `docs/reviews/pelvis-rest-astra-merge-review-2026-09-14.md`. The agent answered round 5 in two
commits (e68e06d the gate rebuilt on three rules and proved by a leaf fuzzer; db3066c B6 and corrections). `git diff 9dda9ac
-- src/` empty. Worktree `/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c`. Read-only, adversarial,
cite the deciding line.

## What the agent reports
- The gate rewritten around three rules: every value derived from named constituents or cross-checked against them (a
  disagreeing stored summary is a FAIL); every read goes through a `Reader` that raises on a missing field or member, the
  clause failing with the path named; every set checked by identity, contact runs by (side, start, end) from the frozen
  mask, which the P instrument now publishes independently. Your six round-5 escapes fall out of the rules.
- `tools/compare/d7c_gate_fuzz.py` walks 18,172 leaves of every input report, mutating each (numbers → 1e6 / −1e6 / 0 /
  deleted; strings mismatched / deleted; booleans flipped / deleted; lists and maps emptied / shortened / duplicated):
  2,246 enforced, 226 REPORT-only, 15,700 read by no clause — the latter two listed with which. Two findings about the fuzzer
  kept in the record: its first run flagged B1's `ci95` upper bound as an escape (the mutation set's gap — a CI bound fails
  on a negative — not the gate's); its second run died with a KeyError because it restored list elements by identity and
  equal floats are interned. Stated limitation: a green fuzz proves the gate depends on what it reads, not that it reads the
  right things.
- B6: the three inversion conclusions removed; the centroid explanation corrected (the weight-gradient term of Kavan eq. 17);
  "coin toss" gone from both sites. Tests 38 passed; the two pre-existing failures.

## Questions
1. MERGE at db3066c, or not? If not, ONLY what blocks.
2. Attack the gate one more time — at least two mutations, and in particular one aimed at what the fuzzer's own limitation admits: a value the gate SHOULD read and does not (name the leaf if you find one among the 15,700).
3. Is the fuzzer's classification of the 226 REPORT-only leaves consistent with the card (every REPORT clause is a report, every merge conjunct is enforced)?
4. Anything that misreads round 5, and any number you cannot reproduce.
