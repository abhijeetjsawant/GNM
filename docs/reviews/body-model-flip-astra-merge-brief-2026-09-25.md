# Merge review brief for Astra GPT6: D4i, the default flip (2026-09-25). ONE ROUND.

You are the reviewer of record for the AutoAnim body-capture lane. The rule is one merge review per step. Answer
adversarially, cite the deciding line, and list ONLY what blocks, then the rest as debt.

## Where to look
Worktree: /Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4i, branch ladder/D4i, base acceb55 (the card
you reviewed, with both of your blockers adopted: docs/LADDER_EXECUTION_PLAN.md §2, the D4i row). The commits are 2146a22
through b073253. Four of them are the coordinator's, per the card's ORDER:
- 1242ed0: post_merge.sh installed from the agent's proposal, and the final roster run recorded;
- 69bd1c1: ladder.py wired and the rig rungs dated;
- a044913: the gate record re-read;
- b073253: the test pin.

Read:
- the review docs/reviews/body-model-flip-2026-09-25.md, with records under docs/reviews/body-model-flip-records/
  (roster.json, gate.json, fuzz.json, stage3-predictions.json, final-roster-coordinator-run.log);
- tools/compare/d4i_flip_gate.py, d4i_flip_gate_fuzz.py, d4i_mhr_roster.py and tools/compare/post_merge.sh;
- the build script diff against acceb55.

## The result
`d4i_flip_gate.py` reads **VERDICT: FAIL (failed: oracle)**. Every other conjunct holds: B1 reproduced exactly, B2, closure,
hygiene, must-fails i–v, the population and own verdicts, a complete three-field roster, and the coordinator's migration
recorded. The oracle fails ONLY on `subject-00/01.body-track.json`, in the single leaf `/body_model/assets`. The frozen fitter
writes its own checkout's absolute path into that leaf (ladder-D4d against ladder-D4i). The other 10 per-subject files are
byte-identical. The leaf was predicted before the build (stage3-predictions.json at af69599), and D4d's tripwire named the
same leaf. It is the coordinator's pre-registration error: the card said "byte-identical, every delivered file". The same
leaf would also have tripped the close-out rollback, since a rebuild on main writes main's path.

## The coordinator's disposition, proposed
- D4i's verdict is FAIL on the oracle, read literally: no normalisation and no exception (CLAUDE.md).
- Per the card's merge rule, non-PASS means RECORDS ONLY on main, with the branch pinned by tag.
- **D4i-b** re-registers the flip, reusing this branch's tooling (it branches from the tag), and changes exactly ONE clause:
  the oracle is byte-identical on every per-subject file, except `/body_model/assets` in the track JSON, whose value must
  equal the BUILDING checkout's own `<root>/.cache/mhr/assets`. Every other leaf of that JSON must be equal.
- The close-out rule carries the same named leaf, with main's own path. Everything else is carried verbatim, including the
  full rollback.
- D4i-b also renames B1's arm to `delivered_MHR`, and makes no other change.

## Questions
1. Is FAIL (oracle) the correct registered verdict, and records-only the correct disposition? Anything INVALID or STOP?
2. Is D4i-b as proposed a legitimate prospective registration, or is naming one environment leaf after seeing the result
   the override the lane forbids? If it is legitimate, what must D4i-b's card say, or re-measure, to be more than a
   re-reading of the same deterministic bytes? For example: a fresh build on main, the leaf's value checked against the
   checkout, the other leaves compared as a set.
3. Were the coordinator's edits (post_merge.sh, ladder.py, the gate record re-read, the test pin) within the card's ORDER
   clause, and are they sound? In particular: does rewriting the committed gate record and its test after the ladder.py
   edit hide anything?
4. Any hole in the roster, the wrappers-as-entries ruling, the fuzz, or the population binding.
