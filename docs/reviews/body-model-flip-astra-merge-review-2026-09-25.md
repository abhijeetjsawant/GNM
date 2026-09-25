# Astra GPT6 merge review of D4i (2026-09-25), one round at MEDIUM effort. Verdict: FAIL (oracle); RECORDS ONLY; D4i-b legitimate on stated terms

Verified against the source before adoption:
- `post_merge.sh` only prints `DIFFERENT` (lines 121 and 140) and never enforces it;
- `d4i_flip_gate.py:299` accepts the roster at `len(rows) >= 18`;
- one fuzz case sets a supplied `held` flag (`d4i_flip_gate_fuzz.py:160`).

| # | finding | change |
|---|---|---|
| 1 | FAIL (oracle) is the registered verdict; predicting the leaf does not amend the card; records only | ADOPTED: records only on main; the complete candidate pinned at tag `ladder/D4i-fail-b073253` |
| 1b | D4i-b is a legitimate NEW prospective registration if it states that its criterion was informed by D4i, and freezes: the two files and the exact JSON pointer (present, a string, equal to the independently determined building checkout's asset path); equality of the WHOLE remaining JSON structure (keys, types, arrays, empty containers); every other file byte-identical; a FRESH no-`--body` build into a clean dir, fresh roster evidence, and the in-place main rebuild under the same comparator; the B1 arm-name correspondence with no numerical change | carried verbatim into D4i-b's card; it shows reproducibility across checkouts, not accuracy |
| 2 | `post_merge.sh` does not enforce the close-out (it prints DIFFERENT, missing files hash to empty strings, verdicts are not compared, no rollback, stale reports can survive) | D4i-b must make the close-out ENFORCE full hashes and file presence, fresh reports, verdict equality and the full rollback before it can merge |
| 3 | the gate trusts `mismatches == []` for the population, accepts the roster at `>= 18`, and compares rounded summaries as "scored rows" | D4i-b repairs all three: expected against observed compared directly, the registered instrument identities, per-cell rows preserved and compared |
| debt | ORDER respected; no concealment in a044913 or b073253; the wrappers are valid; the fuzz mutates one supplied flag; the ladder text "MHR from D4i" is false on main; the review narrative is stale in places | recorded; the branch's ladder text stays on the branch (records only) |

---

**D4i is FAIL (oracle). Records-only on main is correct; pin the complete candidate by tag. No additional INVALID or STOP is established.** I re-evaluated the gate read-only: its entire result exactly matches committed `gate.json`. The deciding registration is “BYTE-IDENTICAL” and “Non-PASS → records only.” [Card](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4i/docs/LADDER_EXECUTION_PLAN.md:180)

**Blockers to the flip or unchanged reuse in D4i-b — none blocks recording this failure:**

1. **The path difference cannot be excused inside D4i.** Both track JSONs fail byte identity, even though `/body_model/assets` is their only differing leaf. Predicting that failure before building does not amend the card. Preserve the literal FAIL.

   **D4i-b is legitimate prospective registration**, provided it explicitly acknowledges that its revised criterion was informed by D4i. The lane expressly permits a “NEW prospective registration” after acceptance FAIL. It does not permit retroactively passing D4i. [Deciding rule](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4i/CLAUDE.md:336)

   Before new execution, D4i-b must freeze:
   - The exact two files and exact JSON pointer; require its presence, string type, and value equal to the independently determined building checkout’s asset path.
   - Equality of the entire remaining JSON structure, including added/deleted keys, types, arrays and empty containers—not merely intersecting leaves. Every other registered file remains byte-identical.
   - A fresh no-`--body` candidate build into a clean output directory, fresh roster evidence, and the subsequent in-place main rebuild under the same comparator. Cached detections may be reused if explicitly declared; old delivered outputs cannot stand in for the new build.
   - The exact B1 arm-name correspondence, with no numerical changes admitted through renaming.

   This demonstrates reproducibility across checkout environments; it supplies no new evidence of fitting accuracy or generalisation.

2. **`post_merge.sh` does not enforce the registered close-out.** Line 121 merely prints `DIFFERENT`; missing files can yield equal empty hash strings. The function ends after running the roster, without comparing its verdicts to the merge-review record or implementing rollback. Reusing an output directory can also retain stale reports after an early process failure. [Implementation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4i/tools/compare/post_merge.sh:100)

   Before D4i-b can merge, make the close-out enforce full hashes/file presence, fresh reports, verdict equality and the registered complete rollback, either directly or through an explicit coordinator procedure. Logging failures is insufficient for “anything else triggers a FULL ROLLBACK.” [Deciding clause](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4i/docs/LADDER_EXECUTION_PLAN.md:217)

3. **The gate cannot yet certify all the claims made about its evidence checks.**
   - Population validation trusts nonempty `expected` and `mismatches == []`. My in-memory mutation changed B3’s **observed population** while leaving that leg true. Compare the required expected/observed fields directly and verify report bindings. [Deciding implementation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4i/tools/compare/d4i_flip_gate.py:275)
   - Roster completeness accepts `len(rows) >= 18`. Deleting `head_gate` from the current 19-entry roster still passes. Require the registered instrument identities and replacement relationships. [Deciding line](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4i/tools/compare/d4i_flip_gate.py:299)
   - “Scored rows equal” compares report dictionaries, while B1 writes rounded summaries. That does not implement the card’s explicit “SCORED ROWS equal, not rounded summaries.” Preserve or recompute and compare the per-cell rows. [Gate](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4i/tools/compare/d4i_flip_gate.py:142), [B1 summaries](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4i/tools/compare/d4_silhouette_paired.py:139)

   These are enforcement repairs, not new acceptance bands. The present population records independently have matching expected/observed fields; the present roster contains all 19 entries. Thus these holes do not overturn the recorded FAIL, but **39/39 targeted fuzz mutations is not proof of complete coverage**.

**The rest is debt:**

- **ORDER was respected.** Both coordinator-owned files changed on the candidate before this review, and the final roster run is recorded. The clause does not require `ladder.py` to precede stage 6. `a044913` changes only the migration conjunct and consequent failed-list; `b073253` updates that expectation while retaining per-conjunct comparisons and the literal-oracle tests. I find no concealment in those edits. [ORDER](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4i/docs/LADDER_EXECUTION_PLAN.md:174)
- Update the stale review narrative and label the fuzz record’s earlier baseline. Also correct ladder text saying the delivered body “is MHR from D4i”; that is false on main after this disposition. Keep those migration commits out of the records-only merge.
- **Wrappers-as-entries is valid under the registered wording.** Scope checking precedes instrument invocation; direct invocation of the underlying instruments remains debt. B3/B4 wrapper PASS certifies execution/population, not scientific quality. [Scope boundary](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4i/tools/compare/d4i_mhr_roster.py:435)
- The fuzz description overstates its independence: one case explicitly mutates a supplied `held` flag. Replace that with underlying refusal evidence when repairing the gate. [Mutation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4i/tools/compare/d4i_flip_gate_fuzz.py:160)
- The absolute asset path, rig half-resolution comparison, capture-side hardcoded paths and pre-existing test failures remain debt. No additional blocker emerged from the build-script diff.