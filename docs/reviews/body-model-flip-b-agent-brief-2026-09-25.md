# D4i-b: the default flip, re-registered. Agent brief (Opus, one step, worktree `.claude/worktrees/ladder-D4i-b`, branch `ladder/D4i-b`)

You are executing ONE step of the AutoAnim body-capture lane. Work only in your worktree, using ABSOLUTE paths. `artifacts`,
`.cache` and `.venv` are symlinked in, and `artifacts/` is SHARED. Never write into `artifacts/commercial-multiview-soma77`
(the shipped delivery) or `artifacts/compare/i6` (the D7c silhouette baseline). Build into NEW dirs under
`artifacts/compare/d4ib-flip/`. momentum: create `.venv-mhr` with `scripts/bootstrap_mhr.sh` (from the tag), or use
`/tmp/momenv`; say which.

**Your FIRST commit merges tag `ladder/D4i-fail-b073253` into this branch.** It brings D4i's whole candidate: the flip, the
roster tooling (`d4i_mhr_roster.py`), the schema-aware silhouette and verifier, the build's refusals, the bootstrap, and the
coordinator's `post_merge.sh` and `ladder.py`.

Read, in order:
- `CLAUDE.md` (the body lane, especially the two D4i rules at the end).
- `docs/LADDER_EXECUTION_PLAN.md` §2: the D4i row (carried verbatim) and the D4i-b row (yours; the pre-registration, which
  you may not change).
- `docs/reviews/body-model-flip-2026-09-25.md` and `docs/reviews/body-model-flip-astra-merge-review-2026-09-25.md`.
- The Astra card review `docs/reviews/body-model-flip-b-astra-review-2026-09-25.md`. Every adopted finding is in the card.

## What you may change
Everything D4i's brief allowed, plus:
- `tools/compare/post_merge.sh`'s `closeout` enforcement and a new `tools/compare/rollback_flip.sh`. These two are yours this
  time: the card makes the enforcement part of the candidate. `ladder.py` stays the coordinator's.
- `tools/compare/d4_silhouette_paired.py` and `silhouette.py`: ADD a per-cell rows output, with the existing reports
  byte-unchanged.
- The B1 arm rename, with a frozen name map, and the extractor following it.
- The new gate `tools/compare/d4ib_flip_gate.py` and its fuzz.

NEVER change `src/` or `tools/fitter/mhr_delivery.py`.

## What you must NOT do
- never run `ladder.py` or `status.py`; never run `post_merge.sh closeout` against the shipped delivery. The rollback
  demonstration runs on a SCRATCH COPY of the delivery tree only.
- never publish or push; never touch the user's uncommitted `tests/`;
- never move a band, and never reuse D4i's outputs as evidence.

## Commit one stage at a time: each stage is its own commit, message in a file, plain `git commit -F`
1. **Merge the tag,** then provenance. Diff the fitter's `body_model.assets` behaviour against
   `git rev-parse --show-toplevel` in this checkout, to confirm clause 1's comparator predicts it exactly.
2. **The code:**
   - the oracle comparator (clause 1);
   - the per-cell rows output (a NEW sidecar) and the RECONSTRUCTION of D4d's reference rows from its hash-bound
     inputs. The reconstruction must reproduce D4d's existing B1 and silhouette reports exactly, else INVALID;
   - the B1 rename;
   - the enforcing close-out and `rollback_flip.sh`;
   - the three gate repairs.
   Each comes with tests.
3. **The FRESH default build** into a new, clean dir, and the oracle by clause 1 against D4d's gated delivery. Then B1
   (rows under the name map), B2, closure and B5.
4. **Hygiene:** a fresh `--body rig` build 8/8 against D7c; `src/` and the fitter byte-identical; i6 unchanged.
5. **Must-fails i–viii,** each by mutation.
6. **The rollback demonstration, in two places:** the actual `git revert` in a DISPOSABLE checkout (record the migration
   commits and how the merge parent is treated), and `closeout` on a scratch copy of the whole delivery tree (a real directory, never the
   shipped one), forcing each failure in (vii). Show the rollback restores the rig delivery byte-identical and exits
   non-zero; show a clean run leaves the MHR default in place. Then STOP and report. The coordinator runs the final roster
   and edits `ladder.py` for the rename; you are resumed for stage 7.
7. **(After you are resumed)** the gate, the fuzz, the tests, the review `docs/reviews/body-model-flip-b-2026-09-25.md`,
   and the stub updates.

At the stage-6 stop, report: the oracle reading (the leaf value against the toplevel, and the rest of the structure), the
fresh B1 rows, the hygiene result, must-fails i–viii, the rollback demonstration, and the commits. At the end, report:
1. the clause table;
2. every FAILED prediction with its attribution;
3. what is open;
4. the commits.

## Gotchas
- `.cache` in a worktree is a symlink to main's. The fitter writes `str(ROOT/.cache/mhr/assets)` from its own file's
  resolved location; check exactly what string comes out.
- The build refuses mixed directories. The rollback must MOVE the MHR delivery aside, not copy over it.
- One momentum process per performer. Blender: set the scene fps before import.
