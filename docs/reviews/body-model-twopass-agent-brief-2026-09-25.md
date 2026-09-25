# D4d: a second calibration pass on the landmark start. Agent brief (Opus, one step, worktree `.claude/worktrees/ladder-D4d`, branch `ladder/D4d`)

You are executing ONE step of the AutoAnim body-capture lane. Work only in your worktree, using ABSOLUTE paths. `artifacts`,
`.cache` and `.venv` are symlinked in. Set `PYTHONPATH=$PWD/src` for EVERY instrument under `tools/compare/`, which run on
`.venv/bin/python`. momentum runs on `/tmp/momenv/bin/python` (pymomentum-cpu 0.1.114.post0); if it is gone, rebuild it from
`tools/fitter/README.md`. The SOMA-77 detector is restored at `.cache/autoanim_gnm/gem-x/inputs/onnx/`, so a fresh detection
can run. Real-take builds may still reuse the cached detections; say which you did.

**Your FIRST commit merges tag `ladder/D4c-fail-1a89cc7` into this branch.** D4c's landmark start and tooling are not on
main, and your candidate is D4c's start plus the second pass.

Read, in order:
- `CLAUDE.md` (the body-lane section, including the D4b and D4c rules at the end).
- `docs/LADDER_EXECUTION_PLAN.md` §2. The D4d row is yours: it is the pre-registration and you may not change it.
- `docs/reviews/body-model-start-2026-09-25.md` (D4c) and `docs/reviews/body-model-start-astra-merge-review-2026-09-25.md`.
- `docs/reviews/body-model-o1-2026-09-24.md` (D4b's WARM and CONVERGED probes).
- The Astra card review `docs/reviews/body-model-twopass-astra-review-2026-09-25.md`. Every adopted finding is in the card.

## The one code change
`tools/fitter/mhr_delivery.py` `fit_one` gains a pass count, defaulting to 1 so the result is byte-identical to D4c. With
passes = 2, the full calibration (stage A, then stage B) runs a second time, started from its own copy of the first pass's
identity, each pass at `max_iter` 30. `tracking.max_iter` and everything else are untouched. `--body mhr` selects the pass
count. Never edit `src/`.

## Reuse; build only what the card names
Reuse by import: D4c's fixture, gate and closure (from the merged tag), D4b's fixture, and D4's B1/B2 instruments.

New files:
- `tools/fitter/d4d_fixture.py`, holding the Phase-1 arms {D4c start, WARM, TWO-PASS, SW*} and the Phase-2 arms;
- `tools/compare/d4d_twopass_gate.py`, ONE verdict;
- `tools/compare/d4d_twopass_gate_fuzz.py`, with CRASH as its own class, never counted as ENFORCED.

The gate re-derives the tripwire equality and B2's checks from the files; it never reads the producers' booleans. Run long
jobs in the FOREGROUND, logging under `artifacts/compare/d4d-twopass/logs/NN-name.log`.

## What you must NOT do
- never select anything; the Phase-1 rule is frozen, and its decision JSON is committed BEFORE any Phase-2 fixture exists;
- never sweep the pass count, and never raise `max_iter`;
- never run `ladder.py`, `status.py` or `post_merge.sh`; never publish or push;
- never edit an existing test or plan document; new tests go in `tests/test_d4d_twopass.py`; never touch the user's
  uncommitted `tests/`;
- never move a band. On a STOP (precondition 0, stage 0b, the WARM fork, TWO-PASS not closing the gap, init-only passing),
  commit and report there.

## Commit one stage at a time: each stage is its own commit, message in a file, plain `git commit -F`
1. **Merge the D4c tag,** then provenance and precondition 0 (D4c's drawn set, reused by sha256).
2. **The code change and the tripwire.** passes = 1 through the new code must reproduce D4c's pinned delivery and acceptance
   cells byte-identical, and `--body rig` must rebuild D7c 8/8.
3. **Phase 1** on the 30 burned fixtures: stage 0b, the four arms with stage A and B readings, and the spine-tercile
   report. Commit the frozen rule's decision JSON. STOP here unless the rule selects TWO-PASS.
4. **Phase 2 acceptance:** 12 × the seven arms, with closure on every candidate cell.
5. **The real take:** rebuild `--body mhr` with the combined fitter; run B1 (both comparators) and B2.
6. **Gate and fuzz.**
7. **Tests, the review and the stub.**
   - Tests go in `tests/test_d4d_twopass.py`.
   - The review is `docs/reviews/body-model-twopass-2026-09-25.md`.
   - The extractor stub is `tools/compare/extractors/d4d_twopass.py`, with its VISUALS entry.
   - Report frames go under `artifacts/compare/d4d-twopass/report/`, plus an mp4.

End with:
1. the clause table (predicted / measured / verdict);
2. the Phase-1 record, including WARM on 20261106/d0;
3. every FAILED prediction with its attribution;
4. what is open;
5. the commit hashes.

## Gotchas
- One process per cell; `calibrate_markers` corrupts a later `load_fbx` in the same process.
- Read rest lengths by FK with the identity retained and the pose zeroed, never from `rest_positions_z_up_m`.
- The trunk chord is not rigid; the start's p90 statistic is D4c's, frozen.
- MHR is centimetres and Y-up. `GltfBuilder(fps=30.0)`; in Blender, set the scene fps before import.
- zsh does not word-split an unquoted `$VAR`.
