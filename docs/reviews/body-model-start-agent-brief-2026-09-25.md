# D4c: the calibration's start. Agent brief (Opus, one step, worktree `.claude/worktrees/ladder-D4c`, branch `ladder/D4c`)

You are executing ONE step of the AutoAnim body-capture lane. Work only in your worktree, using ABSOLUTE paths. `artifacts`,
`.cache` and `.venv` are symlinked in. Set `PYTHONPATH=$PWD/src` for EVERY instrument under `tools/compare/`, which run on
`.venv/bin/python`. momentum runs on `/tmp/momenv/bin/python` (pymomentum-cpu 0.1.114.post0, rebuilt 2026-09-24). If it is
gone, rebuild it from `tools/fitter/README.md` and record the version.

Read, in order:
- `CLAUDE.md` (the body-lane section, including the two D4b rules at the end).
- `docs/LADDER_EXECUTION_PLAN.md` §2. The D4c row is yours: it is the pre-registration and you may not change it.
- `docs/reviews/body-model-o1-2026-09-24.md` (D4b, and its WARM/CONVERGED probes).
- `docs/reviews/body-model-o1-astra-merge-review-2026-09-24.md`.
- `docs/reviews/body-model-2026-09-21.md` (D4: B1, B2, hygiene, the build commands in its appendix).
- The Astra card review `docs/reviews/body-model-start-astra-review-2026-09-25.md`. Every adopted finding is in the card.

## The one code change
`tools/fitter/mhr_delivery.py` may change ONLY as follows:
- `fit_one` gains an optional starting identity that both `calibrate_markers` calls receive in place of `zero.copy()`;
- a function computes the frozen landmark start from the consumed array;
- the build script's `--body mhr` path selects the start. Keep a flag that forces the zero start, for the tripwire.

Nothing else in the fitter changes: `max_iter`, `calib_frames`, `loss_alpha`, the offsets and the tracking all stay as they
are. Never edit `src/`.

## Reuse; build only what the card names
Reuse D4b's fixture, gate and fuzz by import (`tools/fitter/d4b_o1_fixture.py`, `tools/compare/d4b_o1_gate.py`), and D4's
B1/B2/closure instruments (`tools/compare/d4_silhouette_paired.py`, `d4_b2_same_denominator.py`, `d4_glb_closure.py`). The
closure report must now carry the sha256 of the GLB and track it measured, and the gate must verify them.

New files:
- `tools/fitter/d4c_identifiability.py`, the limit-aware rule;
- `tools/fitter/d4c_fixture.py`, the arms, one process per cell;
- `tools/compare/d4c_start_gate.py`, ONE verdict;
- `tools/compare/d4c_start_gate_fuzz.py`.

Run long jobs in the FOREGROUND, logging under `artifacts/compare/d4c-start/logs/NN-name.log`.

## What you must NOT do
- never select anything on the acceptance fixtures;
- the ONLY development choice is the trunk statistic from {median, p90, p95}, by the card's rule, frozen in a JSON and
  committed BEFORE any acceptance fixture is generated;
- never run `ladder.py`, `status.py` or `post_merge.sh`; never publish or push;
- never edit an existing test or plan document; new tests go in `tests/test_d4c_start.py`; never touch the user's
  uncommitted `tests/`;
- never move a band. If a STOP fires (precondition 0, stage 0b, init-only passing), commit and report there.

## Commit one stage at a time: each stage is its own commit, message in a file, plain `git commit -F`
1. **Provenance, and precondition 0.** The limit-aware drawn set, frozen before any fit, with the unbounded projection
   reported beside it. STOP if the spine is not drawn.
2. **The code change and the tripwire.** A zero start through the new code must reproduce D4's `--body mhr` delivery and
   D4's retained O1 cells byte-identical, and `--body rig` must rebuild D7c 8/8.
3. **Development.** The 18 burned fixtures: stage 0b (the spine control must fail as scored on all 18, else STOP), then the
   zero start against the landmark start for each trunk statistic. Freeze the chosen statistic in the development JSON.
4. **Acceptance fixtures.** 12 × {candidate, exact_identity, mean_body, spine_displaced, init_only, legacy}, with closure on
   every candidate cell.
5. **The real take.** Rebuild `--body mhr` with the new start; run B1 (both comparators) and B2.
6. **Gate and fuzz.** One verdict, derived from the inputs.
7. **Tests, the review and the stub.**
   - Tests go in `tests/test_d4c_start.py`.
   - The review is `docs/reviews/body-model-start-2026-09-25.md`: the pre-registration verbatim, the clause table, the
     development record, the falsifier's reading, the blindnesses, and what is open.
   - The extractor stub is `tools/compare/extractors/d4c_start.py`, with its VISUALS entry. Never edit the registry.
   - Report frames go under `artifacts/compare/d4c-start/report/` (JPEG 480 px q40, every 6th frame, camera A, the D4 fit
     beside the D4c fit over the mask outline), plus a full-rate mp4.

End with:
1. the clause table (predicted / measured / verdict);
2. the development record;
3. every FAILED prediction with its attribution;
4. what is open;
5. the commit hashes.

## Gotchas
- `calibrate_markers` corrupts a later `Character.load_fbx` in the same process. Use one process per cell and per performer,
  and load every character before any calibration.
- MHR is CENTIMETRES and Y-up. Capture Z-up metres → MHR is `(x, z, −y) × 100`.
- Read rest lengths by FK with the identity retained and the pose zeroed. NEVER use `rest_positions_z_up_m`, which is the
  mean body.
- The trunk chord root→c_neck is NOT rigid: flexion shortens it. That is why the trunk statistic is a development choice.
- The acceptance fixtures (seeds 20261101–06 × donors 0/1) must not exist on disk before the development JSON is committed.
  The gate checks that the development JSON's commit is older than the acceptance cells' commit.
- `GltfBuilder(fps=30.0)`. Blender export: set `scene.render.fps` before import and check the action range 0..149.
- zsh does not word-split an unquoted `$VAR`.
