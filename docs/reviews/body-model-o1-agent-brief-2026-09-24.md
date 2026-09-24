# D4b: O1 re-registered prospectively. Agent brief (Opus, one step, worktree `.claude/worktrees/ladder-D4b`, branch `ladder/D4b`)

You are executing ONE step of the AutoAnim body-capture lane, and it is a MEASUREMENT step: the source does not change.
Work only in your worktree, using ABSOLUTE paths. The shell cwd persists across Bash calls. `artifacts`, `.cache` and `.venv`
are symlinked in. Set `PYTHONPATH=$PWD/src` for EVERY instrument under `tools/compare/`, which run on `.venv/bin/python`.
momentum runs on `/tmp/momenv/bin/python`, which is GONE and must be rebuilt first (`tools/fitter/README.md`:
`uv venv /tmp/momenv --python 3.12`, then `uv pip install --python /tmp/momenv/bin/python pymomentum-cpu`); record the
installed version.

Read, in order:
- `CLAUDE.md` (the body-lane section).
- `docs/LADDER_EXECUTION_PLAN.md` §2. The D4b row is yours: it is the pre-registration and you may not change it.
- `docs/reviews/body-model-2026-09-21.md` §2.1 and §6.
- `tools/fitter/d4_o1_exactness.py`, D4's fixture, which you extend in a NEW file and never edit.
- `tools/fitter/mhr_delivery.py`, whose sha256 `3136befb…` must be byte-identical at every commit.
- The Astra card review `docs/reviews/body-model-o1-astra-review-2026-09-24.md`. Every adopted finding is in the card.

## Usage is the constraint
One card review has happened, and one merge review will follow. Build only the instruments the card names:
- `tools/fitter/d4b_o1_fixture.py`, the fixture and its arms, one process per cell;
- `tools/fitter/d4b_identifiability.py`, the drawn-set rule;
- `tools/compare/d4b_o1_gate.py`, the wired conjunction;
- `tools/compare/d4b_o1_gate_fuzz.py`, the mutation fuzz, in the `d7c_gate_fuzz.py` pattern.

Reuse `mhr_delivery.fit_one` / `export_glb` / `write_track`, D4's `configured_limits` and `truth_motion` logic (import it,
never copy-and-drift) and `tools/compare/d4_glb_closure.py`. Run long jobs in the FOREGROUND, logging under
`artifacts/compare/d4b-o1/logs/NN-name.log`. Foreground `sleep` is blocked, so chunk the work instead.

## What you must NOT do
- never edit anything under `src/` or `tools/fitter/mhr_delivery.py`;
- never change a momentum setting in a banded arm; `max_iter 300` appears ONLY in the report-only CONVERGED probe;
- never run `ladder.py`, `status.py` or `post_merge.sh`; never publish or push;
- never edit an existing test or plan document. New tests go in `tests/test_d4b_o1.py`. Never touch the user's uncommitted
  `tests/`;
- never move a band or a tolerance rule. If must-fail (ii) passes on the burned cells or on a fresh fixture, STOP there,
  commit, and report.

## Commit one stage at a time: each stage is its own commit, message in a file, plain `git commit -F`
1. **Environment and provenance.** Rebuild momenv. Record the sha256 of `mhr_delivery.py`, both donor files and
   `compact_v6_1.model`, and the pymomentum version.
2. **The drawn set.** Apply the raw and pose-orthogonal sensitivity rule on both donors, and report the rank and condition
   number. Freeze the set in a JSON file BEFORE any fit.
3. **Burned.** Run the L instrument and must-fail (ii) on D4's six retained cells (`artifacts/compare/d4-body/o1/`), and
   record them as BURNED. STOP if (ii) passes.
4. **Fresh fixtures.** 12 fixtures × {oracle, exact_identity, mean_body, spine_displaced, warm, converged}, with the closure
   on every oracle cell.
5. **Gate and fuzz.** `gate.json` must carry predicted, measured and verdict for every clause, each derived from the report
   numbers and never a literal PASS. The verdict is INVALID, PASS or FAIL. The fuzz turns each conjunct by mutating its
   INPUTS, and a short, missing or non-finite population reads FAIL.
6. **Tests, the review and the stub.**
   - Tests go in `tests/test_d4b_o1.py`.
   - The review is `docs/reviews/body-model-o1-2026-09-24.md`: the pre-registration verbatim, the burned reading, the
     fresh reading, the clause table, the mechanism probes, the blindnesses, and what is open, including D4c's evidence if
     it FAILS.
   - The extractor stub is `tools/compare/extractors/d4b_o1.py`. Never edit the registry.

End with the clause table, every FAILED prediction with its attribution, the WARM/CONVERGED reading, what is open, and the
commit hashes.

## The gotchas that bite THIS step
- `calibrate_markers` corrupts a later `Character.load_fbx` in the same process. Use one process per cell and load every
  character before any calibration.
- MHR is CENTIMETRES and Y-up. Capture Z-up metres → MHR is `(x, z, −y) × 100`; report L in mm.
- The rest lengths are FK at MHR's zero pose WITH the identity retained. NEVER the track's `rest_positions_z_up_m`:
  `write_track` computes it with ALL parameters zero, so it reads the mean body on every arm (Astra, verified).
- Seed each identity draw by the pair (seed, donor index); the same seed on two donors must not share a draw.
- The pose-orthogonal projection works per frame: finite-difference the 17 mapped joints against every non-flexible pose
  parameter at that frame, and take the least-squares residual of the identity displacement.
- `scale_hip_height`'s limit is the point [0, 0]. `scale_feet` does not exist, so `scale_foot_length` stands in for it.
  Both are REPORTED under the drawn-set rule, never silently dropped.
- D4's `o1.json` paths point into `.claude/worktrees/ladder-D4/`. Resolve retained cells by sha256, not by path.
- zsh does not word-split an unquoted `$VAR`, and a waiter loop `until ! pgrep -f NAME` matches its own command line.
