# D4i: the body model becomes the default. Agent brief (Opus, one step, worktree `.claude/worktrees/ladder-D4i`, branch `ladder/D4i`)

You are executing ONE step of the AutoAnim body-capture lane. Work only in your worktree, using ABSOLUTE paths. `artifacts`,
`.cache` and `.venv` are symlinked in. Because `artifacts/` is SHARED with the main checkout, never write into
`artifacts/commercial-multiview-soma77` (the shipped delivery) or `artifacts/compare/i6` (the D7c silhouette baseline).
Build into your own dirs under `artifacts/compare/d4i-flip/` (and the rig-arm build at `artifacts/compare/soma77-rig-d4i`).

Set `PYTHONPATH=$PWD/src` for every instrument. momentum is at `/tmp/momenv/bin/python` (pymomentum-cpu 0.1.114.post0); part
of this step moves it to `.venv-mhr/`. The SOMA-77 detector is restored; builds may reuse the cached detections (say which).

Read, in order:
- `CLAUDE.md` (the body-lane section, especially the D4 entries at the end).
- `docs/LADDER_EXECUTION_PLAN.md` §2. The D4i row is yours: it is the pre-registration and you may not change it.
- `docs/reviews/body-model-2026-09-21.md` §6, and `docs/reviews/body-model-twopass-2026-09-25.md` (D4d, whose gated
  `--body mhr` delivery your oracle must reproduce).
- The Astra card review `docs/reviews/body-model-flip-astra-review-2026-09-25.md`. Every adopted finding is in the card.

## What you may change
- `scripts/build_commercial_multiview_comparison.py`: the default, the run report's not-consumed marks and `limitations`,
  refusing a directory that holds the other schema's files, and the `--mhr-python` default;
- `scripts/bootstrap_mhr.sh` (new) and `.gitignore` (`.venv-mhr/`);
- `scripts/verify_commercial_multiview_artifact.py`: the MHR branch;
- `tools/compare/silhouette.py`: the schema-aware exporter and a separate `--work` (the MHR-side exporter is
  `tools/compare/blender_export_mesh_momentum.py`);
- `tests/test_body_model.py`: the two re-pins only;
- new tests and new tools.

NEVER change `src/` or `tools/fitter/mhr_delivery.py`. NEVER edit `tools/compare/post_merge.sh` or `ladder.py`. Write `docs/reviews/body-model-flip-records/roster.json`
(per instrument: observed class, final action, post-migration class, each with evidence and the exact command) and the
coordinator edits both on your branch.

## What you must NOT do
- never run `ladder.py`, `status.py` or `post_merge.sh`; never publish or push;
- never touch the user's uncommitted `tests/`;
- never move a band. The oracle is BYTE-identical and B1 is reproduced EXACTLY; a mismatch is a finding to attribute, not
  something to normalise away.

## Commit one stage at a time: each stage is its own commit, message in a file, plain `git commit -F`
0. **Before Phase 1:** archive the D7c silhouette baseline (`artifacts/compare/i6`) to
   `artifacts/compare/d4i-flip/i6-baseline-archive/` and record its sha256.
1. **Phase 1, the OBSERVED class:** run every current `post_merge.sh` instrument, plus B2, B3, B4, closure and B5, on D4d's
   gated MHR delivery, as each instrument stands today. Record the observed class (a/b/c/d; (d) takes precedence over (b))
   with its log and exact command in `docs/reviews/body-model-flip-records/roster.json`.
2. **The code changes and their tests.** `silhouette.py`'s mesh cache is bound to the GLB hashes and exporter settings, or
   every export is fresh and said so.
3. **The oracle:** the default build (no `--body`) into `artifacts/compare/d4i-flip/default`. The full per-subject manifest,
   reference hashes, subject identities, draws and SCORED ROWS must equal D4d's; B1, B2, closure and B5 reproduced exactly.
4. **Hygiene:** `--body rig --body-run artifacts/compare/d1-fix/body-run-regenerated` into
   `artifacts/compare/soma77-rig-d4i` (a rig-arm build), 8/8 against D7c. The i6 baseline's sha256 is unchanged.
5. **Must-fails i–v, each by mutation:**
   - (i) MHR-scope refusal for every final class-(a) entry, replacements included;
   - (ii) mixed directories;
   - (iii) the mean-body negative control;
   - (iv) a false head claim;
   - (v) positive controls on B3 and B1.
   Also the frozen-population mutations: a missing subject, truncated frames, a substituted reference.
   Fill in each instrument's FINAL ACTION in `roster.json`, and write the proposed `post_merge.sh` roster as
   `docs/reviews/body-model-flip-records/post_merge.proposed.sh`: it takes a delivery-dir argument, and reads each entry's
   OWN verdict field, never the exit code.
   **Then STOP and report.** The coordinator edits `post_merge.sh` and `ladder.py` on YOUR branch and runs the final roster
   against `artifacts/compare/d4i-flip/default`. You will be resumed after that.
6. **(After you are resumed)** the POST-MIGRATION class for every final entry, from the coordinator's recorded run; then
   the gate and fuzz. The fuzz mutates the underlying evidence, never PASS flags, and keeps CRASH as its own class.
7. **Tests, the review and the stub.**
   - The review is `docs/reviews/body-model-flip-2026-09-25.md`. It carries the pre-registration verbatim, the roster table
     with all three fields, the clause table, what the MHR delivery does not carry, Astra's scope claim verbatim, and what is
     open.
   - The extractor stub is `tools/compare/extractors/d4i_flip.py`, including B4's own extractor.

At the stage-5 stop, report: the observed roster, the oracle and hygiene readings, the must-fail readings, and the commit
hashes. At the end, report:
1. the clause table (predicted / measured / verdict);
2. the full roster table;
3. every FAILED prediction with its attribution;
4. what is open;
5. the commit hashes.

## Gotchas
- The same key `root_translation_m` has a different frame in each schema; refuse by `schema_version` before reading any
  field.
- The MHR track's `rest_positions_z_up_m` is the MEAN body.
- `head_gate` and `bootstrap_margin` score a re-solved RIG head: on an MHR delivery they can pass while measuring nothing
  delivered.
- `retarget_cost` runs on the system `python3` and re-runs the RIG converter.
- One momentum process per performer. `GltfBuilder(fps=30.0)`; in Blender, set the scene fps before import.
- zsh does not word-split an unquoted `$VAR`.
