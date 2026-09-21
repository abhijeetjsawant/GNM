# D4 — the body model in the delivery path. Agent brief (Opus, one step, worktree `.claude/worktrees/ladder-D4`, branch `ladder/D4`)

You are executing ONE step of the AutoAnim body-capture lane. Work only in your worktree with ABSOLUTE paths (the shell cwd
persists across Bash calls; `artifacts`, `.cache`, `.venv` are symlinked in). `PYTHONPATH=$PWD/src` for EVERY instrument you
run in the worktree, or `.venv`'s editable install silently measures the main checkout's source (every D7c instrument exits
on this and logs the resolved module path; copy that pattern). Everything under `tools/compare/` runs on `.venv/bin/python`;
momentum runs on `/tmp/momenv/bin/python` (pymomentum-cpu; MHR assets under `.cache/mhr/assets`, lod0..lod6, 127 joints,
204 parameters, CENTIMETRES, Y-up). Read, in order: `CLAUDE.md` (body-lane section), `docs/LADDER_EXECUTION_PLAN.md` §2
(the D4 row is yours — it is the pre-registration and you may not change it; the D7c and D9b rows are the shape of a step),
`docs/reviews/body-model-precard-2026-09-21.md` (the measurement this card is built on) and the pre-card scripts
`tools/fitter/fit_delivery_mhr.py`, `tools/compare/d4_silhouette_paired.py`, `tools/compare/blender_export_mesh_momentum.py`
(reproduce the pre-card FIRST: hygiene and the reproduction clause), `docs/FITTER_PLAN.md` §7 (why offsets are pinned) and
the Astra review `docs/reviews/body-model-astra-review-2026-09-21.md` (every finding is in the card).

## Usage is the constraint (the user's steer, 2026-09-15 and 2026-09-21)
One card review has happened; one merge review will. Do not build instruments the card does not name. Reuse
`tools/compare/silhouette.py` (the band), `tools/compare/delivered_vs_capture.py` (B3, on the MHR joint set),
`tools/swap-harness/retarget_cost.py`'s PAIRS and `tools/head/subject_map.py` (B4), `d3_skeleton_gate.py`'s GLB reader
(B5). Long runs in the FOREGROUND with logs under `artifacts/compare/d4-body/logs/NN-name.log`; background loops here have
restarted and cost hours. Foreground `sleep` is blocked; chunk instead.

## What you must NOT do
- never run `tools/compare/ladder.py`, `status.py` or `post_merge.sh`; never publish, push, or edit an existing test or
  plan document; new tests go in NEW files (`tests/test_body_model.py`); never touch the user's uncommitted `tests/`;
- never let anything MAMMA-derived enter the delivery (MAMMA's mesh and joints are REPORT arms only); never fit a constant on
  a MAMMA arm; never lift SOMA's SAM-licensed 28-dim scale PCA — the 68 raw channels only, from MHR's own release;
- never move a band; if a pre-registered clause fails, measure, attribute, STOP where the card says stop, commit, report.

## Commit one stage at a time, each its own commit, message in a file, plain `git commit -F`
1. hygiene — `--body rig` (the default until this merges) rebuilds the shipped delivery 8 of 8 byte-identical;
2. the reproduction — `--body mhr` at lod6 on the RAW array reproduces the pre-card's 0.789 / 0.730 to 0.001 through the
   build script (this proves the integration path is the pre-card's path; then switch to the smoothed array and lod2);
3. O1 — the exactness oracle and its must-fails (the mean body; free offsets), on six synthetic MHR bodies, BEFORE the
   delivery is scored;
4. the delivery — `--body mhr`, lod2, the smoothed landmarks, `subject-XX.glb` + the `autoanim.body-track/2.0-mhr` track,
   into `artifacts/compare/d4-body/delivery` with `work/` COPIED (never symlinked) from the shipped delivery;
5. the bands — B1 (silhouette, both performers, identical draws, the mean-body and MAMMA arms reported; precision and recall
   per part beside IoU), B2 (same denominator), B3, B4, B5; `gate.json` with every clause's predicted / measured / verdict
   derived from the report numbers (never a literal PASS) and a small input-mutation table showing each conjunct turns
   the verdict;
6. new tests, the review `docs/reviews/body-model-2026-09-21.md` (the D7c review's shape: pre-registration verbatim, the
   defect, what ships, instrument, oracle, real take, must-fails, the clause table, findings, blindnesses, open), the
   extractor STUB `tools/compare/extractors/d4_body_model.py` (never a registry edit), report frames (JPEG, 640 px, q42,
   every 6th frame, camera A and D, rig beside MHR) and an mp4 under `artifacts/compare/d4-body/report/`.

End with: the clause table, every FAILED prediction with its attribution, what is open, the commit hashes.

## The gotchas that bite THIS step
- momentum's `GltfBuilder()` inherits the FBX's 120 fps unless constructed `GltfBuilder(fps=30.0)`: the take runs 4x fast
  and Blender's importer, with `scene.render.fps = 30` set BEFORE import, then reads 0..37 frames. Check the action range
  (0..149) in the Blender export log every time.
- momentum exports one marker sphere per locator; `blender_export_mesh_momentum.py` keeps meshes named `mesh*` only.
- Capture Z-up metres → MHR Y-up centimetres is `(x, z, −y) × 100`; the inverse is `tools/fitter/rootcheck.py`'s `tocap`.
  Verify by closure (export, read the GLB's joints, compare to the fed landmarks), never by reading.
- Offsets PINNED (limit_weight 10). Free offsets reach an 84 mm median and leave the skeleton at the mean (FITTER_PLAN §7):
  that is a must-fail, and its silhouette will look fine — the silhouette cannot see it, the identity channels can.
- The silhouette scores the MESH's outline and placement, not joints or depth; clothing is in the masks and in no mesh.
  Report precision and recall per part beside IoU so an inflated body cannot pass on recall.
- Whole-take medians on 150 correlated frames: block bootstrap (block 15), candidate and controls on identical draws.
- `silhouette.py` needs `DELIVERY/subject-XX.body-track.npz` for the subject map (copy the shipped ones if your track
  format differs, and say so) and the masks cache in `--work` (copy `artifacts/compare/i6/masks-960x540-*.npz`).
- The MAMMA oracle arm needs `.cache/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz` (restored
  2026-09-21 from the Modal volume `autoanim-mamma-data-v1`); the fixture videos and calibration yaml are back too.
- zsh does not word-split an unquoted `$VAR`; a waiter loop `until ! pgrep -f NAME` matches its own command line.
