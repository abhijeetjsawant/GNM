# Resume brief — body-capture lane, written 2026-09-22 after the D4 opt-in merge (paste this into a fresh session)

Resume the AutoAnim body-capture lane. Read, in this order: `docs/LADDER_STATUS.md` (the SessionStart hook prints it),
`docs/LADDER_EXECUTION_PLAN.md` §2 (the D7c row is the newest card; the D7 → D9b rows are the shape of a step) and §6, the
CLAUDE.md body-lane section (every standing rule, including the six added after D7c), the memory file, and
`docs/reviews/pelvis-rest-2026-09-14.md` (the last step review; §5 the clause table, §7 what is open) with the reviews
`docs/reviews/pelvis-rest-astra-review-2026-09-14.md` (the card, seven rounds) and
`docs/reviews/pelvis-rest-astra-merge-review-2026-09-14.md` (the merge, nine rounds).

**STATE.** D7c merged 2026-09-15. **D4 (the body model in the delivery path) is MERGED OPT-IN on 2026-09-22 (285643c) with its
ACCEPTANCE FAIL and open:** `scripts/build_commercial_multiview_comparison.py --body mhr` fits MHR by momentum to the same
smoothed repaired landmarks the rig consumes and delivers MHR's own mesh; silhouette 0.647 / 0.652 → 0.803 / 0.767 against
the D7c rig, CIs clear on both performers (B1, the band, PASS; B2 PASS; hygiene 8/8). The synthetic exactness oracle O1 read
1.030 mm against a 1 mm band written without the tracker's floor (0.51–0.76 mm measured) — FAIL, not moved, not excepted
(the coordinator's first "recorded exception" was an override and was withdrawn on Astra's merge round). The default stays
`rig`; nothing shipped changed; the close-out on the rig default is 8/8 byte-identical with every instrument line-identical
to D7c's. Records: `docs/reviews/body-model-precard-2026-09-21.md`, `body-model-2026-09-21.md` (on the merge),
`body-model-astra-review-2026-09-21.md`, `body-model-astra-merge-review-2026-09-22.md`. Findings: momentum's
`calibrate_markers` corrupts a later `Character.load_fbx` in the same process (one process per performer now); the cache was
wiped for storage and restored from the Modal volume (recipe in CLAUDE.md). "The Solve So Far" carries v2–v12.

**NEXT, in order:** **D4b** — O1 re-registered PROSPECTIVELY: the statistic, a calibrated baseline from the measured 0.757 mm
floor, the tolerance, the identifiable channel set (excluding `scale_foot_length` and `scale_hip_height`), the mean-body
rejection, fresh held-out fixtures — frozen before any number; raising the threshold around 1.030 is explicitly not that.
Then the **integration step that flips the default to `mhr`**: the compositor (`unified_gltf`, the N5.1 assembly) consuming
the MHR track schema, schema-aware artifact checks, every rig-schema instrument made compatible or scoped, `post_merge.sh`
reading the MHR output, an end-to-end rebuild. Usage is the constraint: one Astra card round, one merge round, per step.

**PROCESS, CHANGED ON 2026-09-15 BY THE USER'S STEER.** One Astra card review and one Astra merge review per step; findings
about a gate instrument that do not reach a card-banded verdict are instrument debt, never a merge blocker; when the
candidate stops changing between rounds, the review is bounded. Usage is the constraint now.

**(superseded by D4 above) The body model step as it was planned:** Delivered joints sit ~47 mm from MAMMA's (a
convention-laden reference) but the photographs read ours at 0.62–0.69 IoU against MAMMA's mesh at 0.84–0.89, and that gap
is the MESH: a stock asset stretched over a scaled rig with its old weights, no body model at all. So D9-legs and the
instrument-debt step are SKIPPED for now and the next step is D4 on the MHR body with D5's scaling folded in: momentum IK on
our triangulated joints on the per-performer sized MHR mean body, delivered through the real exporter, with the part-wise
silhouette against MAMMA's SAM2 masks pre-registered as THE band (both performers, identical draws, block bootstrap), the D3
closure and O1-style exactness on synthetic truth as the oracle, and MAMMA's mesh through the same rasteriser as the oracle
arm that reports and never selects. Pre-register before dispatch: if this step does not move the silhouette toward MAMMA's,
the route is wrong rather than unfinished, and lane H or a licensed body becomes the answer. pymomentum installs
(`/tmp/momenv`, MHR assets under `.cache/mhr/assets`; `tools/fitter/README.md`); D4's card in §2 needs restating after D3
and D7c (which rest the solver's output lands on: the per-performer rest on the track).

**Open after D7c, carried:** the pelvis convention (lane H); the mesh-deformation reading is a PROXY, the skinning
Jacobian (Kavan direct methods eq. 17) is D6's instrument; performer 1's unattributed torso rise; the D3 gate's gauge and
frozen references; the four SOMA constants' move to `tools/compare/`; six retrospective provenance stamps.

**PROTOCOL, OTHERWISE UNCHANGED** (worktree `.claude/worktrees/ladder-<ID>` on `ladder/<ID>` with artifacts/.cache/.venv
symlinked, one Opus agent, one commit per stage, never ladder.py / status.py / post_merge.sh from the agent, coordinator
merges `--no-ff` with a message file in the scratchpad, `post_merge.sh <ID> <delivery>` with the step's producers as extras
and `--src-stage refactored`, extractor into RUNGS and VISUALS, pages republished after reading them, commit and push as
separate plain commands, `PYTHONPATH=$PWD/src` for every instrument in a worktree).

**MINE, NOT YOURS:** Lane H and the user's uncommitted tests.
