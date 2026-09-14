# Resume brief — body-capture lane, written 2026-09-15 after the D7c close-out (paste this into a fresh session)

Resume the AutoAnim body-capture lane. Read, in this order: `docs/LADDER_STATUS.md` (the SessionStart hook prints it),
`docs/LADDER_EXECUTION_PLAN.md` §2 (the D7c row is the newest card; the D7 → D9b rows are the shape of a step) and §6, the
CLAUDE.md body-lane section (every standing rule, including the six added after D7c), the memory file, and
`docs/reviews/pelvis-rest-2026-09-14.md` (the last step review; §5 the clause table, §7 what is open) with the reviews
`docs/reviews/pelvis-rest-astra-review-2026-09-14.md` (the card, seven rounds) and
`docs/reviews/pelvis-rest-astra-merge-review-2026-09-14.md` (the merge, nine rounds).

**STATE.** D7, D7b, D8, D9, D8b, D8c, D9b and D7c are merged, rebuilt in place, byte-checked, instrumented and pushed on
`battle0/clean-room-multiview-resolution-invariance` (last commit ec34733). D7c (2026-09-15): the pelvis is fitted to the
rig's own rest offsets about the captured hip midpoint (`E_rig_rest_kabsch`), no constant; on the D3 gate's six exact bodies
the pelvis reads 0.0001° (was a constant 6.865° of SOMA's convention), the Spine origin 0.0001 mm, the torso 0.00, and the
D3 gate's exact-skeleton oracle passes its 0.5 mm arm band on every seed for the first time (0.35 worst). On the take the
pelvis pitched ~9° and the root moved ~13 mm on every frame; the photographs not worse on 8 of 8 cells. Two selector stops
are recorded as they fell and both amendments are post hoc. "The Solve So Far" carries v2–v9 at 9.34 of ~9.5 MB.

**PROCESS, CHANGED ON 2026-09-15 BY THE USER'S STEER.** One Astra card review and one Astra merge review per step; findings
about a gate instrument that do not reach a card-banded verdict are instrument debt, never a merge blocker; when the
candidate stops changing between rounds, the review is bounded. Usage is the constraint now.

**NEXT STEP, BY THE USER'S STEER: the body model in the delivery path.** Delivered joints sit ~47 mm from MAMMA's (a
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
