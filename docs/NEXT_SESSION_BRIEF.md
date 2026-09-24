# Resume brief — body-capture lane, written 2026-09-25 after D4c (paste this into a fresh session)

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

**D4b (2026-09-24, merged d08929b as a measurement, src byte-identical): STOPPED at stage 3; D4 stays open, O1 not
superseded.** The card moved O1's band to rest-segment identity with paired per-fixture floors on twelve fresh bodies (two
donors). Its frozen drawn-set rule (first-order, limit-blind) dropped `scale_spine_length` and `scale_shoulder_width`, so the
trunk went unscored and the band as scored could not reject the displaced-spine control: STOP under the registered must-fail
(ii) (reading B, Astra's merge round, one round at medium). Post-stop exploratory: scored bones <= 0.55x tolerance 12/12, the
trunk beyond tolerance 9/12, WARM holds the spine within 0.005, and 973 of 1056 calibration solves stop at `max_iter` 30.
Records: `docs/reviews/body-model-o1-{card,astra-review,2026,astra-merge-review}-2026-09-24.md` (and the records folder).
Fixtures 20261001–06 × donors 0/1 are burned.

**D4c (2026-09-25): FAIL on L, records only on main (5d27b4f); the tooling and the landmark-start fitter change are
pinned at tag `ladder/D4c-fail-1a89cc7` and NOT merged; D4 stays open.** The trunk was scored this time (the limit-aware rule
drew 8). Development picked p90. L passed on 11 of 12 fresh bodies; the trunk read 1.10× on one body with a strongly
shortened spine. The photographs held (+0.156 / +0.118 over the rig). Records: `docs/reviews/body-model-start-*-2026-09-25.md`.

**NEXT, in order:** **D4d**, carded prospectively (Astra's D4c merge review, the D4d item):
- first, on BURNED fixtures only, separate wrong-start recovery from drift off a correct start: e.g. set only the
  shoulder-width start to the truth, with stages A and B read separately;
- then freeze ONE intervention before new acceptance identities;
- never raise the shared `max_iter` (it also changes tracking);
- keep the scored trunk, the paired floors, the controls, the hash-bound closure and the rebuilt-delivery B1/B2;
- register the non-PASS merge as records-only, or write tooling that stands without the candidate;
- a `Spine1` feed changes the input and needs explicit scope.

Then the **integration step that flips the default to `mhr`**:
- the compositor (`unified_gltf`, the N5.1 assembly) consuming the MHR track schema;
- schema-aware artifact checks;
- every rig-schema instrument made compatible or scoped;
- `post_merge.sh` reading the MHR output;
- an end-to-end rebuild.

Usage is the constraint: one Astra card round and one merge round per step (the user asked for MEDIUM effort on D4b).

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
