# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-25 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3, D4b, D7, D7b, D8, D9, D8b, D8c, D9b, D7c. In flight: D4, D4c. Blocked: nothing.

## In flight

- **D4** A real pose solver (momentum on the MHR body) — an Opus agent, since 2026-09-21
- **D4c** The body-model fit's start and stop, re-registered with the trunk as a prerequisite — an Opus agent, since 2026-09-25

## Next up (unblocked, not started)

- none

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D3's recorded miss, CLOSED at D7c's close-out (2026-09-15): the exact-skeleton oracle's arm band (0.5 mm) PASSES on every seed for the first time (worst arms 0.35 mm, legs 0.00) once the pelvis is exact -- the 2.72 mm since D9b was the D7 convention seen through the leg-root-aligned gauge. The band was never moved. Still owed to the instrument-debt step: the gate's translation-aligned gauge and its frozen D2c/D3 'canonical unchanged' and 'same denominator' clauses (moved by design since D7). And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-22 [D4] Astra's one merge round on D4 (e209eea): NO MERGE -- the coordinator's 'O1 a recorded exception' was an override (never merge on an override), and the gate accepted a 15-frame truncation of B1 and left the frozen control, closure and mean-body must-fail unwired. DISPOSITION (2026-09-22): D4's ACCEPTANCE = FAIL on O1 (1.030 mm vs 1 mm; B1 +0.156/+0.115 PASS, B2 PASS, hygiene 8/8); the implementation merges OPT-IN (--body mhr; default rig unchanged; nothing shipped changes) after the agent's gate repair; O1 is re-registered prospectively as D4b (statistic, calibrated baseline from the measured 0.757 mm tracker floor, identifiable channels, mean-body rejection, fresh fixtures); the default flip is its own gated integration step (compositor, schema-aware checks, instrument compatibility, an end-to-end MHR rebuild). No further review rounds on D4.
- 2026-09-22 [D4] D4 MERGED OPT-IN 2026-09-22 (285643c, --no-ff): --body mhr available, default rig unchanged, the rig path rebuilds the D7c delivery 8/8; D4 acceptance stays FAIL on O1 and OPEN (status stays in_progress until D4b re-registers O1 prospectively). Extractor wired into the ladder (rungs 1 and 7); close-out running on the rig default.
- 2026-09-22 [D4] D4 close-out (post_merge.sh D4 on the rig default): the delivery rebuilt in place 8/8 byte-identical to D7c's; every instrument logged under artifacts/compare/post-merge-D4; the D3 gate line-identical to D7c's close-out (exact-skeleton oracle PASS 0.35 mm; the frozen D2c/D3 reference clauses still moved-by-design); head gate line-identical; three instruments first failed on cache paths and pass after restoring the body-provider run from artifacts/compare/d1-fix/body-run-regenerated and downsampled_verts from the Modal volume; the D4 gate rerun reads acceptance FAIL (O1) with every other conjunct PASS. Nothing shipped changed.
- 2026-09-24 [D4b] D4b dispatched 2026-09-24: O1 re-registered prospectively (band L on rest-segment identity, paired per-fixture floors, 12 fresh fixtures on two donors, spine-displaced must-fail, burned-first STOP); measurement only, src byte-identical; Astra's one card round at medium: dispatchable, findings adopted; worktree .claude/worktrees/ladder-D4b on ladder/D4b
- 2026-09-24 [D4b] D4b MERGED 2026-09-24 (d08929b) as a measurement, src byte-identical: STOPPED at stage 3 under the registered must-fail (ii) (reading B, Astra's merge round). The frozen drawn-set rule (a first-order pose-Jacobian column-space test, blind to the configured limits) drew 6 channels, not 8: scale_spine_length and scale_shoulder_width sit in the pose span only through pose steps of 17-65 units against limits <= 1.5, so the trunk was not scored and the band as scored accepted the displaced spine 6/6 on D4's burned cells. Post-stop exploratory (not evidence; fixtures 20261001-06 x donors 0/1 now burned): scored bones within 0.55x tolerance 12/12, the trunk beyond tolerance 9/12; WARM (calibration from the truth) holds the spine within 0.005; 973 of 1056 calibration solves stopped at max_iter 30. D4 STAYS OPEN, O1 not superseded. The card contradicted itself (PASS closes D4 vs an unscored trunk cannot) -- the lane's fifth pre-registration error, the coordinator's.
- 2026-09-25 [D4c] D4c dispatched 2026-09-25: ONE change, the calibration's landmark-derived starting identity (max_iter 30 kept); limit-aware drawn-set rule with the trunk a PRECONDITION (STOP); development on the 18 burned fixtures chooses only the trunk statistic from {median, p90, p95}, frozen before 12 untouched acceptance fixtures; init-only must-fail; B1 re-run on the D4c fitter; one verdict; fit_one merges only on PASS. Astra's one card round at medium: dispatchable, findings adopted. Worktree .claude/worktrees/ladder-D4c on ladder/D4c

## How to resume

1. Pick the step from *In flight* or *Next up*; its gate card is in `LADDER_EXECUTION_PLAN.md` §2.
2. Start it: `.venv/bin/python tools/compare/status.py set <ID> in_progress --note "..."`.
3. Instruments write a JSON report under `artifacts/` and get an extractor in `tools/compare/ladder.py`
   (Fable owns that registry). Swap-harness scripts run on the *system* `python3`; everything under
   `tools/compare/` on `.venv/bin/python`.
4. Finish it: `status.py set <ID> done --report <path> --note "..."`, then `status.py render`,
   then `.venv/bin/python tools/compare/ladder.py`, then republish the three pages to their URLs
   (ladder, board, progress — URLs in CLAUDE.md), then commit the step's files together.
5. Never select a shipped constant on a MAMMA-referenced arm. The MAMMA arm reports; it never selects.

Pages: ladder <https://claude.ai/code/artifact/56361ab8-b5a0-456d-9171-4d6a09d6c132> · board <https://claude.ai/code/artifact/cf83ef29-a4b7-4afd-9031-0918e8eb6f35> · progress <https://claude.ai/code/artifact/abd3a70c-4c51-4251-8b2f-344f095998c6>
