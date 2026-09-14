# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-14 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3, D7, D7b, D8, D9, D8b, D8c, D9b. In flight: D7c. Blocked: nothing.

## In flight

- **D7c** The pelvis fitted to the rig's own rest offsets, not SOMA-derived constants — an Opus agent, since 2026-09-14

## Next up (unblocked, not started)

- **D4** A real pose solver (momentum on the MHR body) — an Opus agent

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D3's recorded miss, restated at D9b (2026-09-07): the exact-skeleton oracle's arm band (0.5 mm) reads 2.72 mm since D9b, up from 0.80-1.17, because retarget_cost.score aligns each frame on the leg-root midpoint and so charges a correctly re-aimed arm with the hoist's perpendicular part (the gauge, not the candidate; the absolute-frame companion row in d9b_hoist_gate.py improves on every seed). Recommendation: keep it a standing fail, do not move the band; the instrument-debt step re-pins the gate's gauge and its frozen D2c/D3 references together. And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-07 [D9b] 2026-09-07, D9b merged (9cab14b, --no-ff) on all eight pre-registered conjuncts: the root-dependent chain (trunk, neck, clavicles with pass B rerun, arms, hands) re-solved from the hoisted root after the foot-contact projection; every such bone on its ray from the DELIVERED origin to 0.0003 mm (was 3.7-6.3 mm median on the 67 / 22 hoisted frames, 18 mm at worst); root, contacts, hips, legs, feet, toes byte-identical, everything byte-identical on the 83 / 128 unhoisted frames; the tripwire (hoist forced to zero, old src vs new) 8 of 8 byte-identical; photographs level within 0.001 IoU on 8 of 8 cells; pass B's accepted set unchanged. Prediction FAILED and attributed: the D3 gate's aligned oracle median did not hold (arms 1.17 -> 2.70, close-out reads 2.72) because retarget_cost.score subtracts the leg-root offset per frame -- the gauge, band untouched, an absolute-frame companion row improves on every seed. Grok 4.6 reviewed the merge in Sol's place (MERGE stands; the skip-second-pass-B hole is untestable on this take). Close-out: rebuilt in place 8/8 byte-identical, every instrument logged under artifacts/compare/post-merge-D9b; D3 gate lines as predicted (legs 0.07, arms 2.72, the frozen D2c/D3 references still moved-by-design); head gate line-identical to D8c's close-out. Two structural test pins re-pinned in place (the sequence rule runs twice per clavicle; the _joint_origin pin follows the aims into the helpers). Handed forward: the legs' 4.6-9.2 mm ray miss on hoisted frames (D9-legs); the plant's own cost on exact truth, p95 10-14 mm on a third of the frames; the D3 gate's translation-aligned gauge (instrument debt); D8c's head-gate log predates its own rebuild. Next: D7c.
- 2026-09-14 [—] 2026-09-14: D7c registered as the next step (planned; card not yet written). Queue after it: D9-legs, then the instrument debts (the D3 gate's translation-aligned oracle gauge and its frozen D2c/D3 references, --median-from, the D8 headline recheck, a two-view depth-stretch fixture, the contact model's own cost on exact truth, D8c's stale head-gate log), then D6, D5, D4. Reviewer of record from today: Astra GPT6, in the seat Sol held (Grok 4.6 via cursor-agent as the fallback).
- 2026-09-14 [D7c] card written 2026-09-14 after a pre-card measurement (tools/compare/precard/d7c_*.py): the D3 oracle's pelvis reads a constant 6.865 deg pitch from the SOMA template, Spine 21-28 mm, torso 9-12 mm, all 0.000 on the rig's own rest; reviewed by Astra GPT6 in four rounds (docs/reviews/pelvis-rest-astra-review-2026-09-14.md) -- do-not-dispatch three times, every finding adopted after verification against the source; worktree .claude/worktrees/ladder-D7c on ladder/D7c, one Opus agent: hygiene, instrument first, S the selector before the src change, two rig modes with S choosing, the guard, the projection-preservation contract
- 2026-09-14 [D7c] D7c agent STOPPED at S before any src change (ladder/D7c d146aa3..8a82ee4): hygiene 8/8, instrument reproduces the pre-card; (a) E_rig_rest_kabsch wins all six selector cells and beats C-on-SOMA; the frozen-pitch follower clears 2 deg on every body but the >= 2x clause fails on 5 of 6 (ratios 1.45-2.74) -- attributed to the fixture's noise (lever sd 20 mm vs the take's 6.6/11.1; D7's own report recorded the same 1.7-2.9x). Fixture-repair question (calibrate sigma to the take's own lever spread, the larger performer, src untouched, band untouched) sent to Astra as round 5.
- 2026-09-14 [D7c] Astra rounds 5-6 (fixture question): a repair is permissible under the D8b rule with a MATCHED calibration frozen first -- target = the take's spine-lever sd at S's own stage on the guard-kept frames, ddof=0, the larger performer 8.764 mm; one sigma by bisection on [0.10, 1.00], original draws preserved, unreachable/non-monotone => STOP; baseline through observe_body reported never subtracted; G1 amended to an array-level equivalence; G2 corrected. Card amended in the plan (5d48476); the agent resumed on it to reread all of S.
- 2026-09-14 [D7c] The calibrated fixture (target 8.7636 mm reproduced, baseline 1.81 mm, bisection reached sigma 0.335547 -> 8.7495 in 10 evaluations, draws bit-identical) STOPPED on its own monotonicity precondition: one 0.0135 mm dip above the accepted sigma from a keep-mask change on two bodies. Sent to Astra as round 7 with a proposed wording (a dip smaller than the tolerance, and a single crossing of the target, is not a stop).

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
