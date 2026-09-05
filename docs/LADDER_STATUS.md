# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-05 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3, D7, D7b, D8. In flight: D9. Blocked: nothing.

## In flight

- **D9** Aim the arms from their own origin (the green sticks to the yellow) — an Opus agent, since 2026-09-05

## Next up (unblocked, not started)

- **D4** A real pose solver (momentum on the MHR body) — an Opus agent

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D3's recorded miss: the exact-skeleton oracle's arm band (0.5 mm) fails by 0.19 mm on one of six synthetic bodies (D2's clavicle residual on a longer lever). Recommendation (2026-09-04): keep it as a standing fail; D5 re-derives the band from the fitted lever. And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-05 [D7b] merged 2026-09-05 (5b010b5, --no-ff): neck back on its landmark to the trunk-length floor (21.5/18.3 mm whole take, 42.1/11.9 bent tercile; D7 read 58.9/44.0), root and legs bit-identical, photographs not worse on either performer; B5's head clause FAILED against a 1e-9 deg band below the float32 floor (5e-6, the oracle reads the same on D7) and stated; delivery rebuilt in place, 8/8 files byte-identical to the branch build; every instrument rerun
- 2026-09-05 [D7b] 2026-09-05, D7b post-merge: the placement instrument that saw the D7 neck drift is now committed (tools/compare/delivered_vs_capture.py) and on rung 7; retarget_cost.py stays blind to this class. Rung 11's neck row vs MAMMA worsens (91->96 / 89->98) while the nose row improves (132->74 / 103->69): MAMMA's neck is not SOMA-77's Neck2, a convention disagreement, both shown. Handed to D5: the trunk shortens under flexion, 42 mm left on performer 0's bent frames, of which an equal-split flexion over Spine/Chest/UpperChest recovers 32. Coordinator errors recorded: the brief named commercial-multiview-b2/work as the rebuild source (the right one is commercial-multiview-soma77/work); Sol's 1e-9 deg head band was below the storage floor, the lane's third pre-registration error.
- 2026-09-05 [D8] card written 2026-09-05 (Sol-reviewed; effect measured and classified per camera before the band: B/D dropout leaves A-C at 172 deg, depth free along their axis); worktree ladder/D8 from 18f6cf8; runs before D6 and D5
- 2026-09-05 [D8] merged 2026-09-05 (23dbfee, --no-ff): the yellow repaired where two bodies overlap -- B/D drop the falling performer and the A-C pair at 172 deg cannot fix depth along its axis, so near-collinear two-view slots are demoted to the sequence solve; captured shoulder line off the performer's own width 22->4 / 27->16 frames, forearm 18->4, legs 0->0, raw triangulation byte-identical; MAMMA's four-camera body closer on neck and both hands for both performers (report only); delivery rebuilt in place, 8/8 byte-identical to the branch build; every instrument rerun
- 2026-09-05 [D8] 2026-09-05, D8 post-merge: the head gate's first measurement on the repaired landmarks -- the shipped head candidate now PASSES its band (median 7.19 -> 6.56 deg, p95 22.19 -> 14.06; it read FAIL on every build since D2), because the head solve reads the smoothed points and the neck/shoulders under it stopped sliding along the A-C axis. The D3 gate's same-denominator clause reads FAIL by design (the smoothed landmarks moved; the raw array is the fixed reference from now on) and needs re-baselining in the next instrument pass. Reviewer's checks before the merge: the trunk-length and shoulder-width restorations are the sequence solve's own prior (weak); MAMMA's joints on the window through subject_map were the discriminating oracle; a right-hand 'regression against raw' on performer 0 was the raw point being wrong. The synthetic fixture could not select the ray-angle ceiling (exactly rigid and smooth, it prefers recovery at every angle) -- refuted prediction on method; the 150 deg and the 6-frame gap are registered as engineering limits. Reachability fired twice on rig landmarks and the gap clause held 0.0004 of slots: two near-inert rules a later step may retire. The MAMMA worker caps custom runs at four cameras (noted during the bear experiment).
- 2026-09-05 [D9] card written 2026-09-05 (Sol-reviewed; the floor measured for the actual operation and the leg shift predicted before the band); arms only; worktree ladder/D9 from 26bef65

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
