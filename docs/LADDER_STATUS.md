# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-02 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I8. In flight: I7, D1. Blocked: nothing.

## In flight

- **I7** Temporal: lag and outlier repair measured on synthetic truth — an Opus agent, since 2026-09-02
- **D1** Fix the character facing the wrong way — an Opus agent, since 2026-09-02

## Next up (unblocked, not started)

- none

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.

## Recent log

- 2026-09-02 [D1] D1 located: the delivered character IS turned round -- pelvis, chest, head and the mesh's own nose all point opposite the performer (forward-dot -0.92 to -1.00), the feet point the right way because they are solved separately. It is a left/right naming mirror in the rig, in three places, not a yaw. The overlay script that the original diagnosis used had a timebase bug, now fixed, and the original diagnosis stands. The fix waits on the thorax decision.
- 2026-09-02 [D1] Sol reviewed the facing fix proposal: build it on a branch, try renaming the Left/Right bones before mirroring any geometry, and rerun I1, rungs 7 and 11, the head gate and I6 as regression gates. It merges only after the thorax decision and a second review.
- 2026-09-02 [I8] THORAX_SMOOTHING_FRAMES changed 15 -> 9 on the user's decision; provenance audit now CLEAN (exit 0). Head gate re-run with the MAMMA arm reporting only: P1 median improved on both performers (6.39 -> 5.52, 8.25 -> 7.19 deg, gated), p95 worse on performer 0 (16.4 -> 20.0) and better on performer 1 (27.5 -> 22.2); the oracle floor rose on both (3.62 -> 4.27, 3.87 -> 4.31) because a narrower window leaves more frame jitter in the reference frame itself. Performer 0's gate verdict flipped PASS -> FAIL on the p95 band. Reverting on that would be selecting on the MAMMA arm, so the value stands; the outcome is recorded, not smoothed.
- 2026-09-02 [I8] Sol's read of the head gate at window 9: the candidate's p95 rise on performer 0 (+3.6 deg) sits inside the oracle floor's rise (+4.0), so the PASS -> FAIL flip is the frame moving under a fixed band, not the head degrading; candidate-minus-oracle in the same frame improved on both performers. Open question logged on I8: the synthetic sweep put 9 and 15 level, the real floor did not, so either the white-noise model is too gentle or MAMMA's frame is over-smoothed; sent to I7 to size. The delivered body-track files still carry the window-15 head until the one rebuild after the facing fix lands.
- 2026-09-02 [I8] Paired block bootstrap re-run at window 9 (tools/head/bootstrap_margin.py, 2000 draws, identical draws): head-minus-oracle p95 gap 1.8 deg on performer 0 and 5.3 on performer 1 at block 20; P(candidate passes the 20 deg band) 0.39 / 0.22; P(oracle passes) 0.84 / 0.88. At window 15 the oracle passed at 1.00. So the fixed 20 deg p95 band is now one the reference's own answer fails on about one draw in seven: the gate is near miscalibrated for this frame, which is a calibration fact about the gate, recorded and NOT acted on -- moving the band on this evidence would select on the MAMMA arm. The paired gap is now a standing ladder figure.
- 2026-09-02 [D1] D1 fix built on its branch (relabel, not geometry: every torso and head joint turns exactly 180 deg about its own up axis, nothing else moves, I1 and rungs 7/9/11 identical to four decimals; the mirror lived at five sites, the head axes were a half-turn not a mirror). Sol reviewed: not merging yet -- three items first: render the hands before and after (a curl sign can flip with the frame and no gate sees a pose), regenerate the asset through Blender/MPFB so its request hash matches the corrected joint map, and one handedness check on a real SOMA motion export. Two gate bands failed for instrument reasons and are recorded as gate findings: the head cannot reach a forward-dot of 0.9 that MAMMA's own head does not reach (0.84), and I6's silhouette cannot see facing with the limbs held.

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
