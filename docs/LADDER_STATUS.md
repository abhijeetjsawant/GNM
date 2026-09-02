# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-02 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I4, I6, I8. In flight: nothing. Blocked: nothing.

## In flight

- nothing in flight

## Next up (unblocked, not started)

- **I3** Detector reports that actually discriminate — an Opus agent
- **I5** Hands: the held-out-camera test as a report — an Opus agent

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- Set THORAX_SMOOTHING_FRAMES from 15 to 9 in the shipped head solve (I8: interior optimum on synthetic truth, upper bound), then re-run the head gate with the MAMMA arm reporting only. One-line change plus a gate re-run; it is lane D and needs the go.

## Recent log

- 2026-09-02 [I8] I8 done: the audit finds exactly one MAMMA-chosen constant, the thorax window. Re-chosen on synthetic truth it should be 9 frames, not 15. Changing it is a delivery step and waits on the go.
- 2026-09-02 [I6] Surface measured against SAM2 masks: ours 0.52-0.63 IoU, oracle 0.71-0.88; recall falls twice as far as precision, so the delivered body is too small and misplaced. MAMMA's mean body beats ours on every cell, so pose/retarget error exceeds the shape term. Turning our mesh 180 deg makes it worse in 8 of 8 cells: D1's premise of a shipped yaw is not what the surface finds. camera_overlay.py never sets scene fps (24 vs 30): every overlay it rendered ran 25% fast.
- 2026-09-02 [I6] I6 done: the first surface measurement. Our body is too small and misplaced rather than too big, and a plain 180-degree turn would make it worse, so the facing defect is not where the plan put it. The overlay script D1 relies on has a timebase bug.
- 2026-09-02 [D1] premise reopened by I6
- 2026-09-02 [I3] unblocked: I1, I2, I4, I6 have reports
- 2026-09-02 [I5] unblocked: I1, I2, I4, I6 have reports

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
