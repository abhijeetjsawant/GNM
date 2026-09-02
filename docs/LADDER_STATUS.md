# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-02 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I4, I6, I8. In flight: I3, I5. Blocked: nothing.

## In flight

- **I3** Detector reports that actually discriminate — an Opus agent, since 2026-09-02
- **I5** Hands: the held-out-camera test as a report — an Opus agent, since 2026-09-02

## Next up (unblocked, not started)

- none

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- Thorax smoothing window (I8): the proposal is 9 frames, bracket 5-9, against the shipped 15. Be clear what the data supports: p95 alone does NOT separate 9 from 15 below 0.77x real thorax speed (0.47 of paired draws, +0.07 deg); the case rests on lag and attenuation (15 lags 1.8 frames and loses 32% of peak yaw rate) and on both fixture biases running toward wider windows, which makes 9 an upper bound. Decide whether to change it in the shipped head solve (lane D, one line plus a head-gate re-run with the MAMMA arm reporting only).

## Recent log

- 2026-09-02 [I6] I6 done: the first surface measurement. Our body is too small and misplaced rather than too big, and a plain 180-degree turn would make it worse, so the facing defect is not where the plan put it. The overlay script D1 relies on has a timebase bug.
- 2026-09-02 [D1] premise reopened by I6
- 2026-09-02 [I3] unblocked: I1, I2, I4, I6 have reports
- 2026-09-02 [I5] unblocked: I1, I2, I4, I6 have reports
- 2026-09-02 [I3] worktree ladder/I3, Opus agent
- 2026-09-02 [I5] worktree ladder/I5, Opus agent

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
