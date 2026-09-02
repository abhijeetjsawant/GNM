# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-02 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0. In flight: nothing. Blocked: I8.

## In flight

- nothing in flight

## Next up (unblocked, not started)

- **I1** Split the retarget: how much is body proportions, how much is the converter — an Opus agent
- **I2** Perfect-2D oracle: run our whole pipeline on MAMMA's own skeleton — an Opus agent
- **I4** Score MAMMA's feet, then ours against them — an Opus agent

## Blocked

- **I8** Provenance audit: no shipped constant chosen on MAMMA — blocked on: THORAX_SMOOTHING_FRAMES re-selection rule not yet defined

## Decisions waiting on the user

- Give the go for lane I: five instrument agents in parallel (I1, I2, I4, I6, I8).
- Commit the ladder and plan files so worktree agents can see them (they branch from the last commit).
- Lane H: decide the rig and book the marker session; performer releases covering ML training use.

## Recent log

- 2026-09-01 [I0] Measured where we stand against MAMMA on the body: our capture reaches it, the fixed rig loses about a hundred millimetres.
- 2026-09-02 [I0] Built the substitution ladder: MAMMA divided into twelve stages, ours beside each, figures only from reports on disk, with history. Two instruments that only printed now write reports.
- 2026-09-02 [I0] Execution plan reviewed independently by Codex and Sol. The shape fit moved from first to fifth; instruments come first; three lanes.
- 2026-09-02 [I8] Found a constant in the shipped head solve that was chosen using MAMMA. Open until re-chosen from our own data.
- 2026-09-02 [I4] The delivered foot has been solved from the ball-of-foot landmark since yesterday; the old feet measurement now scores the solver against its own input and is retired.

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
