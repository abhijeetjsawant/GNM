# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-02 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0. In flight: I1, I2, I4, I6, I8. Blocked: nothing.

## In flight

- **I1** Split the retarget: how much is body proportions, how much is the converter — an Opus agent, since 2026-09-02
- **I2** Perfect-2D oracle: run our whole pipeline on MAMMA's own skeleton — an Opus agent, since 2026-09-02
- **I4** Score MAMMA's feet, then ours against them — an Opus agent, since 2026-09-02
- **I6** Surface check: our mesh outline against MAMMA's person masks — an Opus agent, since 2026-09-02
- **I8** Provenance audit: no shipped constant chosen on MAMMA — an Opus agent, since 2026-09-02

## Next up (unblocked, not started)

- none

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.

## Recent log

- 2026-09-02 [I1] worktree ladder/I1, Opus agent
- 2026-09-02 [I2] worktree ladder/I2, Opus agent
- 2026-09-02 [I4] worktree ladder/I4, Opus agent
- 2026-09-02 [I6] worktree ladder/I6, Opus agent; id resolution is its first deliverable
- 2026-09-02 [I8] worktree ladder/I8, Opus agent. Rule for THORAX_SMOOTHING_FRAMES set by Fable: sweep the window on synthetic truth (tracked FK fixture, noise at our own detector's self-consistency amplitude), choose the p95 minimum with an interior optimum required; the MAMMA sweep is reported beside it and never selects; the agent proposes, the constant is changed only in lane D
- 2026-09-02 [—] Ladder and plan committed and pushed (e8c88e9). Go given for lane I: I1, I2, I4, I6, I8 dispatched in parallel, one Opus agent per worktree.

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
