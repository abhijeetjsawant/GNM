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
- Thorax smoothing window (I8): the proposal is 9 frames, bracket 5-9, against the shipped 15. Be clear what the data supports: p95 alone does NOT separate 9 from 15 below 0.77x real thorax speed (0.47 of paired draws, +0.07 deg); the case rests on lag and attenuation (15 lags 1.8 frames and loses 32% of peak yaw rate) and on both fixture biases running toward wider windows, which makes 9 an upper bound. Decide whether to change it in the shipped head solve (lane D, one line plus a head-gate re-run with the MAMMA arm reporting only).

## Recent log

- 2026-09-02 [I3] I3 done: the detector's error is mostly per-joint, not a per-camera offset, so the next detector step is training on our own labels. Two old headline figures were retired.
- 2026-09-02 [I5] Held-out camera protocol is a report: 36.5 mm at the subject, mean over 16 hand x fold cells (27-54 per hand). Frozen hands rejected 16/16; a prior-dominated solver drifts at constant velocity with jitter BETTER than MAMMA and is rejected only by the held-out camera; a lag of one frame is caught 6/16; zero-phase low-pass beats the shipped default 14/16 -- the shipped hand is under-smoothed. MAMMA's hand through the same protocol scores 42 mm, above ours: the floor is SOMA-77 pixel error plus a 26-73 mm MHR-to-SMPL-X convention gap, so the gate discriminates degenerate motion, not accuracy. Prior-dominated arm's held-out folds unfinished (54 min/cell).
- 2026-09-02 [I5] I5 done: the hands gate exists and rejects frozen and drifting hands, but only the held-out camera can; smoothness bands reject nothing. It also says the shipped hand is under-smoothed and that MAMMA's hand is not a floor here. Lane I complete except I7.
- 2026-09-02 [I7] worktree ladder/I7, Opus agent. Injection path exists in oracle_2d.py; single-ray slots exercise the sequence solve for the first time
- 2026-09-02 [D1] worktree ladder/D1, Opus agent: LOCATE the defect and repair camera_overlay.py's timebase; no delivery change until the defect is placed and the thorax leak is closed (provenance.py exits 1 while it stands)
- 2026-09-02 [—] Go given for I7 and for locating the facing defect (D1, location only). The D1 fix cannot ship until the thorax window decision lands, because the provenance gate fails a build while the leak stands.

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
