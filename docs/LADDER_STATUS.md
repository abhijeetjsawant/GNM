# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-02 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1. In flight: nothing. Blocked: nothing.

## In flight

- nothing in flight

## Next up (unblocked, not started)

- **D2** Fix the converter: shoulder origin and hip height — an Opus agent

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.

## Recent log

- 2026-09-02 [D2] unblocked: I1 and D1 done. The Hips 0.98 literal is already gone; what remains is the clavicle direction about the rig shoulder origin (0.72 * torso_up at ~:1565/1568). Gate: I1 canonical round trip on the arms drops from 36-47 mm toward the legs' 0.00; legs must hold 0.00; arm B drops in step.
- 2026-09-02 [D1] D1 shipped: the delivered character faces the right way. Rebuilt from the cached detections with the regenerated asset; the rig and every triangulated point are byte-identical to the previous build, only the rotations turned. The old build is kept beside it for the record. D2, the clavicle origin, is next and needs the go.
- 2026-09-02 [D1] Rung 11's canonical figure for performer 1 moved 137.8 -> 137.4 mm across the rebuild. Not the mirror: the fix branch proved rungs 7 and 11 identical to four decimals with both arms at window 15. What changed is the window-9 head solve, and the neck takes a share of the chest-to-head rotation, so the Head joint (the scoreboard's 'nose') moves with it.
- 2026-09-02 [I7] Synthetic exact truth through the real pipeline with dropped views injected. Recovery: on burst outages (real occlusion's shape) the sequence solve lands 7.7 mm vs 51.5 for a drawn line, P=1.00; on isolated single-frame drops the line wins (1.6 vs 2.1 mm), robustly -- not a defect, two good neighbours beat a solve. Smoothing: the shipped window costs 1.15 mm and is zero-phase (lag 0.00 even at 3x the window; a lag band on Savitzky-Golay discriminates nothing), attenuation 0.01 -> 0.17 at 3x; a detector p99 spike goes 21 -> 79 mm without the filter. Held-out-camera lag: the four folds disagree and D001 alone reads +0.9 frames; shifting D001 by one frame collapses all four -- a candidate one-frame sync offset for rung 0, one measurement not a claim. Thorax 9-vs-15: half the real gap is a heavy detector tail (heavy-tail i.i.d. +2.5 deg, the only arm that clears its seed spread; white noise +0.2); the other half is outside every synthetic arm and could be our frame or the reference. Eleven plan corrections in the report.
- 2026-09-02 [I7] I7 done: with a perfect detector and views dropped in bursts, the sequence solve is seven times better than drawing a line; on isolated single-frame drops the line wins. The smoother delays nothing and removes most of a bad detection. One camera, D001, reads a frame late on the held-out test -- a sync candidate for the hardware lane. And half the thorax 9-vs-15 puzzle is the detector's heavy tail.
- 2026-09-02 [H] Sync candidate for lane H: the held-out-camera lag test reads +0.9 frames on D001 and about zero on the other three; moving D001 by one frame collapses every fold to within 0.17 frames. One measurement, to be confirmed on the owned rig, never corrected on this one.

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
