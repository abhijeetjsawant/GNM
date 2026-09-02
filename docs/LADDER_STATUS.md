# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-02 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1. In flight: D2. Blocked: nothing.

## In flight

- **D2** Fix the converter: shoulder origin and hip height — an Opus agent, since 2026-09-02

## Next up (unblocked, not started)

- none

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D2: the clavicle fix is built and gated on ladder/D2 (pushed, unmerged) and failed two must-bands for reasons outside the clavicle. Measured on the branch, shipping nothing: D2 + root placement (Hips placed so the rig's hip joints land on the captured hips, the skeleton's own geometry, no constant) meets the round-trip band on both rigs (0.5/0.1 mm canonical, 0.07/0.04 sized, legs 0.00), lands the delivered arms 124->51 and 90->30 mm, keeps MAMMA's arm in step, and relieves the ground projection (the current build is hoisted 142/110 mm off the floor; 83/49 under it). But the clavicle jitter is unfixed in every configuration: frames over a human's peak joint rate 40/33 against 32/11 before D2, and sized + root makes performer 1 worse. Recommendation: re-scope as D2 + root placement and do not merge until the clavicle's temporal defect has its own step with its own gate (the head's physical-ceiling reject is the precedent), or accept the temporal regression explicitly. Holding for D3-D5 buys nothing on jitter.

## Recent log

- 2026-09-02 [I7] Synthetic exact truth through the real pipeline with dropped views injected. Recovery: on burst outages (real occlusion's shape) the sequence solve lands 7.7 mm vs 51.5 for a drawn line, P=1.00; on isolated single-frame drops the line wins (1.6 vs 2.1 mm), robustly -- not a defect, two good neighbours beat a solve. Smoothing: the shipped window costs 1.15 mm and is zero-phase (lag 0.00 even at 3x the window; a lag band on Savitzky-Golay discriminates nothing), attenuation 0.01 -> 0.17 at 3x; a detector p99 spike goes 21 -> 79 mm without the filter. Held-out-camera lag: the four folds disagree and D001 alone reads +0.9 frames; shifting D001 by one frame collapses all four -- a candidate one-frame sync offset for rung 0, one measurement not a claim. Thorax 9-vs-15: half the real gap is a heavy detector tail (heavy-tail i.i.d. +2.5 deg, the only arm that clears its seed spread; white noise +0.2); the other half is outside every synthetic arm and could be our frame or the reference. Eleven plan corrections in the report.
- 2026-09-02 [I7] I7 done: with a perfect detector and views dropped in bursts, the sequence solve is seven times better than drawing a line; on isolated single-frame drops the line wins. The smoother delays nothing and removes most of a bad detection. One camera, D001, reads a frame late on the held-out test -- a sync candidate for the hardware lane. And half the thorax 9-vs-15 puzzle is the detector's heavy tail.
- 2026-09-02 [H] Sync candidate for lane H: the held-out-camera lag test reads +0.9 frames on D001 and about zero on the other three; moving D001 by one frame collapses every fold to within 0.17 frames. One measurement, to be confirmed on the owned rig, never corrected on this one.
- 2026-09-02 [D2] worktree ladder/D2: the clavicle direction measured from the rig's own shoulder origin (FK of the frame's torso), legs untouched by construction; Sol reviewed the plan 2026-09-02
- 2026-09-02 [D2] D2 built and gated on ladder/D2 (not merged): the clavicle aimed from the rig's own shoulder origin; delivered arms land 181->124 and 218->90 mm on our capture and 200->131, 220->90 on MAMMA's joints, but the canonical round trip reads 67/79 mm, all of it the root placement (Hips on the hip-landmark midpoint, hip joints 80 mm below; 0.5/0.1 mm with the drop removed), and the clavicle chain jitters: frames over the human peak rate 32->48 and 11->49, steps to 160 deg. The pivot sits 60-170 mm from its landmark against 400 before, so landmark wander becomes angle. Two must-bands failed; Sol reviewed; no merge.
- 2026-09-02 [D2] D2 variant measured on the branch, nothing shipped: D2 + root placement passes the pinned round trip on both rigs (0.51/0.08 canonical, 0.07/0.04 sized) and lands the delivered arms 124->51 / 90->30 mm; the projection is relieved, not fought (hoist 142->83, 110->49 mm, penetration 0). The pre-registered prediction that the root fix would shorten the lever was wrong: the rig's shoulder origin already sat at or above the captured shoulder, so the lever lengthened in every cell (79-99 -> 101-160 mm). What discriminates is two constraints: |lever - upper-arm rest length| predicts placement, and the p5 lever predicts jitter. Jitter is unfixed under every variant (40/33 frames over the ceiling vs 32/11 pre-D2; sized + root 41/35). Picture: artifacts/compare/d2-clavicle/jitter/.

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
