# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-02 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I4, I8. In flight: I6. Blocked: nothing.

## In flight

- **I6** Surface check: our mesh outline against MAMMA's person masks — an Opus agent, since 2026-09-02

## Next up (unblocked, not started)

- none

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- Set THORAX_SMOOTHING_FRAMES from 15 to 9 in the shipped head solve (I8: interior optimum on synthetic truth, upper bound), then re-run the head gate with the MAMMA arm reporting only. One-line change plus a gate re-run; it is lane D and needs the go.

## Recent log

- 2026-09-02 [I4] MAMMA's feet scored for the first time: 9-21 deg median spread about the take mean, 31-90 deg flexion range. Our delivered foot tracks it to 6-9.5 deg spread and beats every control on 46 of 48 pairs, but carries an 18-26 deg constant offset on every foot, almost all ab/adduction: toes point 17-24 deg inward. Oracle 2.8-4.6 deg, so the frames are not the cause. Toe articulation unscoreable on either side; inversion/eversion is a DoF no direction carries.
- 2026-09-02 [I4] I4 done: MAMMA's feet measured. Ours track them but point inward by about twenty degrees on every foot -- a delivery defect a tracking gate cannot see.
- 2026-09-02 [I2] Whole pipeline on MAMMA's own skeleton projected into the four cameras: 1.15 mm median floor, 4.2 p95. Raw triangulation is exact to 1e-8 mm, so the entire floor is the temporal stage. Our real capture sits at 36-41 mm on the same reference: reconstruction is ~3% of the gap, the rest is upstream of 2D. MAMMA-grade noise: 4.6 mm; one-frame camera shift: 6.7 mm; crossed pairing: 1243 mm; frozen skeleton: 695 mm with zero reprojection error.
- 2026-09-02 [I2] I2 done: with a perfect detector our pipeline lands about one millimetre from the truth, all of it the smoothing stage. The reconstruction is not where the gap is; the detector and the joint definitions are.
- 2026-09-02 [I8] Manifest: 89 items over 55 delivery functions -- 1 leak (THORAX_SMOOTHING_FRAMES), 2 declared (camera-rig.json is ma_cap; example footage), 17 unknown origin (none MAMMA-derived), 69 clean; provenance.py exits 1 while a leak stands. Sweep on synthetic truth with own-detector noise: interior optimum at 9 frames (bracket 5-9) at every stride reaching real thorax speed; upper bound, so 15 is unsupported. MAMMA arm reports only: shape agrees, level does not.
- 2026-09-02 [I8] I8 done: the audit finds exactly one MAMMA-chosen constant, the thorax window. Re-chosen on synthetic truth it should be 9 frames, not 15. Changing it is a delivery step and waits on the go.

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
