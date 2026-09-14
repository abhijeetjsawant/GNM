# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-14 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3, D7, D7b, D8, D9, D8b, D8c, D9b. In flight: D7c. Blocked: nothing.

## In flight

- **D7c** The pelvis fitted to the rig's own rest offsets, not SOMA-derived constants — an Opus agent, since 2026-09-14

## Next up (unblocked, not started)

- **D4** A real pose solver (momentum on the MHR body) — an Opus agent

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D3's recorded miss, restated at D9b (2026-09-07): the exact-skeleton oracle's arm band (0.5 mm) reads 2.72 mm since D9b, up from 0.80-1.17, because retarget_cost.score aligns each frame on the leg-root midpoint and so charges a correctly re-aimed arm with the hoist's perpendicular part (the gauge, not the candidate; the absolute-frame companion row in d9b_hoist_gate.py improves on every seed). Recommendation: keep it a standing fail, do not move the band; the instrument-debt step re-pins the gate's gauge and its frozen D2c/D3 references together. And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-14 [D7c] The calibrated fixture (target 8.7636 mm reproduced, baseline 1.81 mm, bisection reached sigma 0.335547 -> 8.7495 in 10 evaluations, draws bit-identical) STOPPED on its own monotonicity precondition: one 0.0135 mm dip above the accepted sigma from a keep-mask change on two bodies. Sent to Astra as round 7 with a proposed wording (a dip smaller than the tolerance, and a single crossing of the target, is not a stop).
- 2026-09-14 [D7c] Astra round 7: the calibration's monotonicity precondition amended in the reviewer's wording (any-pair decrease > 0.05 mm or a second sign change stops; a diagnosed keep-mask dip within tolerance does not), recorded as post hoc; the agent's stop stays recorded; the agent resumed to reread all of S at the exact evaluated sigma 0.335546875.
- 2026-09-14 [D7c] The agent reached MERGE on its own reading at ladder/D7c 9dda9ac (E_rig_rest_kabsch ships; hygiene, tripwire 8/8, O1 exact, O2, S reread PROCEED at the calibrated sigma, P on the take, B1 8/8 with three torso cells RISING unpredicted, B2 same denominator). Astra's merge review: NO MERGE -- oracle P2 never measured and the gate's S conjunct excludes G2 (an injected FAIL still returned MERGE); B6's mesh-deformation, rotational-closure and rest/IBM reports owed; a B1 attribution ablation (D9b's root under the candidate's locals) required; six overclaims corrected. Sent back to the agent.
- 2026-09-14 [D7c] Astra merge review round 2 at 0b3eba4: NO MERGE. Oracle P2 resolved; two gate verdicts were literals (an in-memory wrong-origin residual of zero and an (a)/(b) SPLIT both still read MERGE); B6's mesh reading must be finished on a reader Astra verified to 0.005 mm against Blender (the exporter writes the first animated pose as the node default, so the 590/156 mm mismatch was not a defect); playback translation, frame-correct rotational closure, the B1 root share's CI through zero, the stale Head claim. Sent back to the agent.
- 2026-09-14 [D7c] Astra merge review round 3 at 65a5a4d: NO MERGE. The gate's verdicts derive from saved classifications and partial populations (five further mutations still read MERGE); the mesh inversion classifier is unsound (first-vertex dominant joint; a positive-determinant LBS called inverted); B6 maxima over 15 frames; the closure constant fitted from the output. Sent back to the agent.
- 2026-09-14 [D7c] Astra merge review round 4 at 1673e6b: NO MERGE. Five more gate coverage holes (empty G1/G2 maps, a hidden failing oracle seed and take run, an unnamed B1 cell); the tetrahedron inversion test unsound under varying skin weights -- after three unsound classifiers the reading is to be qualified as a proxy and the sound skinning-Jacobian measurement handed to D6. Sent back to the agent.

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
