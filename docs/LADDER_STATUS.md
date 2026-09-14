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

- 2026-09-14 [D7c] Astra merge review round 3 at 65a5a4d: NO MERGE. The gate's verdicts derive from saved classifications and partial populations (five further mutations still read MERGE); the mesh inversion classifier is unsound (first-vertex dominant joint; a positive-determinant LBS called inverted); B6 maxima over 15 frames; the closure constant fitted from the output. Sent back to the agent.
- 2026-09-14 [D7c] Astra merge review round 4 at 1673e6b: NO MERGE. Five more gate coverage holes (empty G1/G2 maps, a hidden failing oracle seed and take run, an unnamed B1 cell); the tetrahedron inversion test unsound under varying skin weights -- after three unsound classifiers the reading is to be qualified as a proxy and the sound skinning-Jacobian measurement handed to D6. Sent back to the agent.
- 2026-09-14 [D7c] Astra merge review round 5 at 843bce5: NO MERGE -- six more gate escapes (a stored S aggregate never derived, global summaries and run identities unchecked, missing measurement fields silent) and surviving B6 inversion conclusions. The agent is rebuilding the gate on one rule (derive or cross-check every value, missing is FAIL, sets by identity) and proving it with a leaf-level fuzzer over every input report.
- 2026-09-14 [D7c] Astra merge review round 6 at e68e06d: NO MERGE -- four leaves the gate should read and does not (the authentication hashes, the follower's denominator, the selector populations, the controls' named channels); the fuzzer's REPORT-only class misclassified 8 enforced control leaves. Sent back to the agent.
- 2026-09-14 [D7c] Astra merge review round 7 at cffaad3: NO MERGE -- four more stored-summary reads whose constituents are on disk (B2's aggregate, P1's bit_identical, the follower's population, the calibration's accepted median). The fuzzer's unread class is being inverted: every unread measurement leaf under a clause's subtree is a GAP unless justified by name.
- 2026-09-14 [D7c] Astra merge review round 8 at 17dc09e: NO MERGE -- the coverage audit's exemptions excuse three banded measurements (B1's ci95 constituents and populations, identical draws, the sigma-1 stop's body constituents); the build-provenance check is a path prefix and must become a source fingerprint; contact counts are not mask identity. Sent back to the agent.

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
