# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-25 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3, D4, D4b, D4c, D4d, D7, D7b, D8, D9, D8b, D8c, D9b, D7c. In flight: nothing. Blocked: nothing.

## In flight

- nothing in flight

## Next up (unblocked, not started)

- **D4i** The body model becomes the default delivery (the integration step) — an Opus agent

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D3's recorded miss, CLOSED at D7c's close-out (2026-09-15): the exact-skeleton oracle's arm band (0.5 mm) PASSES on every seed for the first time (worst arms 0.35 mm, legs 0.00) once the pelvis is exact -- the 2.72 mm since D9b was the D7 convention seen through the leg-root-aligned gauge. The band was never moved. Still owed to the instrument-debt step: the gate's translation-aligned gauge and its frozen D2c/D3 'canonical unchanged' and 'same denominator' clauses (moved by design since D7). And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-25 [D4c] D4c dispatched 2026-09-25: ONE change, the calibration's landmark-derived starting identity (max_iter 30 kept); limit-aware drawn-set rule with the trunk a PRECONDITION (STOP); development on the 18 burned fixtures chooses only the trunk statistic from {median, p90, p95}, frozen before 12 untouched acceptance fixtures; init-only must-fail; B1 re-run on the D4c fitter; one verdict; fit_one merges only on PASS. Astra's one card round at medium: dispatchable, findings adopted. Worktree .claude/worktrees/ladder-D4c on ladder/D4c
- 2026-09-25 [D4c] D4c FAIL (L), 2026-09-25, records only on main (5d27b4f); the tooling and the landmark-start fitter change pinned at tag ladder/D4c-fail-1a89cc7 (1a89cc7), NOT merged (the card: fit_one does not merge on a non-PASS; the tooling imports it, so its merge is deferred debt, Astra's option c). Precondition 0 held (limit-aware rule drew 8, spine included); stage 0b 18/18; development chose p90 (worst trunk ratio 11.93 -> 1.27); acceptance L 11/12 -- the trunk on 20261106/d0 read 2.1341 vs 1.9352 mm (1.10x; the p90 chord read a strongly shortened spine LONG by 0.032 units and 30 iterations recovered a third); must-fails i-iv hold; B1 +0.1555/+0.1178 vs the D7c rig, D4c-D4 ~0; B2 and hygiene PASS. D4 stays open; fixtures 20261101-06 x donors 0/1 now burned.
- 2026-09-25 [D4d] D4d dispatched 2026-09-25: branch = main + tag ladder/D4c-fail-1a89cc7; Phase 1 on 30 burned fixtures (WARM decides the fork; TWO-PASS = a second full calibration pass from pass 1's identity, max_iter 30 each, tracking untouched; SW* report); Phase 2 on 12 untouched fixtures only if the frozen rule selects TWO-PASS; one verdict; PASS merges D4c+D4d, otherwise records only. Astra's one card round at medium: dispatchable, findings adopted. SOMA-77 detector restored and verified.
- 2026-09-25 [D4d] D4d MERGED 2026-09-25 (35abd4d): PASS. D4c's landmark start (its deferred tooling merge discharged) + a second full calibration pass (max_iter 30 each, tracking untouched). Phase 1: WARM held on D4c's 12 (worst 0.214x) and TWO-PASS closed all 12 (worst 0.726x; 20261106/d0 1.103 -> 0.676x), so the frozen rule selected it. Phase 2 (12 fresh): L 12/12, must-fails i-iv, B1 +0.152/+0.116 over D7c, B2, hygiene PASS. Astra's permitted claim: the band holds on 12 fresh identities under two donor motions and the four burned one-pass misses close; fresh acceptance does NOT show improvement over D4c (no donor-0 spine below -0.40 was drawn and the one-pass arm also passed). Reported, not banded: the real-take spine 1.102 past its 1.1 limit, residual +1 mm, B1 vs D4c -0.0039 (CI < 0, performer 0).
- 2026-09-25 [D4] D4 CLOSED 2026-09-25: its O1 FAIL (1.030 mm, 2026-09-22) stays on record and is superseded, by the registered disposition, by D4d's PASS on the combined fitter (D4c's landmark start + a second calibration pass), with B1 re-shown on that fitter (+0.152/+0.116 over D7c). --body mhr carries the combined fitter; the default stays rig until the integration step flips it.
- 2026-09-25 [D4c] D4c's record is unchanged: FAIL on L (1 of 12 at 1.10x). Its fitter change and tooling reached main only through D4d's PASS merge (35abd4d).

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
