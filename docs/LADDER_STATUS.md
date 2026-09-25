# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-25 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3, D4, D4b, D4c, D4d, D4i, D7, D7b, D8, D9, D8b, D8c, D9b, D7c. In flight: D4i-b. Blocked: nothing.

## In flight

- **D4i-b** The default flip, re-registered with the one environment value named — an Opus agent, since 2026-09-25

## Next up (unblocked, not started)

- none

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D3's recorded miss, CLOSED at D7c's close-out (2026-09-15): the exact-skeleton oracle's arm band (0.5 mm) PASSES on every seed for the first time (worst arms 0.35 mm, legs 0.00) once the pelvis is exact -- the 2.72 mm since D9b was the D7 convention seen through the leg-root-aligned gauge. The band was never moved. Still owed to the instrument-debt step: the gate's translation-aligned gauge and its frozen D2c/D3 'canonical unchanged' and 'same denominator' clauses (moved by design since D7). And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-25 [D4d] D4d MERGED 2026-09-25 (35abd4d): PASS. D4c's landmark start (its deferred tooling merge discharged) + a second full calibration pass (max_iter 30 each, tracking untouched). Phase 1: WARM held on D4c's 12 (worst 0.214x) and TWO-PASS closed all 12 (worst 0.726x; 20261106/d0 1.103 -> 0.676x), so the frozen rule selected it. Phase 2 (12 fresh): L 12/12, must-fails i-iv, B1 +0.152/+0.116 over D7c, B2, hygiene PASS. Astra's permitted claim: the band holds on 12 fresh identities under two donor motions and the four burned one-pass misses close; fresh acceptance does NOT show improvement over D4c (no donor-0 spine below -0.40 was drawn and the one-pass arm also passed). Reported, not banded: the real-take spine 1.102 past its 1.1 limit, residual +1 mm, B1 vs D4c -0.0039 (CI < 0, performer 0).
- 2026-09-25 [D4] D4 CLOSED 2026-09-25: its O1 FAIL (1.030 mm, 2026-09-22) stays on record and is superseded, by the registered disposition, by D4d's PASS on the combined fitter (D4c's landmark start + a second calibration pass), with B1 re-shown on that fitter (+0.152/+0.116 over D7c). --body mhr carries the combined fitter; the default stays rig until the integration step flips it.
- 2026-09-25 [D4c] D4c's record is unchanged: FAIL on L (1 of 12 at 1.10x). Its fitter change and tooling reached main only through D4d's PASS merge (35abd4d).
- 2026-09-25 [D4i] D4i dispatched 2026-09-25: the default flip to --body mhr plus the instrument-roster migration (N5.1 scoped out: nothing in src/ reads this delivery). Oracle: the default build = D4d's gated MHR delivery byte-identical, B1 reproduced exactly. A three-field roster by a frozen rule; must-fails i-v; the coordinator edits post_merge.sh/ladder.py ON THE BRANCH before the one merge review; the close-out outcomes and the full rollback frozen. Astra's one card round at medium: two blockers, adopted.
- 2026-09-25 [D4i] D4i FAIL (oracle), 2026-09-25, records only on main (96f0bc9); the candidate (the flip, roster tooling, schema-aware silhouette/verifier, build refusals, bootstrap, the coordinator's post_merge.sh and ladder.py) pinned at tag ladder/D4i-fail-b073253. The default on main stays rig. The oracle failed only on body_model.assets in both track JSONs (the fitter writes its checkout's absolute path; the other 10 per-subject files byte-identical) -- the coordinator's pre-registration error, the lane's sixth. Every other conjunct held: B1 reproduced exactly, B2, closure, hygiene 8/8, must-fails i-v (70 cases), the frozen population, the three-field roster (head_gate/bootstrap_margin removed as class d), the migration recorded. Astra: D4i-b is a legitimate new registration on stated terms, and must also make the close-out enforce itself and repair three gate holes.
- 2026-09-25 [D4i-b] D4i-b dispatched 2026-09-25: branch = main + tag ladder/D4i-fail-b073253; the oracle re-stated with /body_model/assets named (it equals the building checkout's git toplevel + /.cache/mhr/assets, the whole remaining JSON structure equal, stated as informed by D4i); fresh evidence only; B1 arm renamed under a frozen map; per-cell rows with D4d's reference RECONSTRUCTED from hash-bound inputs; an enforcing close-out with a scripted FULL rollback demonstrated (git revert in a disposable checkout + a scratch delivery tree); three gate repairs; must-fails vi-viii. Astra's one card round at medium: one blocker, adopted.

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
