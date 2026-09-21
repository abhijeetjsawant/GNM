# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-22 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3, D7, D7b, D8, D9, D8b, D8c, D9b, D7c. In flight: D4. Blocked: nothing.

## In flight

- **D4** A real pose solver (momentum on the MHR body) — an Opus agent, since 2026-09-21

## Next up (unblocked, not started)

- none

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D3's recorded miss, CLOSED at D7c's close-out (2026-09-15): the exact-skeleton oracle's arm band (0.5 mm) PASSES on every seed for the first time (worst arms 0.35 mm, legs 0.00) once the pelvis is exact -- the 2.72 mm since D9b was the D7 convention seen through the leg-root-aligned gauge. The band was never moved. Still owed to the instrument-debt step: the gate's translation-aligned gauge and its frozen D2c/D3 'canonical unchanged' and 'same denominator' clauses (moved by design since D7). And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-15 [D7c] COORDINATOR DECISION (user: 'enough rounds'): D7c merges on the nine Astra merge rounds recorded. The candidate has not changed since round 2; rounds 3-9 were the gate instrument. Round 10 (the bounded question) was sent and stopped before it answered; the residual gate findings and the six retrospective provenance stamps are instrument debt with the fuzz artifact (gate-fuzz.json: 18,934 paths, 5,170 enforced, 0 gaps) as their record. Merging ladder/D7c at 65e3091.
- 2026-09-15 [D7c] merged 2026-09-15 (53ab3b0, --no-ff) on the nine recorded Astra merge rounds, the user having bounded the review; E_rig_rest_kabsch ships: the pelvis fitted to the rig's own rest offsets about the captured hip midpoint, no constant, SOMA's root landmark no longer read. Oracle: 6.865 deg -> 0.0001 deg, Spine 21-28 -> 0.0001 mm, torso 9-12 -> 0.00, legs within 0.08 mm, contacts identical; the selector stopped twice as pre-registered (sigma-1.0 follower 5 of 6 under 2x; the calibration's first monotonicity wording over a 0.0135 mm dip) and at the calibrated sigma 0.335546875 chose (a) in all six cells with the follower 2.56-3.20x on every body; take: pitch -8.8/-9.2 deg, root 12.4/13.1 mm, photographs not worse 8/8 with three torso cells rising, same denominator PASS, P1/P2 on the take and every oracle body; performer 1's spine lever broken on 29 frames, guarded. Close-out: rebuilt in place 8/8 byte-identical, every instrument logged under artifacts/compare/post-merge-D7c; THE D3 GATE'S EXACT-SKELETON ORACLE PASSES FOR THE FIRST TIME (legs 0.00, arms 0.35 mm worst against the 0.5 band, a standing fail since D3); its frozen D2c/D3 reference clauses still read moved-by-design; D7c's own gate MERGE on fourteen conjuncts with the two stops on their bands. Open: the pelvis convention (lane H); the mesh-deformation proxy and Kavan eq. 17 (D6); performer 1's unattributed torso rise; both amendments post hoc; gate hardening and six retrospective provenance stamps as instrument debt. NEXT, by the user's steer: the body model in the delivery path (D4 on MHR with D5's scaling), the silhouette against MAMMA's mesh pre-registered as the band.
- 2026-09-21 [D4] Body-model pre-card (2026-09-21, coordinator, no agent, no card): MHR fitted by momentum to the delivered raw landmarks and scored on the committed silhouette instrument beside the D7c rig -- pooled IoU 0.647/0.652 -> 0.789/0.730, paired CI clear of zero on both performers (PASS under the pre-registered rule; the route is not wrong). Most of the gain is the body MODEL (MHR's mean body posed by our tracker already reads 0.744/0.722); the fit adds +0.045 on performer 0, +0.007 (CI through zero) on performer 1. Caveats: raw vs repaired input denominator, lod6 mesh, MAMMA oracle not rerun. .cache/ was found WIPED (mamma fixture, SMPL-X, fixtures gone): no rebuild and no MAMMA oracle until restored. docs/reviews/body-model-precard-2026-09-21.md.
- 2026-09-21 [D4] card written 2026-09-21 after the pre-card measurement (MHR fitted 0.789/0.730 vs the rig 0.647/0.652, CI clear both performers); the user chose shape (a): MHR's own mesh is the delivered body; one Astra card review by the 2026-09-15 rule; .cache restored from the Modal volume (SMPL-X, the four fixture videos) and the pinned MAMMA repo (the calibration yaml)
- 2026-09-21 [D4] D4 dispatched 2026-09-21: Astra's one card round (three blockers adopted: the O1 fixture on MHR's native scale channels with the 26 flexible length channels frozen, B1's comparators named with the band = fitted minus D7c lower CI > 0 on both performers, B2 on the consumed input; the rest instrument debt); worktree .claude/worktrees/ladder-D4 on ladder/D4 at 803f111, one Opus agent.
- 2026-09-22 [D4] D4 STOPPED at O1 (ladder/D4 5ebdc28): the exactness oracle reads 0.80-1.03 mm over six seeds against the 1 mm band (FAIL on one seed); the fixture's donor pose violated 23 of MHR's configured limits and was clamped as a fixture parameter (1.25 -> 1.03); what remains is the tracker's own floor (truth identity handed in: 0.51-0.76 mm on exact data) plus scale_spine_length shrunk 14-20 % by momentum's default soft limit and scale_foot_length unidentifiable without toe landmarks. The must-fail (mean body) rejects by 9-39x; hygiene 8/8; the pre-card reproduced to 0.0 on all 600 cells. FINDING: momentum's calibrate_markers mutates a later Character.load_fbx in the same process -- the pre-card's performer-1 figure was contaminated (0.7295 -> 0.7503 repaired). COORDINATOR DECISION: the 1 mm band was set without measuring the instrument floor (the lane's recorded pre-registration error, fourth time); the band is NOT moved, O1 is recorded as a standing FAIL attributed to floor + regulariser, and the step continues to the delivery and B1-B5 under the D3 precedent (a stated, attributed oracle fail beside the photograph band); the merge rule's O1 conjunct becomes a recorded exception, decided at the merge on B1.

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
