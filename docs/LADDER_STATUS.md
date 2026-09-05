# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-06 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3, D7, D7b, D8, D9. In flight: D8b. Blocked: nothing.

## In flight

- **D8b** Captured segments that break the performer's own bone length — an Opus agent, since 2026-09-06

## Next up (unblocked, not started)

- **D4** A real pose solver (momentum on the MHR body) — an Opus agent

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D3's recorded miss: the exact-skeleton oracle's arm band (0.5 mm) fails by 0.19 mm on one of six synthetic bodies (D2's clavicle residual on a longer lever). Recommendation (2026-09-04): keep it as a standing fail; D5 re-derives the band from the fitted lever. And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-05 [D8] 2026-09-05, D8 post-merge: the head gate's first measurement on the repaired landmarks -- the shipped head candidate now PASSES its band (median 7.19 -> 6.56 deg, p95 22.19 -> 14.06; it read FAIL on every build since D2), because the head solve reads the smoothed points and the neck/shoulders under it stopped sliding along the A-C axis. The D3 gate's same-denominator clause reads FAIL by design (the smoothed landmarks moved; the raw array is the fixed reference from now on) and needs re-baselining in the next instrument pass. Reviewer's checks before the merge: the trunk-length and shoulder-width restorations are the sequence solve's own prior (weak); MAMMA's joints on the window through subject_map were the discriminating oracle; a right-hand 'regression against raw' on performer 0 was the raw point being wrong. The synthetic fixture could not select the ray-angle ceiling (exactly rigid and smooth, it prefers recovery at every angle) -- refuted prediction on method; the 150 deg and the 6-frame gap are registered as engineering limits. Reachability fired twice on rig landmarks and the gap clause held 0.0004 of slots: two near-inert rules a later step may retire. The MAMMA worker caps custom runs at four cameras (noted during the bear experiment).
- 2026-09-05 [D9] card written 2026-09-05 (Sol-reviewed; the floor measured for the actual operation and the leg shift predicted before the band); arms only; worktree ladder/D9 from 26bef65
- 2026-09-06 [D9] merged 2026-09-06 (1ff7a8b, --no-ff): the arm bones aimed from their own FK origin; elbows 13.7/13.9 -> 7.1/6.8 and 13.4/15.1 -> 9.1/8.1 mm, wrists 14.7/15.6 -> 8.9/10.4 and 17.6/20.1 -> 10.9/11.9, every cell within 1.6 mm of the floor measured for the actual operation; everything outside the four arm bones bit-identical; photographs up on all four arm cells; delivery rebuilt in place, 8/8 byte-identical; head gate byte-equal to the pre-D9 report
- 2026-09-06 [D9] 2026-09-06, D9 post-merge: (1) the D3 gate's same-denominator clause read FAIL against the pre-registered PASS; audited: the clause compares the shipped delivery against the gate's OWN cached rebuild under artifacts/compare/d3-skeleton/delivery, built on 2026-09-03 and never refreshed, so it has read FAIL since D8 for a stale cache, not for the landmarks -- measured directly, D9's smoothed landmarks are byte-identical to the D8 archive; the cache is being rebuilt from today's src (delivery and delivery-canonical, work/ COPIED) and the gate rerun. (2) tests/test_retarget_cost.py::test_converter_rotations_depend_on_bone_lengths_only_through_the_clavicle is red on main and was red on the D7 and D8 code too: its last clause asserts the root translation is independent of sizing, which stopped being true when D3 put the per-performer rest into the root formula; a pre-D3 contract, not D9's -- listed with the three uncommitted-test items as pre-existing. (3) The reviewer rewrote the two D2 tests that asserted the landmark-direction idiom to assert placement (elbow and wrist near their landmarks on a replaced frame; the swung control far off). (4) Finding handed on: project_generated_foot_contacts translates the root AFTER every aim (hoist p95 12.5/8.8 mm, above 0.5 mm on 67 and 19 of 150 frames; one rigid translation explains all four arm bones to 1e-4 mm) -- the same class as D7b's neck and D9's arms, one operation later: correct the feet before the aims and re-aim, or re-aim after; present since D2b. (5) The user's frame 117/118 on the falling performer: frames 110-122 the captured shoulder line reads 122-274 mm against his 363 with cameras A, C and D AGREEING to 1-11 px -- a well-conditioned triangulation of a detector-collapsed shoulder, which no geometric gate can see; the segment-length consistency reject dropped from D8's card is the next step (D8b).
- 2026-09-06 [D9] 2026-09-06, found during D9's close-out: the D3 gate's EXACT-SKELETON ORACLE has read legs 12.8-16.0 mm (band: 0.00) since the D7b close-out on 2026-09-05 -- oracle_worst_legs 0.0 at 6ea50f4 (D7), 16.01 at 5b010b5 (D7b), 12.77 since -- and every close-out log kept only the gate's verdict line, so it was read as the standing 0.19 mm arm miss. It is not the src: the D3-era code (fffe45f) reads 12.77 on today's inputs too. The oracle takes the DELIVERED rotations as its truth motion, re-posed on a perturbed rest, and rebuilds through positions_to_body_track with NO spine landmark, i.e. the legacy trunk-line hips frame; D7b moved the delivered neck off the trunk line, so the oracle's rebuilt hips frame no longer matches the motion it was given, the leg roots move with that frame (hip_half_span is perturbed by up to 27 %) and the landmark-aimed legs carry the displacement -- D9's mechanism on the legs, exposed by an instrument whose reconstruction is two steps behind the converter. Two repairs, one instrument-side (feed the oracle the spine landmark so the pelvis path runs, and log every gate line at close-out, not the last four), one converter-side (D9's leg dry run: aim the thighs and shins from their own origin, predicted 3-4 mm on the real take and exact on this oracle). Neither done tonight; recorded for the next step. The re-baselined D3 gate also shows 'CANONICAL UNCHANGED' and 'round trip reproduces D2c's figures' as FAIL: both compare against frozen D2c/D3-era references that D7, D7b and D9 moved by design (the round trip's arms improved 0.55 -> 0.17); the gate needs those references re-pinned, not the code.
- 2026-09-06 [D8b] card written 2026-09-06 (Sol-reviewed; effect measured and classified per camera: three cameras agree on the collapsed shoulder, a detector bias no geometric gate can see); worktree ladder/D8b from 82749e7

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
