# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-04 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3. In flight: nothing. Blocked: D7.

## In flight

- nothing in flight

## Next up (unblocked, not started)

- **D4** A real pose solver (momentum on the MHR body) — an Opus agent

## Blocked

- **D7** A pelvis frame of its own (the trunk no longer one rigid block) — blocked on: user decision: merge with the B2a fail stated, or hold for a fixture with real pelvic tilt (lane H)

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D7 (the pelvis frame) is built on branch ladder/D7 (fccdbb1) and FAILS its primary pre-registered band: on the noisy synthetic selector a pelvis frozen bolt upright (the pre-registered degenerate) scores 7.3 deg where the candidate scores 10.2 and today's shipped code 27.6 -- because on all five synthetic clips the true pelvis sits only 1-13 deg from vertical, so 'always upright' is near-truth on that motion source (a fixture property, unmeasured before the band). The silhouette, part-wise on tilt terciles, rises on performer 0 (+0.022 IoU torso+legs on bent frames, monotonic with tilt) and is flat on performer 1; 2 of 7 clauses fail. Against MAMMA per joint (reported): root 37->22 / 40->23 mm, neck 112->91 / 108->89, nose worse, the 15-joint median 47->48 / 46->50. A post-hoc upright-constant control on the real take beats D7 on performer 0's hoist (20 vs 25 mm, from 73) and zeroes the Hips offset by construction, but nearly doubles performer 1's hoist (36->60), so performer 0's gain is not attributed between 'measured pelvis' and 'not the thorax'. Choose: (a) merge with the failure stated, as D2 was; (b) hold until lane H delivers a take with real pelvic tilt; (c) first render the upright-constant delivery through the silhouette (the cheapest discriminator) and decide on that. Re-gating at the detector's measured noise would still tie at performer 1's level, so it is not a route to a pass.
- D3's recorded miss: the exact-skeleton oracle's arm band (0.5 mm) fails by 0.19 mm on one of six synthetic bodies (D2's clavicle residual on a longer lever). Recommendation (2026-09-04): keep it as a standing fail; D5 re-derives the band from the fitted lever. And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-04 [D4] D3 done, so D4 is unblocked on the ladder; it still needs pymomentum installed (a throwaway venv per tools/fitter/README.md: uv venv /tmp/momenv --python 3.12; uv pip install pymomentum-cpu; MHR assets from the MHR release, never the SOMA redistribution) -- installed in neither python on this machine on 2026-09-03.
- 2026-09-04 [D7] 2026-09-04: the brief left the D4-or-D7 choice as '[choose one]'; taken as delegated and D7 chosen first, reversible: SOMA-77's pelvis root is already mapped as 'root' (only Spine1/Spine2 are unmapped), no new dependency, it closes a measured cost (the root's silhouette share, -0.022/-0.013 IoU, tilt-dependent), and D4's card needs restating after D3. pymomentum installed in /tmp/momenv and the MHR assets copied to .cache/mhr/assets, so D4 is unblocked next. Effect size measured on the fixture's own clips before the card: the rigid pelvis (root->Spine1) departs from the trunk line by 26/33 deg median/p95 on the squat clip's bent frames, correlation +0.93 with tilt. Sol reviewed the card: clean arm per candidate, partwise silhouette (the translation trick over-attributes here), length stability not self-agreement, no per-frame frame switching, the tilt-correlation figure restated on the Hips joint.
- 2026-09-04 [D7] worktree ladder/D7; card in LADDER_EXECUTION_PLAN section 2; Sol reviewed the plan 2026-09-04
- 2026-09-04 [D7] built, gated and FAILING its primary pre-registered band on branch ladder/D7 (1bcc98b); not merged
- 2026-09-04 [D7] 2026-09-04: D7 built and gated by the Opus agent on ladder/D7 (1bcc98b), NOT merged: the selector band B2a FAILS (world-vertical 7.32 deg beats the Kabsch pelvis fit 10.19; today's code 27.61). Passes: clean synthetic 0.0000 deg on every candidate; exact-recovery oracle 2e-6 deg with the no-spine must-fail at 34 deg; D3 closure on the rebuilt GLB from its own bytes 5e-7 m; canonical round trip 0.55/0.08 with legs and torso 0.00; legacy path bit-identical; head gate byte-equal; all 16 handedness signs unchanged; rigidity of root-Spine1 on the real take 5-11 mm sd against body controls 9-48. Two pre-registration errors are the coordinator's and Sol's, not the agent's: the card pointed at somaskel77-v1.json for rest geometry it does not carry (the rest is per performer; the shipped pitch is a convention with a 1.75-4.42 deg spread), and the effect size measured pelvis-vs-trunk-line (26 deg) where the world-vertical degenerate needed pelvis-vs-vertical (1-13 deg on every clip). Refuted predictions recorded: the world-vertical floor, the selector, the hoist bound (72.7->25.3 mm on performer 0, in the good direction), the Hips offset on performer 1, and the silhouette clauses, which were unmeasured at whole-person resolution (4 of 8 cells rose, all within 0.035 IoU; MAMMA's mesh bit-identical) and are being finished as part-wise tercile cuts. Rung 11 rerun on the D7 delivery: root and neck improve 15-20 mm on both performers against MAMMA, the 15-joint median 47->48 / 46->50. The window could not be selected (the lag and attenuation instruments are unusable at this noise) and is 0 by the pre-registered fallback.
- 2026-09-04 [D7] 2026-09-04, later: B7 finished on ladder/D7 (fccdbb1) as a wrapper over the committed silhouette instruments, D3 vs D7 rebuild, tilt terciles, block bootstrap on identical draws: performer 0 torso+legs +0.022 IoU on the bent tercile [0.004, 0.028], monotonic with tilt, arms unchanged; performer 1 flat on every cut; 2 of 7 pre-registered clauses fail (performer 1's bent tercile flat; performer 0's upright tercile rose where it was predicted to hold); MAMMA's mesh bit-identical. Post-hoc world-vertical control on the real take: performer 0 hoist 72.7 -> D7 25.3 -> constant 20.1 mm, Hips offset zeroed by construction; performer 1 hoist 35.5 -> D7 34.1 -> constant 60.2. Rung 11 folded per joint: root and neck improve 15-21 mm on both performers, nose worsens, 8 of 15 improve on each, the median moves the wrong way inside a gap. Overall verdict unchanged: FAIL on B2a. Not merged; the choice is recorded for the user.

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
