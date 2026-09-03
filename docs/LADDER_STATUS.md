# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-03 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2. In flight: nothing. Blocked: nothing.

## In flight

- nothing in flight

## Next up (unblocked, not started)

- none

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D2 + D2b (branch ladder/D2, pushed, unmerged): the skeleton is better placed by every joint instrument (round trip 0.5/0.1 mm; hip offset 26/43 -> 0; vs MAMMA all joints 152/104 -> 71/68; ground hoist 142/110 -> 83/49) and the photographs say the mesh overlaps worse (IoU 0.585 -> 0.519: the root's share ~0.02/0.013, tilt-dependent, the pelvis has no frame of its own; the clavicle chain's ~0.04/0.02, present upright, binding suspected). Clavicle jitter 32/11 -> 40/33 frames over the ceiling is D2c's baseline, unaddressed. Options: (1) merge D2 + D2b as the skeleton fix with the three costs stated, D2c gating the jitter first, accepting a measured regression on the one photograph-referenced instrument with two thirds of its mechanism unproven; (2) hold until a pelvis frame (SOMA-77 has an unmapped pelvis root and two lower-spine joints, so it is a converter step, not a wait for markers) and the shoulder binding are addressed; (3) merge and accept both costs without D2c, not recommended. Recommendation: the skeleton is right and should not be un-fixed; the mesh's response is the next thing to instrument; the trade is yours.

## Recent log

- 2026-09-03 [D2] D2b started on ladder/D2 (the user's go, 2026-09-03): root placed so the rig's hip joints land on the captured hips, derived from the skeleton, no constant; Sol reviewed the plan. Gate adds absolute placement after the ground projection (hip offset per frame, rung 11 legs vs arms, I6 silhouette), a faithful-swap check against the measured variant, exact theorems, and a world-vertical shortcut as the discriminating control. D2c (the clavicle temporal step) is designed after D2b's gate.
- 2026-09-03 [D2] D2b gated on the branch (commits 4a90a2f, 9c186ad): the root derivation reproduces the measured variant to 0.00 mm, every theorem held, the round trip meets its band on both rigs (0.51/0.08 canonical, 0.07/0.04 sized, legs 0.00), the world-vertical shortcut and the sign flip fail, the horizontal hip offset goes 26/43 -> 0 mm and rung 11 against MAMMA goes 152/104 -> 71/68 mm on all joints. But the silhouette against the photographs FELL in 8 of 8 cells, 0.585 -> 0.519 median IoU, with the MAMMA oracle bit-identical, so the fall is real. The whole rig moved rigidly 28/43 mm horizontally, which is 80 mm x sin(trunk tilt): the rig has no pelvis frame apart from the torso, so on bent frames the fix pushes the body along the lean. Hypothesis under measurement: the fall grows with tilt (missing pelvis frame) or is flat (mesh shape, D5). D2b is blocked until that reads.
- 2026-09-03 [D2] D2b, the tilt hypothesis refuted by measurement: the geometry was exact (the rig shifts along the trunk axis by 80 mm x sin(tilt), dot +0.93/+0.90, 27.9/43.2 mm) but the silhouette fall is already there on upright frames (-0.045/-0.093 IoU where the body moved 10-13 mm) and does not ride the body's displacement on performer 1. The root-identical D2 build already costs 0.039 on performer 1's upright frames, so the fall travels with the ARMS, whose hands moved 144-188 mm. Reading under test: a person mask is blind to where inside the outline a limb sits; an arm tucked toward the torso counts as overlap, a correctly aimed arm on a rig 190 mm too wide at the shoulders lands outside the real one. Part-wise rasterisation and a folded-arm control decide it.
- 2026-09-03 [D2] D2b resolved as far as this fixture can take it (branch ladder/D2 at 0a2979a, pushed, unmerged). Three predictions refuted in a row and recorded: the lever would shorten (it lengthened), the tilt offset explains the whole silhouette fall (it explains only the root's share), a person mask is blind to a limb hidden inside the outline (arm precision ROSE under D2, and the folded-arm control did not realise the degenerate). What survived: the root's own silhouette cost, isolated exactly by translating D2's rendered mesh by the per-frame root delta, -0.022 / -0.013 IoU with intervals clear of zero, collapsing to -0.005 / ~0 on upright frames: the rig has one frame for the whole trunk and an anatomical pelvis tilts less. The clavicle chain carries about twice that, present upright, mechanism unresolved; under root-identical D2 the torso part falls on performer 1 through skin-weight bleed, so the shoulder-cap binding is the suspect. The picture: every joint instrument, MAMMA's and ours, says the skeleton is better placed; the photographs say the mesh bound to it overlaps worse; the discordance lives in what joins them. SOMA-77 carries an unmapped pelvis root and two lower-spine joints, so a pelvis frame is observable in principle.
- 2026-09-03 [D2] The user chose option 1 (2026-09-03): merge D2 + D2b as the skeleton fix with the silhouette costs stated, D2c gating the jitter first. D2c started on ladder/D2, Sol reviewed the plan: a physical reachability reject at the clavicle joint (accept a frame only if its clavicle local rotation is reachable from the last accepted one at 800 deg/s), slerp across rejected runs, then re-solve the downstream chains against the replaced parent (the converter's one loop becomes three passes). Selector: synthetic truth with the detector's measured heavy-tail noise (I7's fixture); frames over the ceiling are reported and never banded; must-fail controls: an over-smoother, a 5 deg/frame ceiling, the reject measured in world, a step test on a three-frame excursion; oracle: MAMMA's collar joints through the same measure.
- 2026-09-03 [D2] D2 shipped as three changes to the converter, merged 2026-09-03 (18573f1) and the delivery rebuilt in place from the cached detections, byte-identical to the gated build; the previous build is kept under artifacts/compare/delivered-before-d2-2026-09-03. Gates: artifacts/compare/d2-clavicle/gate.json (D2 and D2b), gate-d2c.json (D2c), review docs/reviews/clavicle-origin-2026-09-02.md. The clavicle is aimed from the rig's own shoulder origin (the 0.72 constant leaves), the root is placed so the rig's hip joints land on the captured hips (derived, no constant), and clavicle frames unreachable at 800 deg/s are rejected, bridged and the arm below re-solved. Round trip 0.55/0.08 mm on the arms, legs 0.00; delivered arms on our capture 181/218 -> 51/30 mm; against MAMMA's joints, all fifteen, 152/104 -> 71/68; the single-frame shoulder pop 164 -> 26 deg with the median step untouched; ground hoist 142/110 -> 83/49 mm. Facing and the head gate at window 9 unchanged. Costs stated: silhouette overlap 0.585 -> 0.521 (the skeleton is better placed by every joint measure, the mesh bound to it overlaps worse; the discordance lives in the binding and the missing pelvis frame), the median clavicle step (the lever) unchanged, and two of nineteen D2c bands fail on the instrument, kept as pre-registered: a lag band the unfixed arm fails too, and the round trip, which can no longer score the temporal step. Three refuted predictions on the record.

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
