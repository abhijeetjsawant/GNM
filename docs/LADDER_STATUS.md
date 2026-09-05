# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-05 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3, D7. In flight: D7b. Blocked: nothing.

## In flight

- **D7b** Re-solve the trunk chain onto the captured neck after the pelvis frame — an Opus agent, since 2026-09-05

## Next up (unblocked, not started)

- **D4** A real pose solver (momentum on the MHR body) — an Opus agent

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D3's recorded miss: the exact-skeleton oracle's arm band (0.5 mm) fails by 0.19 mm on one of six synthetic bodies (D2's clavicle residual on a longer lever). Recommendation (2026-09-04): keep it as a standing fail; D5 re-derives the band from the fitted lever. And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-04 [D7] 2026-09-04: D7 built and gated by the Opus agent on ladder/D7 (1bcc98b), NOT merged: the selector band B2a FAILS (world-vertical 7.32 deg beats the Kabsch pelvis fit 10.19; today's code 27.61). Passes: clean synthetic 0.0000 deg on every candidate; exact-recovery oracle 2e-6 deg with the no-spine must-fail at 34 deg; D3 closure on the rebuilt GLB from its own bytes 5e-7 m; canonical round trip 0.55/0.08 with legs and torso 0.00; legacy path bit-identical; head gate byte-equal; all 16 handedness signs unchanged; rigidity of root-Spine1 on the real take 5-11 mm sd against body controls 9-48. Two pre-registration errors are the coordinator's and Sol's, not the agent's: the card pointed at somaskel77-v1.json for rest geometry it does not carry (the rest is per performer; the shipped pitch is a convention with a 1.75-4.42 deg spread), and the effect size measured pelvis-vs-trunk-line (26 deg) where the world-vertical degenerate needed pelvis-vs-vertical (1-13 deg on every clip). Refuted predictions recorded: the world-vertical floor, the selector, the hoist bound (72.7->25.3 mm on performer 0, in the good direction), the Hips offset on performer 1, and the silhouette clauses, which were unmeasured at whole-person resolution (4 of 8 cells rose, all within 0.035 IoU; MAMMA's mesh bit-identical) and are being finished as part-wise tercile cuts. Rung 11 rerun on the D7 delivery: root and neck improve 15-20 mm on both performers against MAMMA, the 15-joint median 47->48 / 46->50. The window could not be selected (the lag and attenuation instruments are unusable at this noise) and is 0 by the pre-registered fallback.
- 2026-09-04 [D7] 2026-09-04, later: B7 finished on ladder/D7 (fccdbb1) as a wrapper over the committed silhouette instruments, D3 vs D7 rebuild, tilt terciles, block bootstrap on identical draws: performer 0 torso+legs +0.022 IoU on the bent tercile [0.004, 0.028], monotonic with tilt, arms unchanged; performer 1 flat on every cut; 2 of 7 pre-registered clauses fail (performer 1's bent tercile flat; performer 0's upright tercile rose where it was predicted to hold); MAMMA's mesh bit-identical. Post-hoc world-vertical control on the real take: performer 0 hoist 72.7 -> D7 25.3 -> constant 20.1 mm, Hips offset zeroed by construction; performer 1 hoist 35.5 -> D7 34.1 -> constant 60.2. Rung 11 folded per joint: root and neck improve 15-21 mm on both performers, nose worsens, 8 of 15 improve on each, the median moves the wrong way inside a gap. Overall verdict unchanged: FAIL on B2a. Not merged; the choice is recorded for the user.
- 2026-09-05 [D7] merged 2026-09-05 (6ea50f4, --no-ff) with the selector band B2a FAILED and stated; shipped on the pre-registered control rule (review section 7c), delivery rebuilt in place
- 2026-09-05 [D7] 2026-09-05: option (c) run on ladder/D7 (6666412): a pelvis frozen upright was rendered as a CONTROL delivery through the real build (nothing in src, hygiene byte-identical) and scored through the part-wise silhouette beside D3 and D7 on identical draws, with the merge rule fixed before the numbers. Rule: merge if the control beats D7 on neither performer with a CI clear of zero AND D7 beats the control on at least one. Both held: performer 0 D7-control +0.005 [-0.000, 0.012]; performer 1 D7-control +0.218 [0.116, 0.236] on the bent tercile (the control's hoist doubles, 36->60 mm, and its arms fall 0.061 with it -- P3 refuted: a root shift moves every vertex). P1 met (performer 0's gain is not-the-thorax, not the measured pelvis), P2 met, P4 met. The user delegated the call ('do whatever you recommend'); the coordinator took it on the rule; Sol reviewed the completed gate and approved. It is not a pass: B2a stays FAILED, B7 2 of 7 clauses stay failed, the window is 0 by refusal. Blindness recorded: the photographs caught the control's ROOT PLACEMENT through the pelvis frame, not the frame's orientation. Merged 6ea50f4 --no-ff; previous build kept under artifacts/compare/delivered-before-d7-2026-09-05; delivery rebuilt in place and checked byte-identical to the gated branch build.
- 2026-09-05 [D7] 2026-09-05, after the merge: drawing each build's DELIVERED skeleton from its own GLB over the footage (a range variant of camera_overlay plus a hand glTF reader, joints scored against the triangulated ones in absolute world) found what no gate reported: D7 moved the neck OFF its landmark, 14 -> 59 mm median on performer 0 (42 upright, 134 bent, r=+0.95 with trunk tilt) and 29 -> 44 on performer 1, while the hips went 19/17 -> 5/5 and the knees/ankles halved. Mechanism: the Spine joint hangs 117 mm off Hips in the PELVIS frame and the chest chain above keeps the thorax frame, so a pelvis pitched away from the trunk line translates the whole upper body and nothing re-solves the chain onto the captured neck -- the standing rule 'after replacing a parent, re-solve the chains below', not applied to the spine. D3 already overshot the neck on bent frames (108 mm past 30 deg on performer 0) because the trunk chain is straight and rigid. The delivered-vs-capture instrument (retarget_cost.py) is BLIND to D7: it re-solves the track through the converter with no spine landmark and reads D3's torso figure (14.46 / 26.21) on the D7 delivery unchanged; against MAMMA the neck 'improved' (112 -> 91) on a different convention. The only figure that saw it was read from the file's own bytes. Follow-up step needed (D7b): re-solve the spine chain to the captured neck after the pelvis frame, and make retarget_cost read the delivered file. Also seen plainly in the mesh videos: the skin, stretched onto the per-performer skeleton with the asset's old weights since D3, balloons at the hips on lying frames and tears at the thigh in the squat; the silhouette's rise hid it (precision fell on performer 1 while recall rose, the dilated-blob pattern). D6 must come before more skeleton work, with a mesh-distortion instrument first.
- 2026-09-05 [D7b] card written 2026-09-05 (Sol-reviewed; floor and lever measured from the delivered files before the band); worktree ladder/D7b from 4acf170

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
