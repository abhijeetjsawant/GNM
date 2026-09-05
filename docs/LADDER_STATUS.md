# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-05 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3, D7, D7b. In flight: D8. Blocked: nothing.

## In flight

- **D8** The captured limbs under occlusion: fix the yellow where two bodies overlap — an Opus agent, since 2026-09-05

## Next up (unblocked, not started)

- **D4** A real pose solver (momentum on the MHR body) — an Opus agent

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D3's recorded miss: the exact-skeleton oracle's arm band (0.5 mm) fails by 0.19 mm on one of six synthetic bodies (D2's clavicle residual on a longer lever). Recommendation (2026-09-04): keep it as a standing fail; D5 re-derives the band from the fitted lever. And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-05 [D7] 2026-09-05: option (c) run on ladder/D7 (6666412): a pelvis frozen upright was rendered as a CONTROL delivery through the real build (nothing in src, hygiene byte-identical) and scored through the part-wise silhouette beside D3 and D7 on identical draws, with the merge rule fixed before the numbers. Rule: merge if the control beats D7 on neither performer with a CI clear of zero AND D7 beats the control on at least one. Both held: performer 0 D7-control +0.005 [-0.000, 0.012]; performer 1 D7-control +0.218 [0.116, 0.236] on the bent tercile (the control's hoist doubles, 36->60 mm, and its arms fall 0.061 with it -- P3 refuted: a root shift moves every vertex). P1 met (performer 0's gain is not-the-thorax, not the measured pelvis), P2 met, P4 met. The user delegated the call ('do whatever you recommend'); the coordinator took it on the rule; Sol reviewed the completed gate and approved. It is not a pass: B2a stays FAILED, B7 2 of 7 clauses stay failed, the window is 0 by refusal. Blindness recorded: the photographs caught the control's ROOT PLACEMENT through the pelvis frame, not the frame's orientation. Merged 6ea50f4 --no-ff; previous build kept under artifacts/compare/delivered-before-d7-2026-09-05; delivery rebuilt in place and checked byte-identical to the gated branch build.
- 2026-09-05 [D7] 2026-09-05, after the merge: drawing each build's DELIVERED skeleton from its own GLB over the footage (a range variant of camera_overlay plus a hand glTF reader, joints scored against the triangulated ones in absolute world) found what no gate reported: D7 moved the neck OFF its landmark, 14 -> 59 mm median on performer 0 (42 upright, 134 bent, r=+0.95 with trunk tilt) and 29 -> 44 on performer 1, while the hips went 19/17 -> 5/5 and the knees/ankles halved. Mechanism: the Spine joint hangs 117 mm off Hips in the PELVIS frame and the chest chain above keeps the thorax frame, so a pelvis pitched away from the trunk line translates the whole upper body and nothing re-solves the chain onto the captured neck -- the standing rule 'after replacing a parent, re-solve the chains below', not applied to the spine. D3 already overshot the neck on bent frames (108 mm past 30 deg on performer 0) because the trunk chain is straight and rigid. The delivered-vs-capture instrument (retarget_cost.py) is BLIND to D7: it re-solves the track through the converter with no spine landmark and reads D3's torso figure (14.46 / 26.21) on the D7 delivery unchanged; against MAMMA the neck 'improved' (112 -> 91) on a different convention. The only figure that saw it was read from the file's own bytes. Follow-up step needed (D7b): re-solve the spine chain to the captured neck after the pelvis frame, and make retarget_cost read the delivered file. Also seen plainly in the mesh videos: the skin, stretched onto the per-performer skeleton with the asset's old weights since D3, balloons at the hips on lying frames and tears at the thigh in the squat; the silhouette's rise hid it (precision fell on performer 1 while recall rose, the dilated-blob pattern). D6 must come before more skeleton work, with a mesh-distortion instrument first.
- 2026-09-05 [D7b] card written 2026-09-05 (Sol-reviewed; floor and lever measured from the delivered files before the band); worktree ladder/D7b from 4acf170
- 2026-09-05 [D7b] merged 2026-09-05 (5b010b5, --no-ff): neck back on its landmark to the trunk-length floor (21.5/18.3 mm whole take, 42.1/11.9 bent tercile; D7 read 58.9/44.0), root and legs bit-identical, photographs not worse on either performer; B5's head clause FAILED against a 1e-9 deg band below the float32 floor (5e-6, the oracle reads the same on D7) and stated; delivery rebuilt in place, 8/8 files byte-identical to the branch build; every instrument rerun
- 2026-09-05 [D7b] 2026-09-05, D7b post-merge: the placement instrument that saw the D7 neck drift is now committed (tools/compare/delivered_vs_capture.py) and on rung 7; retarget_cost.py stays blind to this class. Rung 11's neck row vs MAMMA worsens (91->96 / 89->98) while the nose row improves (132->74 / 103->69): MAMMA's neck is not SOMA-77's Neck2, a convention disagreement, both shown. Handed to D5: the trunk shortens under flexion, 42 mm left on performer 0's bent frames, of which an equal-split flexion over Spine/Chest/UpperChest recovers 32. Coordinator errors recorded: the brief named commercial-multiview-b2/work as the rebuild source (the right one is commercial-multiview-soma77/work); Sol's 1e-9 deg head band was below the storage floor, the lane's third pre-registration error.
- 2026-09-05 [D8] card written 2026-09-05 (Sol-reviewed; effect measured and classified per camera before the band: B/D dropout leaves A-C at 172 deg, depth free along their axis); worktree ladder/D8 from 18f6cf8; runs before D6 and D5

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
