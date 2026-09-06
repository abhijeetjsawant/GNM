# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-07 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3, D7, D7b, D8, D9, D8b, D8c, D9b. In flight: nothing. Blocked: nothing.

## In flight

- nothing in flight

## Next up (unblocked, not started)

- **D4** A real pose solver (momentum on the MHR body) — an Opus agent

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D3's recorded miss, restated at D9b (2026-09-07): the exact-skeleton oracle's arm band (0.5 mm) reads 2.72 mm since D9b, up from 0.80-1.17, because retarget_cost.score aligns each frame on the leg-root midpoint and so charges a correctly re-aimed arm with the hoist's perpendicular part (the gauge, not the candidate; the absolute-frame companion row in d9b_hoist_gate.py improves on every seed). Recommendation: keep it a standing fail, do not move the band; the instrument-debt step re-pins the gate's gauge and its frozen D2c/D3 references together. And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-06 [D8c] D8c card written and reviewed by Grok 4.6 (Cursor) in Sol's place (docs/reviews/hip-line-grok-review-2026-09-06.md). Measured before the card: performer 1's hip line fails in three runs and two classes -- 110-119 and 84-86 an inward collapse along the hip line in three agreeing cameras (D8b's class; frame 113 an outward raw spike; 84-86 one hip at a time), 158-168 a two-view stretch along the A-C baseline while he lies on the floor with his hips pointed at the only two cameras that see him (six of eleven frames over the ceiling, the other five at +9-13 % stay). Decided on data after the review: a per-hip root->hip rule is refused (the pelvis landmark spreads +25 % on honest frames and would fire 35/17 times), both hips charged, 84-86 registered as a known over-charge. The root's dependence on the hip midpoint is pre-registered as reports R1-R4 (hoist-removed, Savitzky-Golay +-4 envelope, R_hips delta), the synthetic injects the MEASURED mode (along the hip line toward its midpoint, knees untouched) and scores through the converter, and B4 on D8b's fixed medians is in the merge rule paired with the photographs as the guard against a recovery that lets the agreeing rays win.
- 2026-09-06 [D8c] worktree .claude/worktrees/ladder-D8c on ladder/D8c, one Opus agent; instrument first, then synthetic, then the one-row src change
- 2026-09-06 [D8c] merged 2026-09-06 (4c1c6e2, --no-ff) on the pre-registered clauses, all seven conjuncts: the hip line joins the segment-length rule with both hips charged; the synthetic confirmed demote at 0.15 on the hips' own geometry (an injected collapse 50.3 -> 3.9 mm); on the take performer 1's hip line off his own width 23 -> 0 on D8b's fixed medians, every other segment's fires identical frame for frame, raw byte-identical, photographs up +0.021 / +0.013 IoU on the two twisting runs and level on the grass run (the error lies along the two cameras' own line of sight, where a silhouette is blind); the delivered root moved 3 mm median on the repaired frames, 23 at worst, the pelvis frame 2.1 deg. Two predictions FAILED and attributed: MAMMA further (its hip line is a constant 118 mm, so the collapse was accidentally nearer it), and the delivered joints move 1-2 mm far from any fire because D3's per-performer rest is sized over the whole take (16 of 55 rest bones move under 1 mm when 18 frames change) -- a lane finding: no later step may predict a DELIVERED change to be local. Grok 4.6 reviewed the merge in Sol's place (docs/reviews/hip-line-grok-merge-review-2026-09-06.md): MERGE, recording that S0's calibration bound mixed the two performers (-9.2/+8.5 where the card froze -8.2/+8.5; the largest scale inside the frozen bound is 0.10, not 0.15; the ceiling did not move and the mode is demote at every scale run) and that frame 113 was a card error (a two-view slot D8 had already withheld, not a fire). Close-out: delivery rebuilt in place 8/8 byte-identical, every instrument rerun with full logs under artifacts/compare/post-merge-D8c; D3 oracle unchanged (legs 0.07, arms 1.17), same-denominator clauses read moved-by-design, head gate line-identical. Open after this step: the one-hip over-charge (three frames), no fixture realises a two-view depth stretch, the ceiling's own remedy, the window block's frame-id labelling repaired in the instrument.
- 2026-09-07 [D9b] D9b card written (LADDER_EXECUTION_PLAN section 2, after D8c) and measured before it from the shipped D8c delivery's own bytes: the foot-contact hoist (two recoveries agreeing to 0.0 mm) is p95 12.5 / 8.7 mm on 67 / 22 of 150 frames, and every root-dependent bone -- four arm bones, both clavicles, the Spine->Neck chain -- sits at 0.0 mm on its ray from the PRE-hoist origin and 3.7-6.3 mm off from the delivered one; the legs (landmark-to-landmark) do not and go to D9-legs. Mechanism pre-registered: re-solve the root-dependent chains after the projection (the D2c rule), the projection run once on today's feet, never twice. ORACLE AUDIT: contacts fire on all six D3 gate bodies (34-67 hoisted frames, p95 10-14 mm) and retarget_cost.score aligns each frame on the leg-root midpoint, so the gate reads identical before/after/hoist-removed on every seed, is blind to this defect, and is predicted to read the correct fix WORSE on arms and torso (registered as the gauge, band not moved; an absolute-frame companion row is added). The brief's 'exact rig with no ground contact' clause did not exist and is replaced by O1-O4. Pre-card scripts committed under tools/compare/precard/. Grok 4.6 reviewed the card in Sol's place (docs/reviews/hoist-reaim-grok-review-2026-09-07.md): dispatch after eight patches, all adopted and verified against the source -- a dedicated re-solve (never a pass-C re-run, which would overwrite the projection's foot locks), the quaternion hemisphere pass not re-run globally, O3 stated as the median it is, B1 as the closure quantity merged only with B2/B4/the tripwire, a pass-B accepted-set change is a hold not a pre-scoped exception, the oracle's Head inherits the trunk, the tripwire's meaning, two more must-fails.
- 2026-09-07 [D9b] worktree .claude/worktrees/ladder-D9b on ladder/D9b, one Opus agent; hygiene, instrument first (and the oracle audit reproduced), the tripwire, then the one-block src change
- 2026-09-07 [D9b] 2026-09-07, D9b merged (9cab14b, --no-ff) on all eight pre-registered conjuncts: the root-dependent chain (trunk, neck, clavicles with pass B rerun, arms, hands) re-solved from the hoisted root after the foot-contact projection; every such bone on its ray from the DELIVERED origin to 0.0003 mm (was 3.7-6.3 mm median on the 67 / 22 hoisted frames, 18 mm at worst); root, contacts, hips, legs, feet, toes byte-identical, everything byte-identical on the 83 / 128 unhoisted frames; the tripwire (hoist forced to zero, old src vs new) 8 of 8 byte-identical; photographs level within 0.001 IoU on 8 of 8 cells; pass B's accepted set unchanged. Prediction FAILED and attributed: the D3 gate's aligned oracle median did not hold (arms 1.17 -> 2.70, close-out reads 2.72) because retarget_cost.score subtracts the leg-root offset per frame -- the gauge, band untouched, an absolute-frame companion row improves on every seed. Grok 4.6 reviewed the merge in Sol's place (MERGE stands; the skip-second-pass-B hole is untestable on this take). Close-out: rebuilt in place 8/8 byte-identical, every instrument logged under artifacts/compare/post-merge-D9b; D3 gate lines as predicted (legs 0.07, arms 2.72, the frozen D2c/D3 references still moved-by-design); head gate line-identical to D8c's close-out. Two structural test pins re-pinned in place (the sequence rule runs twice per clavicle; the _joint_origin pin follows the aims into the helpers). Handed forward: the legs' 4.6-9.2 mm ray miss on hoisted frames (D9-legs); the plant's own cost on exact truth, p95 10-14 mm on a third of the frames; the D3 gate's translation-aligned gauge (instrument debt); D8c's head-gate log predates its own rebuild. Next: D7c.

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
