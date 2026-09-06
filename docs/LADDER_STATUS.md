# Body lane — where we are (generated, do not hand-edit)

*Rendered from `docs/ladder-status.json` on 2026-09-06 by `tools/compare/status.py render`.*
**Read this first in any body-lane session.** Then: `docs/LADDER_EXECUTION_PLAN.md` (what gets
built, in what order, gated by what), `docs/SUBSTITUTION_LADDER.md` (what is measured and how).

## Where we are

An in-house, commercially clean body capture that reaches MAMMA, measured one part at a time so we always know which part moved a number.
Done: I0, I1, I2, I3, I4, I5, I6, I7, I8, D1, D2, D3, D7, D7b, D8, D9, D8b. In flight: nothing. Blocked: nothing.

## In flight

- nothing in flight

## Next up (unblocked, not started)

- **D4** A real pose solver (momentum on the MHR body) — an Opus agent
- **D8c** The hip line: captured hips that break the performer's own width — an Opus agent

## Blocked

- none

## Decisions waiting on the user

- Lane H: decide the rig and book the marker session; performer releases covering ML training use.
- D3's recorded miss: the exact-skeleton oracle's arm band (0.5 mm) fails by 0.19 mm on one of six synthetic bodies (D2's clavicle residual on a longer lever). Recommendation (2026-09-04): keep it as a standing fail; D5 re-derives the band from the fitted lever. And tests/test_body_export.py:145 (your uncommitted file) asserts the old exporter's root; expect the track root without the asset's 0.8 offset.

## Recent log

- 2026-09-06 [D9] 2026-09-06, correction and finding from the oracle repair: the smoothing hypothesis for the oracle's 8-11 mm torso is DEAD (_thorax_frames is called only from the head solve inside reconstruct_multiview, never inside positions_to_body_track). Measured on seed 20260903 with the spine fed: the track's Hips->Spine bone sits 6.86 deg from the truth's, Hips 10.0 mm and Spine 26.6 mm off after hip-midpoint alignment, the chain straight on both (0.000 deg), the Chest->Neck direction 3.09 deg off, and the Neck 10.7 mm off -- the D7b floor mechanism: the trunk is aimed from a displaced Spine origin at the exact neck and lands at rest length along that ray. The displacement is D7's REST-PELVIS CONVENTION: the Kabsch fit uses SOMA-derived rest offsets, and on this rig's exact rest they tilt the pelvis by ~7 deg about the hip line -- invisible to the leg roots (they sit on that line, legs 0.07 mm), visible on everything above the pelvis. The rig carries its own Hips->Spine and Hips->UpperLeg rest offsets; fitting the pelvis to THOSE removes a shipped third-party constant (the head's lesson) and is the next converter item after D8b (candidate name D7c), banded on this oracle (legs AND trunk exact) with the real-take hoist and silhouette as the photographs. The arms' 0.8-1.2 mm on the oracle are downstream of the same 27 mm origin miss through the clavicle pivots; re-read after D7c.
- 2026-09-06 [D8b] merged 2026-09-06 (8ec143d, --no-ff) on the pre-registered clauses after the synthetic fixture was repaired (honest-frame noise calibrated to the take, the collapse injected on a moving run; src byte-identical across the repair): the falling performer's shoulder line off his own width 18 -> 0 frames, performer 0 4 -> 1, legs 0 -> 0 with two correct fires, raw triangulation byte-identical, photographs level on every cell, MAMMA closer in the window; delivery rebuilt in place, 8/8 byte-identical; every instrument rerun with full logs under artifacts/compare/post-merge-D8b
- 2026-09-06 [D8b] 2026-09-06, D8b close-out: (1) stated cost: nine arm fires outside the fault window on performer 1; against MAMMA's body five got 4-8 mm worse, three better, one wrist 30 -> 47 -- a 15 % ceiling buys the shoulder at a few millimetres of ordinary arm motion. (2) On the repaired FIXTURE demote and reject give bit-identical shoulders and D8's gap clause places them on the neck; on the REAL take the shoulders are recovered, not held (|shoulder-neck| sd 20-24 mm, 34-37 deg of direction spread over the fired frames, width 353-381 against his 364). (3) Handoffs: the hip line collapses the same way on 23 frames and is not in the segment list; captured_limb_stability recomputes the performer's median per build (B4 read 4 -> 5 on the moving denominator, 4 -> 4 on a fixed one; D8's own 27 -> 18 / 22 -> 4 headline needs a fixed-denominator recheck; --median-from owed). (4) RESOLUTION SWEEP on the same D9 code at detector widths 1280 / 2560 / 3840 (the user asked whether a bigger GPU would help): performer 1's shoulder-line frames off 18 / 20 / 15, hip line 23 / 29 / 24, and 3840 adds leg misses on performer 0 that 1280 lacks -- the collapse is not a crop-resolution symptom, so D8b treats the cause it names; a GPU changes detection time, not detections. (5) The D8 test's 'off' arm now passes segment_length_ceiling_fraction=None so it is pre-D8 code again; provenance rerun on main after the branch overwrote the shared JSON.
- 2026-09-06 [D8b] 2026-09-06, side experiments on the bear clip (four separate Seedance generations, not one scene; artifacts under artifacts/{mamma,prompthmr,mocapanything,sam3d-multiview,bullettime,videodepth,sv4d}/, page https://claude.ai/code/artifact/0cc6e2fc-c544-4514-a80d-68dd99855b4e): every single-camera system fits a human (or a quadruped template) into the bear; SAM 3D Body into MHR did best per frame and now runs on Modal (L40S, 0.5-0.9 s/frame, GPU/CPU parity 1.5 mm) -- D4's front end is ready; four-camera triangulation of its joints fixes place and scale (held-out camera 2.3-3.1x better, feet on the calibrated floor) and loses bone rigidity, which is the solver's job. Synthetic views from one camera: BulletTime delivers 91-97 % of a 15 deg turn and 50 % of 51, abandoning the scene beyond; SV4D 2.0 delivers exact azimuths with no drift but MAMMA cannot mask its side views; metric video depth (MoGe-2, VDA) fails a pre-registered floor gate by 0.5-1.5x in scale while tracking (TAPIP3D) is near-perfect. STANDING RULES from this: (1) cross-view consistency is not evidence when the views descend from one observation -- the correspondence-scramble control exposes it (synthetic pairs discriminate 1.1x, real pairs 14-16x); score any synthetic set against a real held-out camera only; (2) MAMMA's shape parameters swing 0.03-47 on real footage with the camera set alone and are not a verdict on anything; (3) for a single calibrated camera the bottleneck is metric depth, not tracking, segmentation or focal.
- 2026-09-06 [—] 2026-09-06: the bear side experiments are PARKED (page https://claude.ai/code/artifact/0cc6e2fc-c544-4514-a80d-68dd99855b4e, eleven tabs; workers bullettime/sv4d/videodepth/prompthmr/mocapanything/sam3d_body; rules in CLAUDE.md). Back to the humans on the fixture. Queue, in order: D8c hip line; foot-contact hoist re-aim; D7c pelvis on the rig's own rest; D9-legs; instrument debts (re-pin D3 references, --median-from, D8 headline recheck); D6; D5; D4. Every delivery step since D7 is merged and pushed; The Solve So Far carries v2-v6.
- 2026-09-06 [D8c] D8c card written and reviewed by Grok 4.6 (Cursor) in Sol's place (docs/reviews/hip-line-grok-review-2026-09-06.md). Measured before the card: performer 1's hip line fails in three runs and two classes -- 110-119 and 84-86 an inward collapse along the hip line in three agreeing cameras (D8b's class; frame 113 an outward raw spike; 84-86 one hip at a time), 158-168 a two-view stretch along the A-C baseline while he lies on the floor with his hips pointed at the only two cameras that see him (six of eleven frames over the ceiling, the other five at +9-13 % stay). Decided on data after the review: a per-hip root->hip rule is refused (the pelvis landmark spreads +25 % on honest frames and would fire 35/17 times), both hips charged, 84-86 registered as a known over-charge. The root's dependence on the hip midpoint is pre-registered as reports R1-R4 (hoist-removed, Savitzky-Golay +-4 envelope, R_hips delta), the synthetic injects the MEASURED mode (along the hip line toward its midpoint, knees untouched) and scores through the converter, and B4 on D8b's fixed medians is in the merge rule paired with the photographs as the guard against a recovery that lets the agreeing rays win.

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
