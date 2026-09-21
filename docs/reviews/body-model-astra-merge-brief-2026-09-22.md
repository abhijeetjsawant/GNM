# Merge review for Astra GPT6 — D4, the body model in the delivery path — 2026-09-22. ONE ROUND (the 2026-09-15 rule).

Your card review (`docs/reviews/body-model-astra-review-2026-09-21.md`) is adopted in the card. The agent finished on
`ladder/D4` at e209eea (worktree `/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4`). Read its
review `docs/reviews/body-model-2026-09-21.md`, `artifacts/compare/d4-body/gate.json` (every verdict derived from report
numbers; the input-mutation table), the logs under `artifacts/compare/d4-body/logs/`, and the src diff of
`scripts/build_commercial_multiview_comparison.py` and any new module under `src/`. Read-only, adversarial, cite the line.
Under the rule: list ONLY what blocks the merge (a mutation that reaches a card-banded verdict, or a card clause not
established), then everything else as instrument debt.

## What the agent reports
- hygiene 8/8 (rerun at HEAD); the pre-card reproduced through the build script to 0.0 on all 600 cells (0.7894 / 0.7295);
  repaired performer-1 figure 0.7503 after the momentum finding below.
- O1 exactness FAILED: 0.80–1.03 mm over six seeds against ≤ 1 mm; the fixture's donor violated 23 configured pose limits
  and was clamped as a fixture parameter (1.248 → 1.030, fitter byte-identical, `--no-clamp` reproduces); the tracker's own
  floor with the truth identity handed in is 0.51–0.76 mm; the residual is `scale_spine_length` shrunk 14–20 % by momentum's
  default soft limit (untouched — selecting it on the oracle would be a knob) and `scale_foot_length` unidentifiable without
  toes. Must-fail (mean body) rejects by 9–39×; closure 2.9e-6 m; mutations reject. COORDINATOR DECISION, recorded in the
  status log before the delivery was built: the 1 mm band was set without measuring the floor (the lane's recurring
  pre-registration error); the band is not moved; O1 is a standing FAIL, attributed, and the merge is decided on B1 under the
  D3 precedent (D3 merged with its 0.5 mm arm band failed on one seed and kept). gate.json prints
  `MERGE with O1 a recorded exception`.
- B1 THE BAND: fitted MHR (lod2, the smoothed repaired landmarks) minus the D7c delivery, whole-body IoU, four cameras
  pooled, paired block bootstrap: **+0.156 [0.133, 0.163] / +0.115 [0.084, 0.136]**; pooled medians 0.647 / 0.652 → 0.803 / 0.767;
  the frozen-pose control below the candidate 8/8; MAMMA's mesh bit-identical to its committed value 8/8; parts, terciles,
  P/R and the MHR mean body reported.
- B2 on the CONSUMED input: the array handed to the MHR adapter byte-identical to the rig converter's input on the same
  build; the markers re-derive to 0.0 cm; the omitted feeds (head solve, toes, Spine1; 17 of 19 landmarks) recorded.
- B3 placement (report): all-landmark median 19.9 / 18.4 mm, segment mean abs error 11.9 / 9.0 mm. B4 MAMMA (report):
  34.5 / 43.1 mm, bias 50.7 / 34.5, spread 30.5 / 36.0. B5 bytes: PASS both; facing dot > 0 on 150/150 and 145/150 frames.
- Free-offset must-fail: offsets 131 / 106 mm (fails as it must); the prediction's "identity at zero" half is FALSE
  (13 channels nonzero) and recorded as a failed prediction.
- FINDING: momentum's `calibrate_markers` mutates a later `Character.load_fbx` in the same process (skeleton state sum
  −53292 vs −57778, mesh up to 0.76 m out, 79 vs 98 animation channels, the file still imports cleanly); the pre-card fitted
  both performers in one process and its performer-1 figure was 0.021 IoU low. Guarded now (one process per performer, a
  skeleton-state guard before every GLB write). `Character.with_locators([])` does not clear locators.
- Open: the O1 re-pin against the measured floor; `--body mhr` breaks `post_merge.sh`'s eight-file check (no `mapping.npz`)
  and every rig-schema instrument (`delivered_vs_capture` KeyError) — the close-out protocol cannot run on this delivery
  as written; 4 pre-existing test failures at base.

## Questions
1. MERGE at e209eea, or not? Only what blocks.
2. Is "O1 a recorded exception decided on B1" defensible under the lane's rules (never move a band; never merge on an
   override; D3's precedent), or is it an override? If an override, what is the smallest legitimate path — e.g. the band
   re-pinned against the measured floor as a NEW pre-registration in a follow-up step before the flip to `mhr` by default?
3. One attack on B1: a mutation of a consumed input that reaches its verdict.
4. The close-out: `--body` must stay `rig` by default until the compositor and the instruments consume the MHR schema —
   is that the right disposition, and what must the merge record so the flip is its own gated step?
5. Any number you cannot reproduce; anything that misreads your card round.
