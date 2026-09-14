# Astra GPT6 merge review of D7c — 2026-09-14, round 1 at ladder/D7c 9dda9ac. Verdict: NO MERGE

Brief: `docs/reviews/pelvis-rest-astra-merge-brief-2026-09-14.md` (state, the whole src diff, the agent's figures, five
questions). Astra reproduced the hygiene and tripwire hashes, the six-body C parity, O1/O2, the guard mask, take P1/P2,
the B1 intervals and the guarded restatement; 12/12 bit-parity of the src estimator against the frozen one. Every finding
below was verified against the worktree before adoption (`gate.json:297` lists only P1 for the oracle;
`d7c_gate_report.py:366` selects `S REREAD` clauses only, so G2 is outside the merge conjunct; `d7c_delivered_bytes.py:104/107/124/141`
read track rotations, include norm error in the increment, average composed positions, and substitute positional closure
for the rest comparison; `d7c_pelvis_synthetic.py:647` compares trunk tilt, not pelvis tilt, to the world-vertical control;
`tests/test_pelvis_rest.py:275` injects 6.87° rather than pinning 7.567708°).

## What each finding changes (the agent's next stage)

| # | finding | change |
|---|---|---|
| 1a | oracle P2 (the GLB anchor lock on the six bodies) never measured | export all six bodies, check Foot and Toes at their run anchors against the frozen projection masks, add the conjunct |
| 1b | the gate's S conjunct does not include G2 (a FAIL injected in memory still returned MERGE) | the gate enforces the reread's every stop, G2 included; re-run |
| 3 | B6's mesh-deformation reading, rotational closure, rest / hierarchy / inverse-bind validation, gap-boundary playback all absent | performed as REPORTS before close-out; no band invented |
| 4 | the three B1 rises need attribution | ablation: the candidate rendered with D9b's per-frame root, candidate locals and rest kept, paired part-wise IoU on the same populations and draws — diagnostic only |
| 5 | six overclaims (Head "identical" read from track JSON, between-key error under-read 0.0003 → 0.459 / 0.295 mm, the world-vertical limitation DOES apply at 17.73° vs 16.82° truth, G2 "order of magnitude" is 3.16×, follower ratio not monotone in σ, the SOMA-posed pin not preserved) and three smaller ones (0.029° increment includes norm error, finger identity is a track claim, 19/150 is performer 0 only) | each corrected in the review, the instruments repaired where the number was wrong, the re-pin preserves the 7.567708° fixture |

---

**NO MERGE at `9dda9ac`.** The estimator is supported; the gate does not yet establish the card’s complete conjunction.

1. **The blockers, in order — both verified, confidence 10/10:**

   - **Oracle P2 is missing.** The gate quotes “P1 and P2 on the take and every seed,” but its conjuncts include only **P1** for the oracle. The oracle artifact contains channel-preservation checks, with no GLB anchor-lock measurement. Export and check all six bodies against their frozen projection masks, then include that result. [Deciding gate line](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/artifacts/compare/d7c-pelvis-rest/gate.json:297), [oracle artifact](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/artifacts/compare/d7c-pelvis-rest/projection-preservation-oracle.json:2).
   - **The gate does not enforce S’s guard stop.** Its S conjunct selects only clauses beginning `S REREAD`; G2 is a separate clause. I changed G2’s verdict to FAIL **in memory**, and the gate still returned MERGE. The actual G2 measurements pass, but the merge instrument must enforce them. [Deciding implementation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:366).

2. **The src implementation is the selected estimator.**

   E constructs exactly `{rest[Spine]−mid, rest[L]−mid, rest[R]−mid}`, uses observations about the captured hip midpoint, and applies the un-centred SVD with determinant correction. I independently reproduced **12/12 bit-identical quaternion arrays and guard masks** against the frozen estimator on S’s actual draws. [SVD](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/src/autoanim_gnm/commercial_multiview.py:2400).

   The guard runs only under D/E. Resolution is assessed once for the subject; sufficient observations take the shared interpolation path, otherwise the whole subject falls back. No per-frame definition switch. [Scope and fallback](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/src/autoanim_gnm/commercial_multiview.py:2324).

   **No hidden fitted geometry or correction constant enters E.** The ceiling and fallback threshold are the disclosed inherited settings. Strictly, the function still creates a `root` slice at line 2352, but E never consumes its values. The guard’s finite-hip assumption is satisfied by the converter’s input validation. [Validation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/src/autoanim_gnm/commercial_multiview.py:2941).

3. **B6’s omission does not create an additional numerical merge veto under the frozen predicate. It does leave the card unfinished.**

   The commitment was a **first deformation reading whose figures go to D6**, not permission to hand D6 an unperformed measurement. Require the report before close-out, or explicitly amend that obligation; invent no deformation acceptance band. [Card](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/docs/reviews/pelvis-rest-card-draft-2026-09-14.md:9).

   Moreover, mesh deformation is **not the only missing report**: rotational closure, rest/hierarchy/inverse-bind validation, and proper gap-boundary playback checks are also absent. The instrument explicitly substitutes positional closure for the rest comparison. [Deciding implementation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:141).

4. **The unexplained B1 direction does not fail B1 or change the selector decision.**

   I recomputed all eight intervals from the retained per-frame arrays and frozen bootstrap draws. The three positive intervals reproduce exactly. The assertion that these masks cannot see this change was too strong: changing pelvis estimation changes delivered geometry. A favourable outline does not establish anatomical correctness or sound deformation. [Recorded positive intervals](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/artifacts/compare/d7c-pelvis-rest/silhouette-partwise.json:486).

   **One attribution measurement before close-out:** rerender the candidate with **D9b’s per-frame root translation**, retaining candidate locals and rest, and recompute paired partwise IoU per camera on the same populations and draws. Compare this ablation with the delivered candidate to separate the translation contribution from the articulated/skinned geometry contribution. Keep it diagnostic.

5. **The material overclaims and unreproduced interpretations are:**

   | Claim | What the evidence actually establishes |
   |---|---|
   | Head WORLD rotation “IDENTICAL,” exporter preservation proved | The instrument reads **body-track JSON rotations**, then compares rounded median angles from identity. Neither GLB rotation equality nor agreement with the retained head solve follows. Independent GLB reconstruction finds between-build differences only around **0.000013°**, supporting preservation to numerical precision. [Deciding code](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:104). |
   | Between-key contact error approximately **0.0003 mm maximum** | The code averages already-composed world positions. Actual quaternion interpolation followed by FK gives candidate Foot/Toes midpoint anchor errors up to **0.459 / 0.295 mm**. The same reconstruction reproduces D9b’s **0.665 / 1.311 mm**. This corrects B6; it does not fail keyed-only P2. [Deciding code](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:124). |
   | World-vertical limitation does not apply | **53.733° is trunk tilt**, compared against pelvis error. Actual truth pelvis bent median is **16.8228°**; world-vertical error **17.7256°** is within 2°. The stated limitation **does apply**, without changing S’s stop conditions. [Wrong comparator](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_pelvis_synthetic.py:647). |
   | G2 wins both metrics everywhere “by an order of magnitude” | Strict wins reproduce; that magnitude does not. Aggregate STEP improvement is **8.17935 / 2.58984 = 3.158×**. [Artifact](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/artifacts/compare/d7c-pelvis-rest/selector-reread-sigma0.335546875.json:1810). |
   | Follower ratio varies monotonically with noise | Seed 20260907 reads **2.742 at σ1**, versus **2.645 at σ0.5**. A general tendency is supported, monotonicity is not. [σ1](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/artifacts/compare/d7c-pelvis-rest/selector.json:1276), [σ0.5](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/artifacts/compare/d7c-pelvis-rest/selector-sensitivity-sigma0.5.json:1276). |
   | The new test preserves the original SOMA-posed reading | It instead injects a literal **6.87°** into another fixture and accepts **5–9°**. It does not pin the original **7.567708°** result. Preserve the actual fixture when the coordinator repins. [Deciding test](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tests/test_pelvis_rest.py:275). |

   Other corrections: the reported **0.029°** median increment includes quaternion norm error; normalized GLB rotations give a zero median. Finger bit-identity is established for track arrays, not literally every GLB proximal channel. The head-gate **19/150** count applies only to performer 0; performer 1 is **23/150**. [Increment calculation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:107), [head log](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/artifacts/compare/d7c-pelvis-rest/logs/19-b5-head-gate.log:21).

   I reproduced the hygiene/tripwire hashes, six-body C array parity, O1 origin/residual figures, O2, the guard mask, authenticated take P1/P2, B1 intervals, and guarded take restatement. Calibration and S arithmetic reproduce under the recorded amendments.

   Tests reproduce **4 failed, 47 passed**. The subsequently completed broader run stopped at an unchanged compositor pin: **1 failed, 198 passed, 4 skipped**. It is not full-suite validation; the agreed coordinator repins still belong in the landing commit. [Full-run result](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/artifacts/compare/d7c-pelvis-rest/logs/23-tests-full.log:38).

Read-only throughout; no files changed.