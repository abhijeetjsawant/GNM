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

---

# Merge review round 2 — 2026-09-14, at ladder/D7c 0b3eba4. Verdict: NO MERGE

Verified against the worktree: `d7c_gate_report.py:95` and `:158` assign literal "PASS" to the wrong-origin control and the (a)/(b) split (Astra's in-memory counter-examples still read MERGE); `:481` the flip test flips assigned verdicts, not inputs; `body_export.py:599` writes the FIRST ANIMATED POSE as the node default (so the 590 / 156 mm skin-vs-POSITION mismatch is not a reader defect); `d7c_delivered_bytes.py:286` the inverted-triangle test uses a fixed bind-space normal; `:130/:410` playback interpolates rotation but reads translation at the key; `d7c_gate_report.py:351` still prints the stale Head "IDENTICAL" claim; `b1-attribution.json:171` performer 1's root share +0.00552 has CI [−0.00194, +0.00750].

| # | finding | change |
|---|---|---|
| 1 | two clause verdicts are literals; the flip test proves wiring, not measurement | verdicts derived from the inputs; failure demonstrated at the input level for every conjunct |
| 2 | B6's mesh reading must be finished (the reader is sound to 0.005 mm vs Blender; node defaults are the first pose); the inverted-triangle test is wrong (a rigid 180° rotation trips it); the area / edge ratio changes (28.15 → 29.95, 0.0349 → 0.0173) need interpretation, no band | reading finished on a per-triangle frame-relative basis; the ratios interpreted |
| 3 | the diagnostics exclusion is acceptable | kept |
| 4 | oracle P2 reproduces (blocker resolved); playback must interpolate translation too (0.459 / 0.295 mm candidate, 0.665 / 1.311 D9b); rotational closure measurable after undoing the exporter's bind/alignment change (median 0.000003°, max 0.000013°); B1's root share for performer 1 is a point estimate with a CI through zero; the gate's stale Head claim | all adopted |

---

**NO MERGE at `0b3eba4`.** The gate still has unenforced conditions, and B6’s required report remains incorrectly measured. Confidence: **10/10**.

1. **The gate remains a blocker.**

   I changed the input reports **in memory**, independently:

   - Set the wrong-origin control’s residual to zero. Verdict: **MERGE**. Its clause receives a literal `"PASS"` at [line 95](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:95).
   - Changed `(b) vs (a)` to one better/five worse and marked `S_verdict: SPLIT`. Verdict: **MERGE**. That clause also receives a literal `"PASS"` at [line 158](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:158).

   The flip experiment changes **already-assigned verdicts**, so it proves the final conjunction’s wiring, not that measurements can fail each condition. [Deciding line](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:481). Derive these verdicts from their inputs and demonstrate failure at that level.

2. **B6’s handoff is not sufficient under the unchanged card—and its stated reason is wrong.**

   The deciding exporter line is:

   ```python
   "rotation": animated_rotations[0, index].astype(float).tolist(),
   ```

   These node defaults contain the **first animated pose**, not the bind pose. Comparing skinning under those defaults with `POSITION` therefore does not establish a reader/exporter defect. [Exporter](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/src/autoanim_gnm/body_export.py:599).

   I compared this reader’s animated vertices with the retained Blender meshes on five frames per performer/build: maximum discrepancy **0.00518 mm**. The evidence supports the skinning reader on those samples. Also, the reported default-pose mismatches are not identical: **594.005→590.159 mm** and **158.929→156.052 mm**.

   Finish the reporting commitment before closure, or explicitly amend it—as round 1 required. This introduces **no deformation acceptance band**. [Round-1 disposition](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/reviews/pelvis-rest-astra-merge-review-2026-09-14.md:41).

   Two report defects still need repair:

   - “Inverted triangles” uses the dot product against a fixed bind-space normal. A harmless rigid 180° rotation triggers it; I reproduced that false positive. [Deciding calculation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:286).
   - “Within 1%…D7c did not change it” is unsupported even by its own figures: performer 0’s area-ratio maximum changes **28.1542→29.9488**, and minimum edge ratio **0.03494→0.01731**. These require interpretation, not a new band. [Baseline](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/artifacts/compare/d7c-pelvis-rest/b6-delivered-bytes.json:167), [candidate](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/artifacts/compare/d7c-pelvis-rest/b6-delivered-bytes.json:535).

3. **Twelve conjuncts do not yet enforce the complete rule; the diagnostics exclusion is acceptable.**

   The two counterexamples above decide the enforcement question. Excluding the run-report diagnostics from the numerical predicate is consistent with its REPORT status; it hides no observed acceptance failure here. [Explicit exclusion](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:405). Reporting obligations still need completion.

4. **Reproduction and round-1 adoption:**

   - **Oracle P2 reproduces:** six bodies, **18/12/18/10/12/20 runs**, worst **4.855407658×10⁻⁷ m**. That blocker is resolved.
   - **B6 playback remains wrong.** It interpolates rotations but reads translation at `frame`, freezing the root at the first key. [Translation read](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:130), [midpoint call](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:410). Interpolating translation too reproduces round 1: candidate maxima **0.459063 / 0.295016 mm**, baseline **0.664888 / 1.311246 mm**. The revised millimetre-scale medians are not viewer playback.
   - **Rotational closure is measurable.** Undoing the exporter’s documented bind/alignment frame change gives candidate median **0.000003°**, maxima **0.000011 / 0.000013°** across all 8,250 joint/frame samples per performer. Replace the raw 32° comparison with this frame-correct closure. [Exporter’s transformation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/src/autoanim_gnm/body_export.py:379).
   - **B1’s ablation and all eight cells reproduce**, including intervals. However, performer 1’s root contribution **+0.00552** has CI **[−0.00194, +0.00750]**. Describe its attribution as the point-estimate decomposition; a definite positive root effect is not established. [Interval](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/artifacts/compare/d7c-pelvis-rest/b1-attribution.json:171).
   - Head differences, normalized increments, and the **16.8228°** truth-pelvis comparator reproduce. The gate nevertheless still generates the stale Head claim **“IDENTICAL between the builds.”** [Deciding line](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:351).
   - Focused validation: **32 passed**, including pelvis-rest and provenance tests. I did not rerun the full suite.

Read-only throughout; worktree unchanged.

---

# Merge review round 3 — 2026-09-14, at ladder/D7c 65a5a4d. Verdict: NO MERGE (two blockers)

Verified against the worktree: `d7c_gate_report.py:220` reads saved (b)-vs-(a) classification strings, `:213` checks that an accepted σ exists without its residual, `:96` and `:293` accept a nonempty subset / a vacuous `all({})`, `:248` trusts the stored follower ratio; `d7c_delivered_bytes.py:273` takes the FIRST VERTEX's dominant joint (a cyclic reorder changes the counts 317→318, 4373→4362); a constant-weight LBS with positive determinant 0.04 is called inverted; `:340` the 325/317 figures are maxima over 15 sampled frames (performer 1 ranges 35–338); `:531` fits the closure constant from frame 0 so a constant 10° leaf error is invisible; `pelvis-rest-2026-09-14.md:764` "only one of four cells" vs two positive articulation intervals.

| # | finding | change |
|---|---|---|
| 1 | five further input mutations still read MERGE (numerical SPLIT, calibration residual, hygiene 1/8, follower winner, empty denominator maps) | every S / calibration / hygiene / denominator verdict recomputed from the measurements with required population coverage (8/8 files, six cells, six bodies, both performers) |
| 2 | the carried-normal inversion test is not a sound classifier | replaced by the sign of the per-triangle deformation determinant (deformed edges and carried rest normal against the rest triangle), or the counts qualified as a heuristic |
| 3 | the B6 maxima are over 15 frames, and "deep hip crease / a handful / not a tear" are unsupported | per-frame ranges reported; the interpretation qualified or localised |
| 4 | the rotational closure fits its constant from the output | the exporter's actual bind/alignment transform reconstructed (`body_export.py:379`), median 0.000003°, max 0.000011 / 0.000013°; the "one of four cells" line corrected |

---

**NO MERGE at `65a5a4d`.** Two blockers remain, both verified at confidence **10/10**: incomplete gate enforcement and incorrectly characterized B6 measurements. No deformation acceptance band is needed.

1. **The twelve conjuncts are input-dependent, but they do not enforce every required condition.**

   I reproduced the saved gate and **16/16 committed mutations → NO MERGE**. These additional, independent mutations still returned **MERGE**:

   | Input mutation | Missed condition |
   |---|---|
   | Set `(b)` whole-take orientation error to **4°**, against `(a)`’s **5.28078°**, leaving the other five comparisons worse | Numerical SPLIT |
   | Set the accepted calibration sample’s residual to **1 mm** and its `passes` field false | Outside **0.05 mm** tolerance |
   | Remove seven hygiene comparisons, leaving one matching hash | **1/8** is insufficient coverage |
   | Set one follower row’s `winner_i_deg` to **100**, retaining its saved ratio | Follower no longer achieves 2× |
   | Empty both landmark-array comparison maps | Missing denominator evidence |

   The deciding lines are:
   
   - `bva = list(re_["b_vs_a"].values())`: S reads saved classification strings instead of deriving them from the six numerical comparisons. [Line 220](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:220)
   - `... and c is not None`: calibration checks that an accepted sigma exists, without checking its measured residual. [Line 213](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:213)
   - Hygiene requires only a nonempty matching subset; denominator checks allow vacuous `all({}.values())`. [Line 96](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:96), [line 293](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:293)
   - The follower condition trusts the stored ratio. [Line 248](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:248)

   Removing literal PASS assignments fixed the earlier examples. Measurement derivation and required-population coverage still need enforcement.

2. **The carried-normal test is rigid-frame invariant, but it is not a sound general inversion classifier.**

   Its supposed triangle-dominant joint is actually the **first vertex’s** dominant joint:

   ```python
   slot = dominant[triangles[:, 0]]
   ```

   [Deciding line 273](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:273).

   Cyclically reordering vertices preserves geometry and winding, yet changes candidate performer 0’s maximum **317→318** and total **4373→4362**. Performer 1’s maximum changes **338→337**.

   More fundamentally, I constructed constant-weight LBS with weights **0.4/0.3/0.3** and joint rotations **identity/180°/180°**. Its deformation is `diag(1, −0.2, −0.2)`, with **positive determinant 0.04**, yet this test calls the triangle inverted. Normal opposition to one selected joint is a heuristic, not proof of inversion.

   The Kabsch explanation also overreaches: rank-two covariance does not make a **proper rotation** a coin toss. A planar, noncollinear triangle rotated 180° reproduced with determinant **+1** and zero RSSD using [SciPy’s Kabsch implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html). Such a fitted rotation still cannot independently establish inversion.

3. **The area/edge numbers reproduce; the stronger B6 interpretations do not follow.**

   I regenerated **every B6 JSON value exactly**, in memory. The reported stretching and pinching changes support “tails move both ways.”

   However, **325/317 and 344/338 are maxima across 15 sampled frames**, not counts applying to every frame. Candidate performer 1 ranges **35–338**, or **1.44–13.94%**. [Maximum calculation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:340).

   “Deep hip crease,” “a handful,” and “not a systematic tear” are not established by these aggregate statistics or the defective inversion classifier. Those claims need supporting localization/inspection or qualification. [Deciding interpretation](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/docs/reviews/pelvis-rest-2026-09-14.md:841).

4. **The rotational-closure adoption weakens round two’s requested measurement.**

   The implementation fits the correction from the tested output:

   ```python
   constant = ours[0].inv() * theirs[0]
   ```

   [Deciding line 531](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:531).

   That measures constancy relative to frame zero. It does not validate the exporter’s known bind/alignment transformation. Adding a constant **10° leaf-joint error** leaves this fitted residual around **0.000011°**.

   Reconstructing the actual exporter transformation reproduces round two: candidate median **0.000003°**, maxima **0.000011/0.000013°**. The current fitted calculation instead reports candidate medians **0.000006/0.000004°**, maxima **0.000014/0.000011°**.

   Playback’s rounded **0.459/0.295 versus 0.665/1.311 mm** reproduces. B1’s revised attribution agrees with its recorded intervals; “only one of four cells” at [line 764](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/docs/reviews/pelvis-rest-2026-09-14.md:764) is inconsistent with the **two** positive articulation intervals.

Read-only throughout. Gate and B6 regenerated in memory; B1 was checked against its artifact, without rerendering.

---

# Merge review round 4 — 2026-09-14, at ladder/D7c 1673e6b. Verdict: NO MERGE (two blockers)

Verified: `d7c_gate_report.py:350/358` accept empty G1/G2 body maps and a missing seed; `:433` and `:421` read saved maxima instead of the per-seed / per-run measurements; `:487` checks `len(cells) == 8`, not the eight named cells; `:570` prints a null for a removed field; `d7c_delivered_bytes.py:334` transforms the rest centroid with the averaged matrix, which is not the centroid of the transformed vertices under varying weights (Astra's rigid two-bone counter-example flips with a vertex swap; the candidate's ranges move 274–326 → 265–313 and 45–344 → 72–340); all four pinning tests use one constant affine skin; `pelvis-rest-2026-09-14.md:838` keeps the "coin toss" Kabsch claim, `:764` the false bold sentence; the closure medians are 3e-6 / 4e-6°, not 3e-6 for both.

| # | finding | change |
|---|---|---|
| 1 | five more mutations pass (empty G1 / G2 maps, a missing G2 seed, a failing oracle P2 seed hidden by the global summary, a failing take run hidden by the subject maximum, a renamed B1 cell) | named populations and cells required everywhere; every maximum derived from, or cross-checked against, its constituent measurements |
| 2 | the carried-tetrahedron test is not a sound LBS inversion classifier (varying weights) | after three unsound classifiers the measurement is QUALIFIED as a proxy with its failure modes stated and no inversion claim (Astra's round-3 allowance); the sound skinning-Jacobian measurement (Kavan eq. 17) is D6's instrument |
| 4 | the Kabsch "coin toss" text, the false B1 sentence, the closure medians, the null B6 row | all corrected |

---

**NO MERGE at `1673e6b`. Two blockers remain, both confidence 10/10: gate enforcement and the inversion classifier.**

1. **The gate still accepts missing evidence and contradictory measurements.**

   I reproduced the saved MERGE verdict and **31/31 committed mutations detected**. These additional mutations, applied independently in memory, still return **MERGE**:

   | New input mutation | Deciding line |
   |---|---|
   | Empty G1’s six-body map | `all(... for row in g1.values())` accepts zero bodies. [Line 350](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:350) |
   | Empty G2’s six-body map—or remove only seed `20260903` | Per-body checks have no coverage requirement; retained aggregate medians pass. [Line 358](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:358) |
   | Set oracle seed `20260903`’s `worst_travel_m` to **0.001 m**, with FAIL and a failing run; retain the global summary | `worst_o = oracle_p2["worst_travel_m_over_all_seeds"]` ignores the failing seed’s measurement. [Line 433](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:433) |
   | Set the take’s first LeftFoot run to **0.001 m**, `holds=False`; retain the subject summary | Reads the saved subject maximum instead of the run measurements. [Line 421](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:421) |
   | Rename performer 0’s required arm/whole-take B1 cell to `clause_duplicate` | Coverage checks `len(cells) == 8`, not the eight required identities. [Line 487](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:487) |

   Require the named populations/cells and derive or cross-check maxima against their constituent measurements. The five round 3 examples are repaired; the general enforcement requirement remains unfinished.

2. **The carried tetrahedron is sound for a constant affine transformation, but this implementation is not a sound general LBS inversion classifier.**

   The deciding line is:

   ```python
   posed_d = apply(triangle_matrices, fourth)
   ```

   [Line 334](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:334).

   With varying weights, transforming the rest centroid with the average matrix does **not** produce the centroid of the transformed vertices. That displacement contaminates the signed volume independently of the normal offset. A skinning Jacobian must account for spatially varying weights. [Kavan’s derivation, equation 17](https://skinning.org/direct-methods.pdf).

   **New counterexample:** triangle `(0,0,0), (1,0,0), (0,1,0)`; two bones, identity and `Rx(60°)`; weights `(1,0), (1,0), (0,1)`. Every delivered vertex equals the same proper rigid rotation of its rest vertex. With barycentrically interpolated weights, the deformation’s Jacobian determinant at the centroid is **0.722222**, positive. Nevertheless:

   - Original order: **inverted**.
   - Swap the first two vertices: **uninverted**.

   This also affects the actual candidate:

   | Candidate | Recorded sampled range | After swapping two vertices |
   |---|---:|---:|
   | Performer 0 | 274–326 | **265–313** |
   | Performer 1 | 45–344 | **72–340** |

   All four committed tests use **one constant affine skin**, which removes the problematic centroid displacement. [Test fixture](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tests/test_pelvis_rest.py:344).

   **The ranges reproduce exactly as outputs of this classifier; they do not establish inversion counts.** Consequently, the current B6 text still overclaims despite withdrawing the mechanism and tearing language. Repair the measurement or explicitly qualify it as a proxy, as round 3 allowed. No deformation acceptance band is needed.

On question 4:

- **The Kabsch correction was missed.** The document still says planar points make the determinant’s sign “a coin toss.” I again reproduced a proper planar Kabsch rotation with determinant **+1**, RSSD **0**. [Retained claim](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/docs/reviews/pelvis-rest-2026-09-14.md:838).
- **The B1 correction’s bold sentence remains false:** “only PERFORMER 1’s two shares straddle zero.” Performer 0’s root shares also straddle zero, as the following sentence correctly states. [Line 764](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/docs/reviews/pelvis-rest-2026-09-14.md:764).
- **Rotational closure is now reconstructed appropriately.** Candidate medians reproduce as **3e-6° / 4e-6°**, maxima **1.1e-5° / 1.4e-5°**—not 3e-6° for both performers. [Performer 1 artifact](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/artifacts/compare/d7c-pelvis-rest/b6-delivered-bytes.json:806).
- The generated gate’s B6 row still requests the removed `inverted_triangles_frame_relative` field and prints **null**. [Line 570](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:570).

I regenerated **every B6 JSON value exactly**, including ranges, ever/always counts, joint percentages, area/edge statistics and closure. Focused tests: **36 passed**. I did not rerun the full suite or independently recheck the two pre-existing failures.

Read-only throughout; worktree unchanged; `git diff 9dda9ac -- src/` remains empty.

---

# Merge review round 5 — 2026-09-14, at ladder/D7c 843bce5. Verdict: NO MERGE (two blockers)

Verified: `d7c_gate_report.py:304/313` read S's stored aggregate and never derive it from `bodies`; `:518` the oracle P2 global summary is not cross-checked; `:509` run identity is not required; `:93` a missing measurement field contributes nothing; the follower's stored ratio still escapes; `pelvis-rest-2026-09-14.md:1055` still claims "inverts 13–14 % of its triangles per frame on EVERY build" and a "deep hip crease" (the proxy spans 1.86–14.19 % on performer 1), `:840` the "coin toss" text survives, `d7c_delivered_bytes.py:359` the docstring still says "normal FLIPS … an inverted triangle"; `:303` misstates the centroid argument (the averaged matrix IS the centroid's skinning matrix under barycentric weights; transforming the centroid differs from averaging the transformed vertices).

| # | finding | change |
|---|---|---|
| 1 | six more mutations pass; the general rule is not applied | the gate is rebuilt on one rule enforced mechanically: every value it reads is either derived from named constituents or cross-checked against them, a missing field is a FAIL, every set is checked by identity; proven by a leaf-level fuzzer over every input report (perturb-to-fail and delete, one leaf at a time), with the ignored leaves listed and justified |
| 2 | B6 still draws inversion conclusions the proxy cannot support | every inversion conclusion removed; the docstring and the centroid explanation corrected; the "coin toss" text removed at its second site |

---

**NO MERGE at `843bce5`. Two blockers remain, both confidence 10/10: gate enforcement and surviving B6 inversion claims.**

1. **The gate still accepts missing evidence and contradictory summaries.**

   The saved MERGE verdict and **39/39 committed mutations detected** reproduce. These **new, independent in-memory mutations** nevertheless return **MERGE**:

   | Mutation | Deciding implementation |
   |---|---|
   | Change S’s stored `(a)` whole-take orientation median from **5.28078° to 1°**, leaving all six constituents valid and unchanged | `agg = re_["aggregated_median_of_six"]`; the gate never derives this aggregate from `bodies`. [Line 304](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:304) |
   | Set all six underlying `(a)` whole-take orientation errors to **100°**, retaining the passing aggregate; separately, empty S’s main `bodies` map | Both return MERGE through the same aggregate-only path. [Line 313](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:313) |
   | Change oracle P2’s global `worst_travel_m_over_all_seeds` to **0.001 m**, leaving every seed and run unchanged | Cross-checks cover per-seed maxima but omit the global summary. [Line 518](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:518) |
   | Replace oracle seed `20260903`’s first run with a duplicate of its second, preserving count and maximum | `have_runs` checks only nonemptiness, not required run identities. [Line 509](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:509) |
   | Delete both Foot/Toes measurement fields from the take’s first run, retaining `holds=True` | `run_maximum` scans whatever `_max_m` fields remain; missing measurements contribute nothing. [Line 93](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:93) |

   Changing only the follower’s saved ratio to **0.1** also escapes. Changing G2’s stored median alone correctly produces **NO MERGE**.

   The round-4 examples are repaired; the claimed general rule is not. Require S’s constituent population, complete run identities and measurements, and consistency of retained summaries.

2. **The proxy qualification is acceptable in scope, but the report still contradicts it.**

   The decisive surviving claim is:

   > “inverts 13–14 % of its triangles per frame on EVERY build”

   It then attributes this to “a deep hip crease.” Neither conclusion follows from this proxy. [Lines beginning 1055](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/docs/reviews/pelvis-rest-2026-09-14.md:1055). The instrument’s own docstring also retains “normal FLIPS … an inverted triangle.” [Line 359](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:359).

   Remove those conclusions consistently. The qualified proxy and D6 handoff are sufficient under round 4; this requires no new deformation band.

3. **Two explanations still misread the correction.**

   - The proxy says the averaged matrix is **not** the centroid’s skinning matrix. Under barycentrically interpolated weights, it **is**. The problem is that **transforming the centroid differs from averaging the transformed vertices**. I reproduced that distinction directly. Correct [line 303](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:303), including its artifact/document copies. [Kavan equation 17](https://skinning.org/direct-methods.pdf) supplies the weight-gradient term for the Jacobian.
   - The planar Kabsch “coin toss” assertion **still appears at [line 840](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/docs/reviews/pelvis-rest-2026-09-14.md:840)**, despite the later correction. Proper planar Kabsch again reproduces **determinant +1, RSSD 0**.

4. **Reproduction:** every B6 JSON value and both take/oracle P2 reports regenerated exactly. The proxy counterexample gives **det +0.722222**, firing before the swap and not afterward; candidate swapped ranges reproduce **265–313 / 72–340**. Closure reproduces **3e-6° / 4e-6°** medians and **1.1e-5° / 1.4e-5°** maxima. The corrected B1 sentence agrees with its intervals.

   **38 tests passed** (`test_pelvis_rest.py`, `test_provenance_audit.py`). I did not rerun the two reported pre-existing failures. The “13–14% every frame” number fails even as a proxy summary: candidate performer 1 spans **1.86–14.19%**.

Read-only throughout; worktree unchanged; `git diff 9dda9ac -- src/` is empty.

---

# Merge review round 6 — 2026-09-14, at ladder/D7c e68e06d. Verdict: NO MERGE (gate enforcement)

Verified: `d7c_gate_report.py:571` consumes the saved `authenticated` boolean, not the two hashes; `:442/:452` the follower's denominator is the duplicated `winner_i_deg`, not the selected body's own measurement; `:379` the selector's medians are read without their populations; `:668` a control passes on any nonempty `failing_channels`; `d7c_gate_fuzz.py:169` classifies REPORT-only by `elif moved:` without checking clause status or conjunct membership (164 report, 40 diagnostics, 14 historical FAIL, 8 ENFORCED controls misclassified), `:158` "enforced" = at least one mutation failed; `d7c_delivered_bytes.py:407` still says the averaged matrix is not the skinning field at the centroid; 18,172 = 13,763 leaves + 4,409 containers; the float-interning explanation does not reproduce.

| # | finding | change |
|---|---|---|
| 1 | four leaves the gate should read and does not (authentication hashes, the follower denominator, selector populations, the controls' named channels) | each derived; the fuzzer's classification made to check clause status and conjunct membership; "enforced" reported as what it is |
| 2 | 8 enforced control leaves misclassified as REPORT-only | fixed with the classification |
| 3 | the centroid text at :407; the leaf / container count; the interning explanation | corrected |

---

**NO MERGE. Gate enforcement still blocks, confidence 10/10.**

The commit order is reversed in the brief: `843bce5 → db3066c → e68e06d`. The worktree is clean at `e68e06d`; the rewrite belongs to that newer commit. I tested both gates in memory. The attacks below return **MERGE on both**; deciding-line links refer to the newer rewrite.

| Independent mutation | Result and deciding line |
|---|---|
| Set `projection/subjects/subject_00/P1_channel_preservation/authentication/glb_body_track_sha256` to 64 zeros, retaining the recomputed hash | **MERGE**, with no clause changing. This exact leaf is among the **15,700 inert entries**. The gate consumes only the saved `authenticated` boolean. Deleting either hash also passes. [Line 571](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:571). |
| Set `reread/bodies/20260905/arms/a_kabsch_guarded/bent_tercile/i_orientation_deg` to **100°** | **MERGE**, with no clause changing. The six-body median remains unchanged, but this body’s follower ratio becomes **14.6138 / 100 = 0.146138**, violating ≥2×. The gate uses the duplicated `winner_i_deg=5.71179`; it cross-checks only the follower’s numerator against its body row. [Line 442](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:442), [line 452](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:452). |
| Set the selected arm’s seed `20260903`, `whole_take/n_frames` to **0** | **MERGE**, unchanged. Another inert leaf: the gate reads the medians without validating their population. [Line 379](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:379). |
| Replace performer 0’s control-1 `failing_channels` with `["local::Head"]` | **MERGE**, control still PASS. Head is outside the protected foot channels; the gate accepts any nonempty list. [Line 668](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:668). |

The required fixes follow directly: derive authentication from both hashes, cross-check the follower denominator against the selected body’s measurement, validate selector populations, and require the controls’ named channel failures.

**The 226 REPORT-only classification is not consistent with the card.** Its actual breakdown is:

- **164** affect REPORT clauses.
- **40** affect the explicitly excluded diagnostics clause, currently labelled PASS.
- **14** affect the preserved historical sigma-1 FAIL.
- **8** affect **enforced P1 controls**.

The diagnostics and historical exclusions remain acceptable. The eight control entries are misclassified. The deciding code is simply `elif moved:` followed by the REPORT-only justification; it never checks clause status or conjunct membership. [Fuzzer line 169](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_fuzz.py:169). Likewise, “enforced” means **at least one mutation failed**; it does not establish consistency under other mutations, as the 100° follower example demonstrates. [Line 158](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_fuzz.py:158).

On round-5 adoption and reproduction:

- The original round-5 attacks are caught by **e68e06d**. They still escape **db3066c**.
- B6’s inversion conclusions and “coin toss” text are removed. **The centroid correction remains incomplete:** the generated JSON still says the averaged matrix is “not the skinning field at the centroid.” [Generator line 407](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_delivered_bytes.py:407). Under barycentric weights it is; transforming the centroid differs from averaging transformed vertices. The Jacobian additionally requires the weight-gradient contribution in [Kavan equation 17](https://skinning.org/direct-methods.pdf).
- I regenerated **every B6 JSON value exactly**, reproduced **38 passing tests**, and reproduced the **entire fuzz output exactly**: 18,172 / 2,246 / 226 / 15,700. Strictly, 18,172 counts **13,763 leaves plus 4,409 containers**.
- The “equal floats are interned” explanation does not reproduce in this runtime: separately decoded equal floats are distinct; repeated small integers share identity.
- I did **not** rerun the two pre-existing failures. Both commits’ `src/` diffs against `9dda9ac` are empty.

Read-only throughout; no files changed.

---

# Merge review round 7 — 2026-09-14, at ladder/D7c cffaad3. Verdict: NO MERGE (gate enforcement, the only blocker)

Verified: `d7c_gate_report.py:811` B2 reads the aggregate `same_denominator` while its per-subject constituents exist in the report; `:644` P1 trusts `bit_identical` while `frames_that_differ` and the contact counts exist (`d7c_projection_preservation.py:128`); `:488` the follower reads its metric bypassing `per_body()`'s population check; `:381` the calibration reads the stored 8.7495 aggregate while the per-body values exist (a 100 mm per-body value moves the derived median to 9.028); `:972` the "every other saved boolean" claim is false (`:726` oracle P1 consumes more); `:93` attributes a 49-pair requirement to round 6 that round 6 never made. The three `np.array_equal` families may remain instrument outputs (Astra compared the NPZ arrays: 4/4 byte-identical).

| # | finding | change |
|---|---|---|
| 1 | four leaves with constituents in the reports still bypass derivation | each derived; and the rule inverted in the fuzzer: every UNREAD leaf is classified (label / provenance / diagnostics / measurement) and every unread MEASUREMENT leaf under a subtree any clause reads is a GAP unless justified by name |
| 2 | the saved-boolean inventory incomplete; the 49-pair attribution wrong | inventory regenerated mechanically from the gate's reads; the attribution removed |

---

**NO MERGE at `cffaad3`. Gate enforcement remains the only blocker, confidence 10/10.**

I reproduced four additional independent mutations that return **MERGE with all 46 clauses unchanged**. These also answer question 3:

| Mutation, using the gate’s report aliases | Missed enforcement |
|---|---|
| Set `b2/triangulated_landmarks_byte_identical_across_arms/subject_00/D7c` to `false` | B2 reads only `value = r.at("b2", "same_denominator")`, ignoring its available constituents. [Gate:811](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:811) |
| Set `projection/subjects/subject_00/P1_channel_preservation/channels/local::LeftFoot/frames_that_differ` to **150** | P1 still trusts `bit_identical=True`. Changing the left delivered-contact count from **36→0** also escapes. Cross-checking `failing_channels` against those booleans does not validate them against the reported measurements. [Gate:644](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:644), [measurement producer:128](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_projection_preservation.py:128) |
| Set `reread/bodies/20260903/arms/frozen_pitch_follower/bent_tercile/n_frames` to **0** | The follower reads its metric directly, bypassing `per_body()` and its population validation. Setting its `n_pairs` to zero also escapes. [Gate:488](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:488) |
| Set `calibration/calibration/bisection/evaluations/9/per_body_mm/20260903` from **8.9585→100** | The accepted evaluation’s derived median becomes **9.028 mm**, missing **8.7636 mm** by **0.2644 mm**, outside **0.05 mm**. The gate still reads the stored **8.7495 mm** aggregate. [Gate:381](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:381) |

All four representative leaves are classified `read_by_no_clause` by the saved fuzzer. Apply the existing population-validation and summary-consistency requirements to these paths before merging.

On question 2: **the three listed equality families can remain instrument outputs; moving landmark comparison into the gate is not an additional merge prerequisite.** I independently compared the delivered NPZ arrays: raw and smoothed, both performers, **4/4 byte-identical**.

However, the claimed inventory is incomplete. P1 directly consumes additional saved equality booleans, and B2 consumes an aggregate boolean whose constituents already exist in its report. Therefore “every other saved boolean … is now derived or cross-checked” is false. [Inventory claim:972](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:972), [oracle P1:726](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:726).

On question 4:

- **All four original round-6 attacks now return NO MERGE.** The broader population and boolean-audit claims remain incomplete, as demonstrated above.
- **47 bent-tercile pairs reproduces on every body**, alongside 150/149 and 50 frames. The recorded merge round 6 contains no requirement for 49 pairs; that attribution at [gate:93](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:93) is unsupported by the [recorded finding](/Users/abhi_macbook/Projects/apps/AutoAnim/docs/reviews/pelvis-rest-astra-merge-review-2026-09-14.md:353).
- **The entire current fuzz output reproduces exactly:** 18,172 = 13,763 leaves + 4,409 containers; 2,336 enforced, 0 gaps, 164 REPORT, 40 diagnostics, 14 historical-FAIL, 15,618 unread; **0 monotone-check failures**. All **eight** previously misclassified control leaves are now enforced. Those counts do not establish completeness.
- **38 tests pass.** Every B6 JSON value reproduces, including the corrected artifact string. The interning explanation is appropriately withdrawn. Both historical STOPs remain FAIL.

Read-only throughout; regeneration and mutations stayed in memory. Worktree clean; `git diff 9dda9ac -- src/` empty.

---

# Merge review round 8 — 2026-09-14, at ladder/D7c 17dc09e. Verdict: NO MERGE (gate enforcement; three exemptions excuse banded evidence)

Verified: `d7c_gate_report.py:1439` reads B1's copied verdicts while `:1871` exempts `silhouette/subjects/**` (the ci95 constituents); `:1867` exempts `silhouette/statistics/**` including `every_arm_on_identical_draws`, which the card bands; `:579/:1803` the preserved σ-1 STOP reads the duplicated follower table and exempts the body measurements behind it; `:304` `startswith(ROOT + "/")` accepts the nested worktree from main and rejects an equal checkout elsewhere; `:1155` contact COUNTS cannot derive mask identity (a moved contact keeps [36, 36]); `:2134` "4,165 leaves read" = 3,511 scalars + 654 containers; the review's inventory prose stale (103 families, old totals).

| # | finding | change |
|---|---|---|
| 1 | B1's eight cells, identical draws, and the historical follower table bypass validation through exemptions | the eight cells cross-checked against their subject/cut measurements and populations; identical draws enforced; the σ-1 follower table validated against its body constituents; an exemption may never cover a measurement the card bands |
| 3 | the provenance check is tied to a path prefix | tied to a source fingerprint (content hash of the resolved module) and the build mode per stage; the absolute path kept as provenance only; hygiene on unchanged C, the C-held tripwire and the E candidate kept distinct |
| 4 | contact counts do not derive mask identity; the leaf count mislabelled; stale prose | the array-equality flag kept explicitly trusted or frame differences published; counts relabelled; prose regenerated |

---

**1. NO MERGE at `17dc09e`. Gate enforcement remains the only blocker.** B1’s banded evidence and the preserved STOP’s constituents still bypass validation through unjustified exemptions.

**2. Three reproduced attacks.** Each returns **MERGE / COVERED / NAMED, with all 49 clauses unchanged**:

| Mutation using report aliases | Missed enforcement |
|---|---|
| Set `silhouette/subjects/subject_00/cuts/whole_take/arm_D7c_minus_D9b/ci95/1` from **0.00115 → −0.0001** | The underlying interval now establishes worsening. The gate reads its unchanged copy under `preregistered_clause_verdicts`. The producer explicitly copies this measurement; the audit exempts the source through `silhouette/subjects/**`. [Gate:1439](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:1439), [producer:349](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_pelvis_rest_silhouette.py:349), [exemption:1871](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:1871). |
| Set `silhouette/statistics/every_arm_on_identical_draws` to **false** | The card explicitly requires identical draws. `silhouette/statistics/**` incorrectly excuses this requirement as “per-frame overlap statistics.” [Card:156](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/docs/LADDER_EXECUTION_PLAN.md:156), [exemption:1867](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:1867). |
| Set `sigma1/bodies/<seed>/arms/a_kabsch_guarded/bent_tercile/i_orientation_deg` to **1° on all six seeds** | The constituent follower ratios become **15.469–21.110×**, clearing both requirements everywhere. The gate nevertheless retains the original STOP by reading the duplicated follower table. Its justification explicitly exempts the body measurements behind that stop. [Gate:579](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:579), [exemption:1803](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:1803). |

Setting B1’s underlying whole-take population from **150 → 0** independently produces the same escape.

Before merging, cross-check the eight B1 cells against their subject/cut measurements and populations, enforce identical draws, and validate the historical follower table against its body constituents. Naming an exemption does not justify exempting evidence the card actually bands.

**3. The coordinator’s predicted main-checkout path failure does not occur in this layout.**

I changed `ROOT` in memory to `/Users/abhi_macbook/Projects/apps/AutoAnim`, retaining all existing reports: **MERGE / COVERED / NAMED**. The old worktree lies beneath that directory, so `startswith(ROOT + "/")` accepts it. The check therefore permits the wrong nested checkout when run from main. [Check:304](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:304).

Conversely, equivalent artifacts from a checkout outside that prefix would fail solely because of location.

Tie artifact validity to the **expected source revision/content fingerprint and build mode for that stage**. At execution, check the resolved module against the exact intended checkout/module; retain its absolute path as provenance. The current producer records a path and descriptive `src_state`, without a source fingerprint. [Producer:343](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_pelvis_rest_delivery.py:343).

Preserve the distinction between historical hygiene on **unchanged C**, the refactored **C-held tripwire**, and the **E candidate**. Rebuilding current E against D9b and demanding historical hygiene byte identity would test the wrong stage. [Hygiene obligation:45](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/docs/reviews/pelvis-rest-2026-09-14.md:45).

**4. Adoption and reproduction.**

- **All four original round-7 attacks now return NO MERGE.** The changed calibration median reproduces as **9.028 mm**, **0.2644 mm** outside the target. The 49-pair attribution is withdrawn.
- **Contact counts do not derive mask identity.** Moving performer 0’s left contact from frame 21 to frame 0 preserves counts **[36, 36]** while changing the mask. Round 7 required consistency with those counts; it did not make them sufficient proof of bit-identity. Keep the array-equality flag explicitly trusted, or publish frame differences. [Current derivation:1155](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:1155).
- **“4,165 leaves read” is mislabeled:** it is **3,511 scalar leaves + 654 containers**. The implementation counts every touched path. [Counting line:2134](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:2134).
- The baseline clauses, audit and inventory reproduce exactly: **49 clauses; 35 PASS / 12 REPORT / 2 FAIL; thirteen conjuncts; 1,416 cross-checked reads; 106 trusted families; 7,904 unread measurements in 86 families**.
- **Every fuzz bucket and row reproduces exactly:** 18,172 visited; 4,918 enforced; 148 REPORT; 40 diagnostics; zero preserved-STOP; 13,066 unread; zero reported gaps or monotone failures. The **368-path** transition from `880fc1b` also reproduces, including **150 containers**.
- Prose still says **103** trusted families and retains obsolete unread totals; the newer table and brief are correct. [Stale inventory:1073](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/docs/reviews/pelvis-rest-2026-09-14.md:1073), [stale totals:1110](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/docs/reviews/pelvis-rest-2026-09-14.md:1110).

**38 focused tests pass.** Full suite not rerun. Read-only throughout; worktree clean; `git diff 9dda9ac -- src/` empty.

---

# Merge review round 9 — 2026-09-15, at ladder/D7c f56d27d. Verdict: NO MERGE (two areas)

Verified: `d7c_gate_report.py:1532` requires only a nonempty mask-cache map (`silhouette.py:244` loads a named cache); `d7c_source_fingerprint.py:104` `fingerprint_now` emits neither `stage` nor `build_order/log_mtime`, which `:327/:352` require, and `d7c_pelvis_rest_gate.py:673` emits no fingerprint at all; `:149` the retrospective stamp overwrites `source_fingerprint` wholesale; `:2600` the sweep check is vocabulary membership, not applicability; `:356` hygiene compares against its duplicated timestamp map, not the tripwire's own value; prose 117 vs 120; `:1862` calls B2's distances MAMMA-referenced (they are against our own capture, `delivered_vs_capture.py:531`); `:1560` equal `draws_used` counts are not draw identity (the producer's shared draw list at `d7c_pelvis_rest_silhouette.py:289` is).

| # | finding | change |
|---|---|---|
| 1 | the consumed mask cache not required by identity; the producer-to-gate fingerprint contract broken (fields missing, the instrument producer emits none, the retrospective stamp overwrites genuine ones) | cache by identity; the producers emit the full contract at build time and the gate is verified against a genuine producer output; the retrospective stamper never overwrites a genuine stamp |
| 2 | the sweep check verifies vocabulary, not applicability; the stage-order check reads a duplicated map | applicability per conjunct (a reason is legal only for the conjunct the card gives it for); stage order derived from each stage's own value with all five stages present |
| 3 | "by construction today" honest, insufficient as implemented | resolved by 1 |
| 4 | three overclaims | the prose count, the B2 reason, draw identity from the shared draw list |

---

**1. NO MERGE at `f56d27d`. Two blocking areas, confidence 10/10.**

- **Required evidence can still disappear or contradict its source without failing the gate.** Deleting `mask/masks-960x540-A001_B001_C001_D001.npz` from the silhouette’s cache-proof map leaves **MERGE / COVERED / NAMED**, all 50 clauses unchanged. That is the cache the reader actually loads; the gate merely requires a nonempty map. Require the consumed cache by identity. [Gate:1532](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:1532), [reader:244](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/silhouette.py:244). The stage-order escape below needs the same constituent and membership checks.

- **The promised build-time fingerprint handoff is broken.** Substituting the actual return from `fingerprint_now("E_rig_rest_kabsch")` into the delivery report produces **NO MERGE: missing `stage`**. Adding only `stage` exposes **missing `build_order/log_mtime`**. The helper emits neither; the gate requires both. Moreover, the oracle/take producer still emits **no fingerprint at all**. Complete and verify the producer-to-gate contract without subsequently replacing genuine stamps with retrospective ones. [Helper:104](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_source_fingerprint.py:104), [requirements:327](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:327), [requirements:352](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:352), [instrument producer:673](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_pelvis_rest_gate.py:673).

**2. Both requested attacks escape.**

| In-memory attack | Result |
|---|---|
| Give `FAMILY_SWEEP["silhouette/subjects/**"]` the reason **“a preserved recorded STOP”**, retaining its B1 grouping. The card gives B1 no such exclusion. | **MERGE / COVERED / NAMED**, all clauses unchanged. The check is only `row[1] not in WHY_NOT_BANDED`; it verifies vocabulary membership, not applicability to that family. [Check:2600](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:2600). |
| Change the tripwire’s own `build_order/log_mtime` to **1789397462**, one second before hygiene’s **1789397463**. | **MERGE / COVERED / NAMED**, all clauses unchanged. Hygiene compares against its duplicated timestamp map, not the tripwire’s own value. Deleting four of that map’s five stage entries also passes. [Check:356](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:356). |

**3. “Holds by construction today” is honest; the disposition is insufficient as implemented.**

I reproduced the retained pre-change hash and the actual log ordering. Keeping historical hygiene explicitly retrospective is reasonable. But the planned close-out rebuild cannot currently supply evidence the gate accepts through its genuine producer path. Repair that handoff before relying on “the next build.” The retrospective stamping command otherwise overwrites `source_fingerprint` wholesale. [Stamp replacement:149](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_source_fingerprint.py:149).

**4. Reproduction and remaining overclaims.**

The three round-8 attacks now return **NO MERGE**. All headline counts reproduce, including **50 clauses; 36/12/2; thirteen conjuncts; 3,681 scalar leaves + 664 containers; sweep 81/16; inventory 1,450/120**. I reran every fuzz row: **18,360 visited, 5,130 enforced, zero gaps, zero monotone failures**, with no row differences. All **30 silhouette cells** reproduce from retained arrays, including **1,838–2,000 draws** and the five-frame-floor explanation.

Three corrections remain:

- Prose still says **117 trusted families**, versus the reproducible **120**. [Prose:1083](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/docs/reviews/pelvis-rest-2026-09-14.md:1083).
- The exemption calls B2’s distances **MAMMA-referenced**. They are against the delivery’s own captured landmarks. [Wrong reason:1862](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:1862), [producer:531](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/delivered_vs_capture.py:531).
- Equal `draws_used` counts establish count consistency, not draw identity. The producer’s shared draw list supports identity; calling the count comparison its measurement repeats round 8’s count-versus-identity mistake. [Claim:1560](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_gate_report.py:1560), [shared draws:289](/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c/tools/compare/d7c_pelvis_rest_silhouette.py:289).

**38 focused tests passed.** Full suite not rerun. Read-only throughout; worktree clean; `git diff 9dda9ac -- src/` empty.