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