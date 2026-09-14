# Round 5 for Astra GPT6 — D7c stopped at S; the fixture question — 2026-09-14

The agent ran hygiene (8/8), the instrument (every pre-card figure reproduced; the wrong-origin control caught only by the
metre residual; `frozen_upright` 7.2166° and the D9b build 6.865° fail O1 as required), then S — and STOPPED before any
`src/` change, as the card instructs. Branch `ladder/D7c` (worktree `.claude/worktrees/ladder-D7c`), commits d146aa3,
7d5f3a3, ee4b71f, 8a82ee4; its review `docs/reviews/pelvis-rest-2026-09-14.md` (§3.2 the attribution, §3.3 why it did not
repair the fixture, §3.4 the sensitivity sweep, §4 the (a)-restatement); reports `artifacts/compare/d7c-pelvis-rest/selector*.json`.

## What S read (σ scale 1.0, the pre-registered fixture)
- (a) `E_rig_rest_kabsch` beats (b) `D_rig_rest_hipline` on all six cells (three metrics × two populations) — no split.
- The winner beats C-on-SOMA on (i): 9.549 vs 15.970 (whole), 11.217 vs 15.510 (bent).
- The frozen-pitch follower reads 15.47–21.11° on the bent tercile on every body (≥ 2° holds).
- **STOP: follower / winner ratio 1.991, 1.925, 1.448, 1.573, 2.742, 1.755 — the ≥ 2× clause fails on 5 of 6 bodies.**
- Attribution (the selector's own `fixture_attribution` rows): with NO noise both rig modes read 0.0000° and the follower
  14.401° (its structural floor = the true pelvis pitch about the hip line relative to gravity) — the discrimination is
  unbounded; the fixture's noise alone raises the winner to 6.9–12.2°. The fixture's `midhips→Spine1` sd is **20.07 mm**
  (median of six; 13.7–26.2 per body) against the take's **6.61 / 11.10 mm** measured by `d7_pelvis_rigidity.py` on the real
  detector — 1.8–3.0× too harsh, the same discrepancy D7's own report recorded (`calibration_sweep` in
  `d7_pelvis_synthetic.py:441`, "the noise model is 1.7–2.9x too harsh", recorded there as a post-hoc instrument repair).
- The take's sd is CONFOUNDED: the D3 rig is rigid so the synthetic spread is pure noise, while the take's mixes noise with
  real lever variation. The take's sd is therefore an UPPER bound on the take's observation noise.
- Sensitivity sweep (REPORT ONLY, refuses to overwrite `selector.json`): σ 0.5 → lever sd 13.1 mm, ratios 2.28 / 2.12 /
  **1.88** / 2.18 / 2.65 / 2.08 (STOP on one body); σ 0.35 → 9.1 mm, ratios 2.47–3.08 (PROCEED); σ 0.25 → 6.6 mm, 3.30–4.10
  (PROCEED). (a) wins all six cells at every σ. G2 at σ 0.35/0.25: guarded 3.2–5.7° vs unguarded 62–77°, miss rate 0.033.
  G1's bit-identity holds on 5 of 6 bodies (seed 20260904: the guard demotes one extra frame on its own).

## The coordinator's proposal (nothing run yet)
Apply the lane's fixture-repair rule (CLAUDE.md, D8b, 2026-09-06: "a step's fixture can fail a pre-registered clause for the
fixture's own defects — measure the discrepancy, repair the FIXTURE as a fixture parameter with src byte-identical across the
repair, rerun the same clauses; never merge on an override and never move a band"):
- **Calibration rule, fixed before the rerun:** one σ scale for all six bodies, chosen so that the median-of-six synthetic
  `midhips→Spine1` sd equals the take's measured sd on the SAME statistic, taking the LARGER performer (11.10 mm) — the
  conservative side of the confound (the take's sd is an upper bound on its noise, so the calibrated fixture is still at
  least as harsh as the real detector). By the sweep that σ lies between 0.35 and 0.5 (~0.42), where the verdict is
  genuinely open (PROCEED at 9.1 mm, STOP at 13.1 mm) — the rerun is a test, not a foregone pass.
- `src/` byte-identical across the repair (nothing under src has changed); the 2× band untouched; every S clause rerun at
  the calibrated σ with the same draws law; G1/G2 rerun there; the σ-1.0 verdict kept in the report as it fell.
- Then the agent resumes: the card's (b)-conditional predictions restated from the pre-card's (a) arm and a GUARDED (a)
  rebuild; B2's hip clause restated as a measured residual (not zero by construction); tripwire, O1–O3, P1–P3, B1–B6.

## Questions
1. Is this a fixture repair (a fixture parameter, src untouched) or a band move in disguise? If a repair, is the calibration
   rule sound given the confound — is "the larger performer's sd" the right conservative side, and is one σ for all six
   bodies right when the per-body spread at σ 1.0 runs 13.7–26.2 mm?
2. If you would NOT repair: what is the alternative — re-pin the 2× band with a stated reason (a band move the lane
   refuses), or declare the fixture non-discriminating and ship (a) on the remaining conjuncts (a pass the card forbids)?
3. (a) wins at every σ and beats C-on-SOMA at every σ: does that stand as the selection regardless of the follower clause,
   or must the whole of S be re-read at the calibrated σ before any selection is recorded?
4. The (a) restatement: what must be restated from the (a) arm before delivery, and does B2's hip clause become a band or a
   report under (a)?
5. G1's 5-of-6: is one extra guard demotion on seed 20260904 a defect in the guard, the fixture, or the equivalence claim?
