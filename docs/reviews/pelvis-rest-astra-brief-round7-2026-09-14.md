# Round 7 for Astra GPT6 — the calibration stopped on its own monotonicity precondition — 2026-09-14

The agent ran the frozen calibration exactly as amended (ladder/D7c 210405d, 1fd9baf; `artifacts/compare/d7c-pelvis-rest/selector-calibrated.json`; estimators frozen at 8a82ee4, `src/` untouched). Every precondition passed but one:
- target reproduced: 5.9944 / **8.7636** mm (guard-kept, ddof=0); synthetic statistic guard-kept and ddof=0; zero-noise baseline through `observe_body` 1.8131 mm (rig modes 0.41–0.51°, follower 14.37–14.64°, C-on-SOMA 6.81–6.89°), never subtracted; bracket [0.10, 1.00] = [3.0777, 12.5108] contains the target; original draws bit-identical over two runs;
- bisection: 0.10 → 3.0777, 1.00 → 12.5108, 0.55 → 11.9565, 0.325 → 8.4760, 0.4375 → 10.6170, 0.38125 → 9.3457, 0.353125 → 8.8275, 0.339063 → 8.8410, 0.332031 → 8.6582, **0.335547 → 8.7495 (|Δ| = 0.0141, inside 0.05, the 10th evaluation)**;
- **in σ order the statistic DECREASES once: 0.339063 → 8.8410, then 0.353125 → 8.8275 (−0.0135 mm)**. Cause (the agent's attribution, verified in the JSON): at 0.353125 the guard newly rejects one frame on two bodies (20260903 frame 84, 20260907 frame 21, each that body's largest-lever frame), their sd falls by 0.02 / 0.007 while the other four rise by 0.19–0.39, and those two are the 3rd and 4th of six, so the median dips. Exactly the keep-mask-changes-with-σ case the amendment named.
- The frozen wording says "not monotone in σ across the evaluations ⇒ unreachable, STOP". The agent applied it, stopped, and did not argue it. The matched fixture is 1.43× the take (12.5108 vs 8.7636), not the 1.8–3.0× first stated on the unmatched raw-triangulation row.

## The coordinator's reading, for you to refute
The precondition's purpose was well-posedness: one σ answers the target. The dip is 0.0135 mm (27 % of the tolerance), lies ABOVE the accepted σ and the target, and every evaluation above 0.339 reads ≥ 8.8275 > 8.7636 + 0.05, so the target level is crossed once and no second σ lies inside the tolerance; the final sub-bracket 0.332031 → 0.335547 → 0.339063 is strictly monotone. Proposed amendment, in the reviewer's words if you accept it: "non-monotone" means a decrease between σ-adjacent evaluations LARGER than the tolerance (0.05 mm), or more than one crossing of the target level across the evaluations; a smaller dip from a keep-mask change is recorded and is not a stop. Under that wording the calibration is REACHED at σ 0.335547 and the reread of S proceeds there. The σ-1.0 STOP and this STOP both stay recorded.

## Questions
1. Amend the precondition as proposed, or retain the STOP? If you amend, give the wording.
2. Is anything about σ 0.335547 selected on the verdict rather than on the target (the sweep at 0.35 PROCEEDed; the agent ran the bisection blind to S)?
3. Anything else the calibration record must carry before S is reread.
