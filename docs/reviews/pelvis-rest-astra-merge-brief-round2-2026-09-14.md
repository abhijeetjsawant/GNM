# Merge review round 2 for Astra GPT6 — D7c at ladder/D7c 0b3eba4 — 2026-09-14

Your round-1 merge review (NO MERGE at 9dda9ac) is recorded at `docs/reviews/pelvis-rest-astra-merge-review-2026-09-14.md`
with what each finding changed. The agent answered every item in five commits (fa28865 blockers, ffc215a B6 + provenance,
0f1f6f5 the B1 attribution, 257c1d2 the corrections and the clause table, 0b3eba4 the gate re-run). Read the worktree
`/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c`: `docs/reviews/pelvis-rest-2026-09-14.md` (§5 is
now the single current clause table), `artifacts/compare/d7c-pelvis-rest/gate.json` (twelve explicit conjuncts, each
flipped to FAIL one at a time to prove it turns the verdict), `projection-preservation-oracle.json` (oracle P2 now measured),
the B6 and B1-attribution artifacts, `tools/compare/d7c_gate_report.py`, `tools/compare/provenance.py`, `tests/test_pelvis_rest.py`.
Read-only, adversarial, cite the deciding line.

## What the agent reports
- Oracle P2: six bodies exported through the real exporter; Foot and Toes at their run anchors from each GLB's own arrays
  against the frozen masks: worst 4.86e-7 m vs 1e-5, 10–20 runs per seed, 6/6. New conjunct.
- The gate: an explicit conjunct table; the S entry names every stop (calibration REACHED, the (a)/(b) split, C-on-SOMA, the
  follower, G1, G2); the flip experiment found SIX further unenforced clauses (both must-fails, the tripwire's second reading,
  both P1 controls, B1's oracle, landmark byte-identity), now conjuncts; twelve conjuncts, each turns the verdict; one stated
  exclusion (the run-report diagnostics block).
- B6: rotational comparison reported and NOT called a closure (32° = the node rest rotation); hierarchy joint-for-joint,
  bone-length error 0.0 mm on 54; between-key playback by interpolating quaternions then FK: 5.2–8.6 mm median inside a run
  (the 0.0003 mm withdrawn); increments on normalised quaternions, median 0.0°. **Inverse binds and mesh deformation NOT
  ESTABLISHED:** skin-at-node-rest misses POSITION by 590 / 156 mm IDENTICALLY on the shipped build (exporter or the
  reader; Blender renders both fine); attribution established as pre-existing, measurement not; handed to D6 with the reader.
- B1 attribution: performer 0's torso rise is the ARTICULATION (+0.0119 vs root −0.0048); performer 1's is the ROOT
  translation (+0.0055 vs articulation −0.0025), so performer 1's rise is not evidence about the pelvis; the arm cells' ~0 are
  two opposite effects cancelling.
- Corrections: Head world reconstructed from the GLB's channels, between-build 4e-6° median / 1.3e-5° max, identity withdrawn;
  world-vertical compared to the truth PELVIS (16.8228 vs 17.7256°), the limitation applies and S still proceeds; G2 10.6× on
  corrupted frames, 3.16× on transition pairs; the follower ratio a tendency; the pin imports D7's fixture and pins 7.567708°
  to 1e-4; the 5g items. §5 rewritten as the current table with both STOPs and both post-hoc amendments.
- Provenance: `RIG_REST_PELVIS_MODES` registered as a scoping predicate with no fitted value; `PELVIS_FRAME_SOURCE`'s entry
  rewritten; `test_provenance_audit` passes. Full suite in the worktree (before these commits): 7 failed, 1216 passed —
  four pelvis-frame pins the coordinator re-pins in place at the merge, two pre-existing (fail on D9b's worktree), one now fixed.

## Questions
1. MERGE at 0b3eba4, or not? If not, ONLY what blocks.
2. The B6 mesh-deformation reading is NOT ESTABLISHED because the reader's skin-at-node-rest misses the mesh by 590 / 156 mm
   on BOTH builds. Is "pre-existing, handed to D6 with the reader" an acceptable disposition of a card commitment, or must the
   reader be fixed before this step closes?
3. Does the twelve-conjunct gate now enforce the card's merge rule as written, and does the one exclusion hide anything?
4. Any number you cannot reproduce, and anything adopted that misreads round 1.
