# Merge review round 5 for Astra GPT6 — D7c at ladder/D7c 843bce5 — 2026-09-14

Rounds 1–4 are recorded at `docs/reviews/pelvis-rest-astra-merge-review-2026-09-14.md`. The agent answered round 4 in two
commits (c08f8ac the gate, 843bce5 the proxy and corrections). `git diff 9dda9ac -- src/` empty. Worktree
`/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D7c`. Read-only, adversarial, cite the deciding line.

## What the agent reports
- Gate: your five round-4 counter-examples read NO MERGE — six seeds by identity for G1 and G2 with the G2 aggregates
  recomputed from per-body values; oracle P2's worst derived per seed from its own runs (`d7c_projection_preservation.py`
  now emits `run_measurements`); the take's P2 derived from the runs with every run's `holds` required; the eight named B1
  cells. General rule applied: named populations by identity, derived aggregates, and where a report also stores a summary
  the gate cross-checks it (four new clauses). 39/39 mutations detected.
- The inversion reading is labelled `carried_tetrahedron_PROXY` with both demonstrated failures in the artifact (your
  rigid two-bone example at det +0.72; the vertex-order dependence with the moved ranges); no inversion claim; the skinning
  Jacobian (Kavan direct methods eq. 17) named and handed to D6; the tests now say what they pin (constant skin) and two
  document the failures.
- Corrections (a) planar Kabsch (b) the B1 sentence (c) closure medians 3e-6 / 4e-6°. Tests 38 passed; two pre-existing failures.

## Questions
1. MERGE at 843bce5, or not? If not, ONLY what blocks.
2. Try at least two mutations you have not tried, including one that corrupts a stored summary while its constituents stay valid.
3. Is the proxy's qualification honest as written, and does any B6 text still draw a conclusion the proxy cannot support?
4. Anything that misreads round 4, and any number you cannot reproduce.
