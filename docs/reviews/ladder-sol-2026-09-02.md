# Sol's review of the ladder execution brief — 2026-09-02

Sol is the advisor. It saw the brief in `ladder-brief-2026-09-02.md` and the session
transcript up to that point, and not Codex's answer. Transcribed verbatim.

---

**Q2 first, because it changes Q1: the strongest case against Step A is in your own
scoreboard.** The sized-rig arm already exists and measures what a fitted body's
*proportions* buy through today's converter: 151.6→136.6 and 137.8→113.0. Fifteen to
twenty-five mm of the ~100. The plan's own retarget decomposition (BODY_LANE_PLAN ~line
637, `retarget_cost.py`) says the same: performer bone lengths recover 19–35% of the
retarget gap. The other 65–80% is the converter: the clavicle-origin defect (36–47 mm on
arms) and "no rig joint origin sits where its captured landmark does." Rung 2's 27.7/40.1
didn't change one part; it changed two — body model *and* pose recovery (analytic
converter → least-squares fit). The commit says so. So "the 100 mm is the rig" is true
but "rig" is two parts, and the ladder can't split them yet. Building an MHR shape fitter
and wiring it through the current analytic converter would land near 136/113, not near
28/40, and you'd have spent the build on the smaller lever. Two caveats on the sized arm:
it reuses rotations solved on the canonical body (check `mamma_scoreboard.py` —
`local_rotations_xyzw` from the delivered track applied to the sized skeleton), so it's
a lower bound on proportions; and 15 mm vs 25 mm across performers is the
bimodal-population trap the board already records.

**Q1, ranking.** Instruments before builds, and every instrument below reads retained
files only:

0. **Split rung 7 into its two parts, from disk.** (a) Re-run `positions_to_body_track`
   on the sized skeleton, not just FK of old rotations → proportions-only arm. (b)
   `pred_joints` via `PAIRS` into the converter → converter cost on clean input (your
   Step B). (c) Give `retarget_cost.py` JSON and register it. Three scripts, no new
   capability, and they decide whether Step A is a shape problem or a converter problem.
   Also the oracle arm the whole ladder lacks: **project `pred_joints` into the four
   cameras and run our entire pipeline on that perfect 2D** — association,
   triangulation, sequence, smoothing, converter — scored vs `pred_joints`. That's the
   floor our pipeline imposes with a perfect detector; it's the CLAUDE.md oracle
   principle applied to rungs 2–7 at once, and it's cheap.
1. **Pose recovery (rung 7 proper): MHR mean body + momentum IK on our triangulated
   joints**, shape held at mean. One part changed: the solver. Band relative to the
   SMPL-X arm on identical frames.
2. **Shape (rung 6): fitted MHR on top of 1.** Now the two parts have separate figures.
3. Detector instrument (your C) — but see Q3; the 2.2× figure is not same-denominator.
4. MAMMA's feet as the bar (D). 5. Hands as a report (E). 6. Mesh instrument (Q4). 7. F
   only if 3 says bulk. 8. Fixture.

**Q3, per step — what's blind, what degenerate solution passes.**
- *Step A band "within 10 mm of the SMPL-X arm":* not same-axis. The SMPL-X arm shares
  the joint regressor with `pred_joints`; MHR's joints are defined differently (hand
  root 2.5–3 cm from SMPL-X wrist, documented). An MHR arm carries a per-joint
  definition offset the oracle doesn't. Score bias and spread separately (the
  BATTLE1_BODY_PROFILE method) and band on spread; keep bias visible, never subtract it
  silently. Degenerate solutions: an "IK" that places joints exactly on the triangulated
  points scores 36.1/41.3 and passes "beat the sized rig" while the body did nothing —
  require constant bone lengths *and* the vs-our-capture control between 0 and the raw
  distance; and the 0.00% bone-length tautology from the hand fit applies to any
  body-model arm. Note the SMPL-X oracle beat raw capture on P0 only (40.1 vs 41.3 on
  P1), so "must beat raw capture" would fail the oracle itself — miscalibrated. Band
  against the oracle, with the block bootstrap the head work already built (lag-1
  autocorrelation 0.99). *[Sol's later correction: 40.1 does beat 41.3; the point is
  that a 1.2 mm margin is inside any block-bootstrap interval, so the band is not
  robustly passed by the oracle.]*
- *Step B:* pricing rung, fine. Canonical round-trip 0.00 is the control. Output goes
  under `artifacts/compare`, never a delivery directory.
- *Step C:* the 2.2× bulk is each detector against *its own* 3D — ours against raw
  triangulation, MAMMA's against a body-regularised fit that consumed those landmarks.
  Not the same denominator; part of 2.2× is how much a body model absorbs. The clean
  detector figures are reference-free: cross-view self-agreement (`sam3d_ladder.py`) and
  per-camera static translation (`common_mode.py`, which already has a shuffled
  control). Register those; retire the 2.2× as a headline.
- *Step D:* band on foot direction relative to the shin, not world; controls: constant
  axis and foot-welded-to-shin. Blind: MAMMA's feet at 5 m may themselves be noisy —
  measuring that *is* the point.
- *Step E:* the angle-sd band is a quantity the temporal term optimises directly. Per
  CLAUDE.md it proves nothing about the hand; held-out reprojection is the only
  discriminating band. A frozen hand has sd 0 — the pair must move together.
- *Step F:* `use_sparse_body_ldmks` selects `self.body_idx` from MAMMA's own 512. Before
  any Modal work, confirm the loss accepts a joint-target term rather than a landmark
  subset. The cheaper mirror is item 0's oracle arm.

**Q4, what's missing.**
- *Surface.* `ma_masks` — 1,832 SAM2 files on disk, "not used, deliberately" — is the
  one reference on disk that is *not* model-mediated. Silhouette IoU of our rendered
  mesh per camera per frame against them, with MAMMA's own mesh rendered against the
  same masks as the oracle/bar. Degenerate: an inflated blob maximises recall — report
  precision and recall, not IoU alone. Blind to depth and everything inside the
  silhouette. Second instrument: symmetric point-to-surface distance to `pred_vertices`
  (no topology correspondence needed). The delivered rung's joint figures are blind to
  the 180° facing defect because it lives in mesh binding; a mesh instrument is what
  catches that class.
- *Temporal.* Not unmeasurable for *lag*: hold out one camera, reproject the smoothed 3D
  into it at time t, compare against the raw detection — transverse lag is observable
  in projection. It can't score accuracy (depth), and the doc should say exactly that.
- *Sync* stays where it is.

**Q5.** Research before code: MHR identity regularisation (running); the 127-joint
layout of `pred_joints` for ankle/foot indices; `masks.npy` shape and MHR render path
for silhouettes. Everything in item 0, 3, 4, 5 and the mesh instrument is read-only
against retained files and independent — run in parallel. Items 1 and 2 are sequential
and one part each; neither starts until its instrument has written a report and a
history line exists.

**Q6, leak paths.** Any MHR regulariser weight, locator-offset pin, foot-axis convention
or hand temporal weight *selected* by its MAMMA-referenced arm is a shipped constant
fitted on a research artifact. Selectors must be held-out-camera reprojection, the
vs-our-own-capture control, or anatomy. The MAMMA arm reports; it never selects. A
per-joint MHR↔SMPL-X offset table is instrument-only and stays in `tools/compare`. Step
B's outputs and the oracle arm are MAMMA-derived and stay out of
`artifacts/commercial-multiview-*`.

Merge protocol: where Codex disagrees, record it inline, don't smooth it. The one claim
I'd defend against dissent is Q2 — the sized arm is on disk and its number is what it is.

---

**Sol's final-review additions (same day):** the feet figures must render with no
direction after f6a4973 (done: `better=COUNT`); verify `_thorax_frames` is reached from
the shipped head solve before the leak claim stands; save both reviewers' texts beside
the plan so the dissent register is auditable (this file and
`ladder-codex-xhigh-2026-09-02.md`).
