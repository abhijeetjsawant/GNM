# Battle 1, increment 1 — cross-view association

Status: landed 2026-08-27. One increment of Battle 1, not the whole battle.
Plan: `docs/OWNED_BODY_CAPTURE_PLAN.md`.

## What was replaced

The clean-room pipeline associated detections across views by **exhaustive
search**: enumerate every per-camera assignment of detections to subjects, take
the product across cameras, and score each candidate by triangulating every core
joint of every subject.

That search space is `(subjects!)^cameras`:

| cameras | subjects | assignments | exhaustive | graph | speedup | same answer? |
|---|---|---|---|---:|---:|---|
| 4 | 2 | 16 | 0.65 s | 0.05 s | 12× | yes |
| 4 | 3 | 1,296 | 67.75 s | 0.08 s | **852×** | yes |
| 6 | 2 | 64 | 16.16 s | 0.36 s | 45× | yes |
| 4 | 4 | 331,776 | *intractable* | — | — | — |
| 6 | 4 | 191,102,976 | *intractable* | — | — | — |
| 8 | 4 | 110,075,314,176 | *intractable* | — | — | — |

**The pipeline could not have done three performers, and could barely do six
cameras.** Battle 2's shoot list includes crossing identities and multi-person
staging, so this was a hard blocker on the next increment, not a tidy-up.

## What replaced it

`associate_frame_graph` in `src/autoanim_gnm/commercial_multiview.py`:

1. Score every cross-view detection **pair** once by **median symmetric
   epipolar distance** over joints both cameras can see. Median, not mean, so a
   single mislabelled limb does not decide identity.
2. Match each camera pair with the **Hungarian algorithm**, keeping
   non-assignment available — an occluded person has no counterpart, and forcing
   one is how ghost identities appear.
3. Grow people as **connected components** under the rule that a person holds at
   most one detection per camera. That rule *is* cycle consistency: a triangle
   violation would require two detections from one camera in one component.
4. Order components onto stable subject slots — by capture world X with no
   history (the existing determinism contract), by Hungarian match to the
   previous frame's roots otherwise. A **soft** prior, applied once per frame;
   the bounded-acceleration gate downstream remains the safety net, so one bad
   frame cannot pin identities forever.
5. Score the chosen assignment with `_score_assignment`, extracted from the old
   path so both strategies optimise **the same objective** and
   `association_objective_median` keeps meaning the same thing.

`reconstruct_multiview` now takes an `associator` parameter, so the two paths
stay comparable and a regression is diagnosable by flipping one argument.

## Measured before cutting over

Reference fixture, 150 frames, 2 performers, same cached detections:

| associator | valid | interp | temporal rej | bone sd | identity switches | assoc objective | seconds |
|---|---|---|---|---|---|---|---|
| exhaustive | 88.2% | 11.8% | 14 | 33.7 mm | 0 | 18.37 | 85.8 |
| **graph** | 88.2% | 11.8% | 14 | 33.7 mm | 0 | 18.37 | **43.7** |

**Frames where the two assignments differ: 0 / 150.** Reprojection identical to
three decimals. The verifier passes on a fresh artifact with byte-identical
metrics (`median_reprojection_error_px = 4.588984179617748`).

## Two heuristics tried and rejected, by measurement

- **Rank components by size.** Fails when a view reports a phantom third person:
  the phantom is epipolar-consistent with a real detection, steals its node, and
  no choice of components recovers the optimum. Cost 4.57 higher on frame 25.
- **Complete-linkage merging** (every member consistent with every member).
  Intended to reject phantoms; measured *worse* — it fragments components and
  changed the chosen assignment on 7 frames rather than 3.

What works: single-linkage merge, plus **defer to the exhaustive search on
frames where any view reports more people than the shot contains**. That is
where identity is genuinely ambiguous, it is 3 frames in 150 here, and it is why
the output is identical rather than merely close.

## Honest limitations

- The speedup on *this* fixture is ~2×, not the 12–852× above, because the
  ambiguity guard pays full exhaustive cost on surplus-detection frames. The
  large numbers are what matters for Battle 2's multi-person captures.
- A better phantom-robust merge would remove the guard and recover the full
  speedup. Not attempted; two candidate heuristics were measured and rejected.
- Coverage is 88.2%, below Battle 1's ≥90% exit gate. Association was never the
  binding term there — the spatiotemporal triangulation increment is.

## Battle 1 remaining

| increment | state |
|---|---|
| cross-view association | ✅ this document |
| AniPose-style spatiotemporal solve with per-shot bone lengths | not started — this is what should move coverage to the ≥90% gate |
| SAM2 masks + temporal identity propagation | not started |
| detector swap | **blocked pending a viability probe.** MediaPipe 0.10.35's *face* graph aborts natively on this host (`docs/MAMMA_MULTIVIEW_EXECUTION.md`); the *pose* graph is unverified. Probe before planning it in. |
