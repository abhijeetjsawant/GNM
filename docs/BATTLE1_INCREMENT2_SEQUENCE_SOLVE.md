# Battle 1, increment 2 — spatiotemporal sequence solve

Status: landed 2026-08-27. Plan: `docs/OWNED_BODY_CAPTURE_PLAN.md`.

## The diagnostic that shaped it

Per-frame triangulation needs two views. Before this increment, 7.3% of
joint-slots had none. Broken down by how many cameras actually saw the joint:

| cameras seeing it | slots | share of missing | |
|---|---:|---:|---|
| 0 | 9 | 2.2% | unrecoverable — no evidence exists |
| **1** | **344** | **82.7%** | **a single ray: underdetermined alone, recoverable with priors** |
| 2 | 62 | 14.9% | failed the inlier/cheirality gate |
| 3 | 1 | 0.2% | as above |

Worst joints: wrists 18–19%, eyes/ears 17–19%, elbows 11%.

So the ceiling was ~97.8%, and four fifths of the shortfall was a *single ray* —
real evidence that per-frame triangulation simply cannot use.

## What was built

`solve_sequence_positions` solves a whole take at once per subject:

- **reprojection** for every confident 2D observation, *including singletons*
- **limb length** as a percentage error against a per-shot length
- **temporal** second difference, normalised by the take's own mean per-frame
  displacement

A calibrated ray meets the sphere of known limb length about the parent in at
most two points — limb toward the camera or away — and the temporal term picks
between them. (An earlier draft of this doc said "circle"; that is the case where
the length is known but the ray is not.)

Residual scaling follows Anipose (Karashchuk et al., *Cell Reports* 2021),
reimplemented from the published description: soft-l1 on the reprojection block
only, limb error as a percentage so one weight covers a forearm and a thigh
alike, and the temporal term motion-normalised. Anipose is BSD-2-Clause; no code
was vendored.

Two deliberate deviations, both measured:

- **Limb lengths are held fixed** at the median over directly-triangulated
  frames. Anipose frees them. With wrists 19% missing, free lengths drift toward
  whatever makes the residual small.
- **Head landmarks carry no limb constraint.** Nose, eyes and ears have no rigid
  bone to a parent, and GNM owns the head anyway. They keep the existing
  neck-fallback path.

## The metric integrity problem, and how it was handled

Filling slots and counting them as `valid_joint_fraction` would have pushed the
number past 90% by **redefinition** — the same defect class as Battle 0's
shifting pixel gate, one layer up. Instead:

- `valid_joint_fraction` still means exactly what it always meant: per-frame
  triangulation succeeded from two or more views. **It did not move: 0.8823
  before and after.**
- `constraint_recovered_joint_fraction` is new: slots resolved from at least one
  ray plus limb and temporal constraints.
- `interpolated_joint_fraction` keeps its meaning and now covers only slots with
  no observation at all.

A recovery is evidence-based; an interpolation is not. That is why they are
separate numbers, and why the gate below is **restated rather than met by a
changed definition**.

## Result

| metric | before | after |
|---|---:|---:|
| `valid_joint_fraction` | 0.8823 | **0.8823 (unchanged)** |
| `constraint_recovered_joint_fraction` | — | 0.0575 |
| `interpolated_joint_fraction` | 0.1177 | **0.0602** |
| `median_reprojection_error_px` | 4.5890 | 4.5890 (unchanged) |
| `temporally_rejected_subject_frames` | 14 | 14 (unchanged) |
| retarget endpoint median | 158.1 mm | 157.0 mm |
| **direct + constraint-recovered** | 0.8823 | **0.9398** |

Interpolation roughly halved. Runtime 43.7 s for two subjects over 150 frames.
Verifier passes; 43 tests green.

Note what did **not** change: every pre-existing metric. The solve deliberately
does not overwrite slots that already triangulated — measured, letting it do so
moved them by a median of 11–14 mm and up to 700 mm, the temporal and limb terms
outvoting good geometry. It is a recovery pass for missing evidence, not a
re-estimator of present evidence.

## The finding that changes the plan

**Battle 1's bone-length exit gate — "bone lengths stable within 2% across a
take" — is not reachable with Apple Vision, and no geometry work will reach it.**

Measured on frames where *both* endpoints triangulated directly, so neither
interpolation nor recovery is involved:

| limb | sd | as % of length |
|---|---:|---:|
| right shoulder→elbow | 20.9 mm | 7.2% |
| right elbow→wrist | 26.9 mm | 9.5% |
| left shoulder→elbow | 87.9 mm | 26.8% |
| left elbow→wrist | 51.3 mm | 18.6% |
| right hip→knee | 17.6 mm | 4.1% |
| right knee→ankle | 23.4 mm | 5.7% |
| left hip→knee | 26.0 mm | 5.9% |
| left knee→ankle | 25.3 mm | 6.0% |

Median across limbs and subjects: **10.4%**, against a 2% gate.

This is not triangulation noise. Per-frame scatter of the same directly
triangulated joints against a 5-frame median is only **3–7 mm**. The limb length
is varying because the detector's idea of where a shoulder or a wrist *is* moves
with pose — the same model-limited conclusion Battle 0 reached about precision,
now visible in a second, independent measurement.

**Consequence for the plan:** the ≥90% coverage half of Battle 1's exit is met.
The 2% bone-length half should be restated as a Battle 4 gate, because it is a
property of the detector, not of the solver. Restating it is not lowering the
bar — it is putting the bar where the work is.

## Limitations

- Recovery is capped at the 97.8% ceiling; 9 slots on this fixture have no
  observation at all and stay interpolated.
- Head landmarks are excluded by design, and they are 17–19% of the missing mass.
  They recover only through reprojection and temporal continuity, not limb length.
- Chained single-ray joints (an unresolved parent) degrade toward
  temporal-interpolation-only depth. Not measured on this fixture.
- The two-solution branch ambiguity is resolved by the temporal term. Under fast
  motion a wrong branch could latch; the unit test constructs the ambiguity but
  does not stress fast motion.
- 43.7 s per 150-frame two-person take. Anipose chunks past ~1000 frames with 15
  pinned overlap frames; not implemented, not needed at this length.
- **Review depth is lower than increments 0 and 1.** Those each had a multi-agent
  adversarial review, which found a critical regression both times. This one had
  the advisor design and completion checkpoints only.
