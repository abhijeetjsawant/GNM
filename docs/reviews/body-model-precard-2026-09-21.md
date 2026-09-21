# The body model in the delivery path — pre-card measurement, 2026-09-21

Pre-registered before the number was read (`artifacts/compare/d4-body/PREREGISTRATION.md`, reproduced below).
Scripts: `tools/fitter/fit_delivery_mhr.py` (momentum fits MHR to the delivered take's own raw triangulated landmarks,
pinned locator offsets, per-frame pose, skinned GLB at 30 fps), `tools/compare/blender_export_mesh_momentum.py`,
`tools/compare/d4_silhouette_paired.py` (the committed silhouette instrument's own rasteriser, scorer, masks and
tracklet map; paired moving-block bootstrap, block 15, 2000 resamples, identical draws). Reports under
`artifacts/compare/d4-body/`.

## The number

Median silhouette IoU against the SAM2 masks, 960x540, whole take, per performer (0 / 1):

| arm | pooled over 4 cameras | A | B | C | D |
|---|---|---|---|---|---|
| D7c rig delivery (baseline) | 0.647 / 0.652 | 0.634 / 0.650 | 0.696 / 0.658 | 0.635 / 0.674 | 0.680 / 0.569 |
| MHR mean body, pose tracked (control) | 0.744 / 0.722 | 0.726 / 0.744 | 0.781 / 0.705 | 0.737 / 0.734 | 0.794 / 0.634 |
| **MHR fitted, pinned offsets (candidate)** | **0.789 / 0.730** | 0.779 / 0.743 | 0.796 / 0.720 | 0.792 / 0.737 | 0.810 / 0.633 |
| MAMMA's mesh (committed oracle, reported) | ~0.87 / ~0.84 | 0.871 / 0.848 | 0.846 / 0.862 | 0.875 / 0.842 | 0.879 / 0.708 |

Paired, 600 frame-camera cells per performer, median difference [95 % CI]:

| comparison | performer 0 | performer 1 |
|---|---|---|
| fitted − rig | +0.142 [+0.120, +0.150] | +0.078 [+0.053, +0.093] |
| mean body − rig | +0.097 [+0.078, +0.108] | +0.070 [+0.044, +0.088] |
| fitted − mean body | +0.045 [+0.032, +0.058] | +0.007 [−0.001, +0.015] |

**Verdict under the pre-registered rule: PASS on both performers — the route is not wrong.**

## What it says, honestly

- Most of the gain is the BODY MODEL, not the fit: a plausible mesh (MHR's mean body) driven by our own pose takes
  0.65 → 0.74 / 0.72; fitting the 68 scale channels adds +0.045 on performer 0 (CI clear) and +0.007 on performer 1
  (CI through zero). Consistent with FITTER_PLAN §7: pinned offsets under-fit by the convention offset they refuse to model.
- The instrument is the committed one: the comparer's baseline reproduces the D7c close-out's `ours_delivered` medians
  to four decimals on all 8 cells, AND (after SMPL-X was restored from the Modal volume on 2026-09-21) the full
  `silhouette.py` run on the fitted arm reads MAMMA's mesh BIT-IDENTICAL to its committed value on all 8 cells
  (`artifacts/compare/d4-body/silhouette-fitted.json`) with the fitted medians matching the comparer's (0.779 / 0.743, ...).
- Denominator caveat: the MHR arms consumed the RAW triangulated array; the D7c rig was built from the smoothed array
  with D8/D8b/D8c's repairs. Both are scored on identical masks, so the verdict stands, but "body model vs rig" is
  confounded with "raw vs repaired input". An integration step feeds the smoothed array.
- Mesh resolution: MHR `lod6` (595 vertices) against the MPFB asset's thousands — the bias runs against the candidate;
  `lod0` exists for whatever ships.
- The bootstrap's block resampling runs across the four concatenated camera series (minor).
- Blind to: depth along the ray; clothing; the fit's joint accuracy; the pelvis and hip conventions.

## State of the machine

`.cache/` had been wiped between 2026-09-15 and 2026-09-21 (the user cleaned it for storage). Restored the same day: `.cache/mhr`
(re-downloaded), `SMPLX_NEUTRAL.npz` and the four fixture videos from the Modal volume `autoanim-mamma-data-v1`, the
calibration yaml from the pinned MAMMA repo. Still gone (not needed by this lane's rebuild):
`.cache/mamma` (the fixture videos and calibration yaml `post_merge.sh` rebuilds from; `SMPLX_NEUTRAL.npz` the MAMMA
oracle rasterises), `.cache/autoanim_gnm` fixtures, the GEM-X outputs. No in-place rebuild and no MAMMA oracle until
restored; both are licence-gated and were not re-downloaded.

## The decision this hands the user

What was measured is MHR's OWN mesh delivered. Two integration shapes: (a) ship MHR's mesh as the body — what this
number supports, the cheap path; (b) drive the MPFB character from the MHR skeleton — D6 re-skinning, not measured here.

## Pre-registration, verbatim
# D4 pre-card pre-registration, written 2026-09-21 BEFORE the silhouette is read

Candidate: MHR (own release, 68 raw scale channels, no SAM PCA) fitted by momentum to the delivered take's own raw
triangulated landmarks -- two-stage calibration with locator offsets PINNED (limit_weight 10, FITTER_PLAN section 7's
honest setting), per-frame pose tracked, exported as a skinned GLB at 30 fps. MAMMA enters nowhere.
Baseline to beat: the D7c rig delivery (artifacts/commercial-multiview-soma77), median IoU per camera 0.62-0.69.
Controls that must not win: MHR's MEAN body (identity scales zero) posed by the same tracker -- what a gamed fit with free
offsets collapses to; the instrument's own frozen-pose control.
Reported, never selected: MAMMA's mesh through the same rasteriser (0.84-0.89), bit-identical to its committed value or
the instrument moved.
Instrument: tools/compare/silhouette.py unchanged, scale 4 (960x540), the same masks cache, all four cameras, whole take.
Verdict rule: the candidate beats the D7c delivery on BOTH performers on the median IoU pooled over the four cameras,
with a paired moving-block bootstrap (block 15, identical draws) interval clear of zero -- or the route is declared WRONG
rather than unfinished. The mean-body control beating the candidate would mean the fit is not what wins.
Blind to: depth along the viewing ray; clothing (in the masks, in neither mesh); the fit's accuracy (a silhouette scores
outline and placement, not joints); the pelvis/hip conventions.
