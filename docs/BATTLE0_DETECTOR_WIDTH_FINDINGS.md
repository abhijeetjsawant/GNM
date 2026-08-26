# Battle 0 — detector working resolution

Source data: `artifacts/battle0-detector-width/` (gitignored; regenerate with
`scripts/build_commercial_multiview_comparison.py --detector-width {1280,1920,3840}`).

Run 2026-08-26/27. Identical footage for every condition: MAMMA example fixture,
4× 3840×2160 @30fps, frames [60,210), 2 performers, calibration
`iphones_outdoors.yaml`. The A001 input SHA matches the recorded baseline run,
and the w1280 condition reproduces the baseline median reprojection error to 16
significant figures (4.632413734670453) — the pipeline is deterministic and this
is a controlled comparison.

**Headline: the hypothesis was wrong, and so were two of my own analyses. The
actionable win turned out to be JPEG quality, not resolution.**

---

## 1. The naive table — do not act on it

mm/px = 5.57 / 3.71 / 1.86 at 1280 / 1920 / 3840, cross-checked geometrically
(rig-centre depth ÷ focal) and empirically (median depth of triangulated joints
along each camera's optical axis).

| width | repro med | p95 | max | bone sd | valid | interp | temporal rej |
|---|---|---|---|---|---|---|---|
| 1280 | 25.8 mm | 61.5 | 110.7 | 38.0 mm | 82.8% | 17.2% | 33 |
| 1920 | 21.8 mm | 46.9 | 74.8 | 37.3 mm | 78.0% | 22.0% | 35 |
| 3840 | **9.6 mm** | 23.6 | 38.0 | 39.8 mm | **58.2%** | **41.8%** | 52 |

Reads as "native is 2.7× better". It is an artifact.

## 2. Why — survivorship from a fixed pixel gate

`src/autoanim_gnm/commercial_multiview.py` carries hard-coded pixel constants
that never scale with detector width:

- `triangulate_point(..., inlier_threshold_px: float = 14.0)` — line 241
- `inlier_threshold_px=40.0` in the association solve — lines 409, 496
- association cost mixes units: `12.0 × support_deficit` + `50.0 × metres`
  + `0.25 × pixel screen_steps`, capped at `250.0` px — line 428 onward

`CalibratedCamera.scaled` (line 117) scales intrinsics. **It does not scale the
gates.** So 14 px is a 78 mm physical gate at 1280, 52 mm at 1920, **26 mm at
3840** — and the reported error is computed only over observations that passed
it. At 3840 that discards 41.8% of observations and keeps the easy ones.

Two independent corroborations that this is censoring, not accuracy:

- **Stratified by inlier count, error rises with width.** 4-camera solves fall
  35.6% → 25.0% → **2.3%** while 2-camera solves rise 31.0% → 37.9% → **65.6%**.
  A 2-camera DLT fits 3 DOF to 4 numbers so its residual is near-zero by
  construction. Within the 4-camera stratum the residual *rises* 5.59 → 7.55 →
  9.30 px. Only the mixture makes the aggregate fall.
- **`maximum_reprojection_error_px` is a readout of the constant**: 19.89 /
  20.15 / 20.46 px across a 3× resolution change. Uncensored maxima are
  500–1300 px.

**This is a defect in its own right:** the pipeline's accept/reject behaviour
silently depends on detector resolution.

## 3. Controlled — precision is flat

All detections renormalised into a common 1280-px frame before reconstruction,
so the gates bite identically and only sub-pixel detector precision varies.

| detector | repro med | p95 | bone sd | valid | interp | temporal rej |
|---|---|---|---|---|---|---|
| 1280 | 25.8 mm | 61.5 | 38.0 mm | 82.8% | 17.2% | 33 |
| 1920 | 25.8 mm | 60.7 | 33.6 mm | 85.5% | 14.5% | 20 |
| 3840 | 25.5 mm | 60.5 | 33.5 mm | 85.4% | 14.6% | 21 |

Precision flat. Robustness apparently better at 1920/3840 — **but see §5, that
turns out not to be a resolution effect either.**

## 4. Apple Vision saturates below 1280

A direct sweep of the detector at 320 / 480 / 640 / 854 / 1280 / 1920 / 2560 /
3840 on the same frames: median normalised temporal jitter is 2.284e-3 at 480 px
and 2.196e-3 at 3840 — **3.9% change for 8× the pixels** — with mean confidence
flat (0.741 → 0.751). Only 320 px is measurably worse (3.054e-3), which shows the
metric does have sensitivity.

Renormalised to a common frame, Vision's 2D output moves only ~5.9 mm at the
subject between 1280 and 3840. **At most ~6 mm of improvement was ever
available** against a ~26 mm residual.

Note the pipeline runs one whole-frame `VNDetectHumanBodyPoseRequest` with no
per-person ROI crop. For a fixed-input network, native pixels only pay off
through cropping. A null on `--detector-width` does **not** license "image detail
doesn't matter" — a crop stage is a separate, untested lever.

## 5. The actual win: JPEG quality, not resolution

`_extract_frames` passes no `-q:v`, so JPEG quality is a free variable that
degrades as resolution rises (mean luma quantiser 20.3 / 24.4 / 33.0; mean frame
33 KB / 69 KB / 245 KB). So "higher resolution" and "worse compression" were
confounded in every condition above.

Full quality x width factorial. Robustness columns are directly comparable
across every cell; reprojection mm is **not** comparable across widths (each
width is censored by the same 14 px gate at a different physical size), so it is
omitted here — see §2.

| condition | bone sd | valid joints | interpolated | temporal rej |
|---|---|---|---|---|
| 1280 default | 38.0 mm | 82.8% | 17.2% | 33 |
| **1280 `-q:v 2`** | **33.7 mm** | **88.2%** | **11.8%** | **14** |
| 1920 default | 37.3 mm | 78.0% | 22.0% | 35 |
| 1920 `-q:v 2` | 39.4 mm | 81.9% | 18.1% | 25 |
| 3840 default | 39.8 mm | 58.2% | 41.8% | 52 |

Quality helps at both widths (1280: 82.8% → 88.2%; 1920: 78.0% → 81.9%) — that
is a clean within-width comparison, same gates, same everything. And with quality
pinned at both, **1280 beats 1920 on every column.**

So the robustness gain §3 credited to resolution was an encoder artifact: bigger
frames got more compression, smaller frames got less, and what actually changed
was how much JPEG noise the detector had to see through. **`-q:v 2` at 1280 is
the best cell in the factorial.**

## 6. Errors in my own analysis, corrected

- **Quaternion ordering.** My analysis scripts passed the rig's `wxyz`
  quaternion straight to `scipy` `Rotation.from_quat`, which expects `xyzw`.
  `commercial_multiview.py:218` does the reorder; my scripts did not. This made
  mm/px come out 1.93× too small (2.89 instead of 5.57) and understated the
  triangulation Monte-Carlo by the same factor. **Corrected throughout.**
- **Jitter was measured on smoothed data.** `triangulated_world_positions_z_up_m`
  is the output of `_fill_and_smooth_positions` — `np.interp` across every gap
  then `savgol_filter(9, 2)` — with **0 NaN cells in 5700**. Interpolated
  segments are jitter-free by construction, so the 1.9 mm figure I reported is
  not raw triangulation noise and cannot be repaired from the shipped npz. The
  pre-fill array with NaNs intact would need to be persisted.
- Consequently my earlier claim that "triangulation contributes ~6 mm, so the
  gap is systematic bias" was wrong. Corrected figures: ~25 mm median / ~59 mm
  P95 at 1280.

## 7. Verdict

Apple Vision carries **~26 mm of 2D error at the subject, and that does not
improve with input resolution.** It is model-limited, not pixel-limited. That
error alone triangulates to ~25 mm median / ~59 mm P95 — a large share of our
60–80 mm gap.

So the detector is the programme, exactly as
`docs/OWNED_BODY_CAPTURE_RESEARCH.md` argues — but for a sharper reason than
"semantics": Vision is simply not precise enough at any resolution, and no rig
change reaches past it.

## 8. Actions — 1, 3, 4, 5 APPLIED 2026-08-27

1. ✅ **Pinned `-q:v 2` in `_extract_frames`, staying at 1280.**
2. **Do not build a 3840 lane.** No precision gain, 9× the pixels. (Nothing to do.)
3. ✅ **Pixel constants now scale with detector width.** `triangulate_point` and
   `associate_frame` take a keyword-only `pixel_scale`; `reconstruct_multiview`
   derives it from the observation width against a new
   `REFERENCE_DETECTOR_WIDTH_PX = 1280`. Metre-denominated gates (the
   bounded-acceleration tolerance, the 0.28 m body-volume gate, the 2.0 m and
   1.0 m clamps) are deliberately left unscaled and now say so in comments.
4. ✅ **`run-report.json` records `detector_width` and `frame_jpeg_quality`.**
   `_extract_frames` also writes an `extraction.json` stamp so a settings change
   invalidates cached frames — previously frames were cached by path alone, so
   the `-q:v` change would silently not have applied to any existing run.
5. ✅ **`raw_triangulated_world_positions_z_up_m` added to each subject npz** —
   pre-interpolation, NaNs intact. The existing smoothed array is untouched, so
   the verifier's all-finite requirement still holds.

### Verified end to end

The threshold refactor is a **no-op at the reference width**: replayed against
the cached 1280 frames it reproduces the prior baseline bit-identically on all
seven reported metrics (median reprojection `4.632413734670453`,
p95 `11.052257609066443`, max `19.886388217808165`, valid `0.8277192982456141`,
interpolated `0.17228070175438598`, temporal rejections `33`, association
objective `18.195939844400847`).

A fresh full run with the encoder pinned (`artifacts/commercial-multiview-q2/`):

| metric | before | after |
|---|---|---|
| valid joint fraction | 82.8% | **88.2%** |
| interpolated | 17.2% | **11.8%** |
| temporal rejections | 33 | **14** |
| median reprojection | 4.632 px | 4.589 px |
| p95 reprojection | 11.052 px | 10.877 px |
| retarget endpoint median | 164.0 mm | **158.1 mm** |
| retarget endpoint P95 | 378.0 mm | **364.8 mm** |

`scripts/verify_commercial_multiview_artifact.py` returns `"status": "pass"`.
33 tests pass, including three new ones:

- `test_inlier_gate_is_invariant_to_detector_width` — same accept/reject set and
  position at 1× / 2× / 3×, and the unscaled path demonstrably censors an
  observation the reference width accepts.
- `test_association_cost_scales_homogeneously_with_detector_width` — the cost is
  homogeneous of degree one in `pixel_scale`, so the argmin is invariant.
  **Mutation-tested:** removing the scale from any of the four terms
  (`12.0` support deficit, `50.0` root continuity, `60.0` joint continuity,
  the `250.0` screen-step clamp) makes it fail. The fixture deliberately drops a
  subject from one camera to force a non-zero support deficit, and offsets the
  previous observations past the clamp, or those two terms go untested.
- `test_pixel_scale_rejects_nonpositive_values`.

### A defect the review caught in the fix itself

The first version of the frame-cache stamp was **half a fix, and worse than
none**. `_extract_frames` re-extracted on a settings change, but `_detect`
caches detections on line count alone — which does not change when the frames
behind it are re-encoded, re-sized, or taken from a different window. So a
re-run into the existing artifact directory would have re-extracted frames at
`-q:v 2`, silently reused the detections made from the *old* frames, and written
a report stamping `frame_jpeg_quality: 2` on metrics that no truthful run can
produce. Before the stamp existed, frames were never re-extracted, so frames and
detections stayed stale-but-coherent; the stamp broke that coupling.

Now fixed: `_extract_frames` returns whether ffmpeg actually ran, and the caller
deletes the observations when it did. The stamp also carries `source_sha256`, so
swapping the footage under an unchanged output directory invalidates the cache.
Verified across all five triggers (settings unchanged → no re-extract; JPEG
quality, source SHA, frame window, and missing stamp → re-extract, detections
regenerated).

Two further review findings applied: `detector_width` in the report is now read
from the observations actually reconstructed from rather than the CLI argument;
and the verifier pins `detector_width == 1280` / `frame_jpeg_quality == 2` and
cross-checks both — plus the source SHA — against each camera's extraction stamp
rather than trusting the report's self-reported fields. The verifier's 6/12/25 px
gates are detector-native, so they only mean what they say at the calibrated
width.

**Recorded, not fixed** (pre-existing, outside this change): in
`_fill_and_smooth_positions`, `nose` (joint 0) hard-fails where eyes/ears (15–18)
succeed on identical input because the neck fallback is evaluated inside the
joint loop, and when raw neck is finite the nose branch assigns unsmoothed
values. Fixing it changes reconstruction output and needs its own verification.
6. Prefer **leave-one-camera-out reprojection error in mm** as the primary
   accuracy metric — the in-sample residual is both fitted and gate-censored.
7. A **per-person ROI crop stage** is an untested lever worth its own experiment;
   a local probe found crop-vs-full disagreement ~4× larger than the width effect.

## Provenance

Conditions in `w1280/`, `w1920/`, `w3840/`, `w1280q2/`. Analysis scripts in the
session scratchpad: `geom_floor.py`, `tri_floor.py`, `battle0_analyse.py`,
`battle0_controlled.py`, `detector_delta.py`, `detector_matrix.py`,
`velocity_fixed.py`, `q2_compare.py`. Design audit: 53-agent workflow
`battle0-design-audit`, 19 confounds surviving adversarial verification.
