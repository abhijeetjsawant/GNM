# I8 — the provenance audit, and `THORAX_SMOOTHING_FRAMES` re-selected without MAMMA

**2026-09-02. Ladder step I8.** Two deliverables, both regenerable:

| what | script | report |
|---|---|---|
| provenance manifest for the delivery path | `tools/compare/provenance.py` | `artifacts/compare/provenance.json` |
| re-selection sweep for `THORAX_SMOOTHING_FRAMES` | `tools/head/thorax_window_sweep.py` | `artifacts/compare/thorax-window-sweep.json` |

Extractor stub: `tools/compare/extractors/i8_provenance.py` (`x_provenance`). It is not
registered — Fable owns `ladder.py`'s registry. Tests: `tests/test_provenance_audit.py`.

**Nothing under `src/` was changed.** Lane I is read-only against the pipeline; the
constant moves in lane D after review.

---

## Part 1 — the manifest

89 items across the delivery call graph (55 reachable functions in
`commercial_multiview.py`, `head_orientation.py`, `body_projection.py`, `soma_motion.py`
and `workers/commercial_multiview/soma77_pose.py`), plus the six data inputs the build
consumes. `provenance.py` exits **1** while any leak stands, so a delivery build can gate
on it.

| class | count |
|---|---|
| **MAMMA-DERIVED (leak)** | **1** |
| declared-and-known MAMMA input | 2 |
| unknown (no stated origin anywhere) | 17 |
| clean (anatomy 6, literature 1, synthetic truth 1, own capture 2, engineering limit 15, third party 7, asset 6, schema 31) | 69 |

### The leak

| constant | file:line | value | evidence |
|---|---|---|---|
| `THORAX_SMOOTHING_FRAMES` | `src/autoanim_gnm/commercial_multiview.py:1679` | `15` | The comment at :1678 says **"Chosen by the oracle arm alone"**. `_thorax_frames`'s docstring gives the sweep, whose two columns are *"oracle P1 median"* and *"oracle P1 p95"* — MAMMA's own head orientation expressed in our thorax frame. Commit `08a6c89`: *"Smoothing our thorax as a rotation, window chosen by the ORACLE arm alone"*. Its caller `_solve_head_for_subject` (:1854) is the shipped head solve, reached from `reconstruct_multiview` (:2120) with the pipeline's smoothed positions. |

One leak, and it is the one already known. **No second undeclared leak was found.**

The strongest single fact behind that: `run-report.json` records
`mamma=false, mamma_outputs=false, mamma_weights=false, smplx_model=false`, and **nothing
on the delivery path is trained in this repo at all** — the detectors are NVIDIA's and
Apple's vendor weights. There is therefore no training-example manifest to audit, and a
leak into weights is currently impossible because there are no weights of ours.
`LADDER_EXECUTION_PLAN.md` §2 already defers a pseudo-label detector until after lane H
for exactly this reason.

### Declared and known — listed so the audit does not condemn every build

| item | what it is | retired by |
|---|---|---|
| `artifacts/commercial-multiview-soma77/camera-rig.json` | **is** MAMMA's `ma_cap` calibration (ladder rung 0, *not owned*). sha256 `80d28a41…`, byte-identical to `artifacts/soma77-full/camera-rig.json`, which the synthetic fixture and this sweep also use. Every metric-scale figure in the lane rests on it. | lane H |
| MAMMA example take `pushing_and_lifting_from_ground`, frames [60, 210) | the licence-scoped footage. `run-report.json`: *"MAMMA example footage/calibration: research comparison only"*, `production_claim: false`. Not a constant, but it is the population every band in this lane is quoted on. | lane H |

### Unknown — 17 constants with no stated origin

None of these is MAMMA-derived; each predates or sits outside any MAMMA comparison.
`unknown` here means *the code, the docs and `git log -S`/`git blame` were searched and
none of them says where the number came from*. It is never guessed.

| constant | file:line | value | what the history does say |
|---|---|---|---|
| `inlier_threshold_px` | `commercial_multiview.py:315` | 14.0 | `6579460` states what 14 px *means* (~78 mm at the subject at 1280, ~26 mm at 3840) and makes it scale with detector width. Never where 14 came from. |
| `maximum_epipolar_px` | `commercial_multiview.py:659` | 60.0 | Introduced whole in `9eb0393`, which validates the associator against the exhaustive path (0/150 frames differ) but never the value. |
| `minimum_shared_joints` | `commercial_multiview.py:660` | 4 | Same commit, same silence. |
| `ambiguity_ratio` | `commercial_multiview.py:662` | 0.7 | `9b69d8b` validates the *result* on synthetic noise sweeps (0/60 disagreements at 0.5 px) — a check that the gate does not break the answer, not a selection of 0.7. |
| `ambiguity_margin_px` | `commercial_multiview.py:663` | 2.0 | Same commit, same silence. |
| `smooth_weight` | `commercial_multiview.py:998` | 2.0 | `16f24ec` cites Anipose for the residual *scaling* and the percentage limb error, and names two deviations — but not the weight. §4 lists "selecting `smooth_weight` toward MAMMA's amplitude" as a leak path *to avoid*, not one already walked. |
| `length_weight` | `commercial_multiview.py:999` | 2.0 | Same commit, same silence. |
| `0.72 * torso_up` (clavicle origin) | `commercial_multiview.py:1565`, `:1568` | 0.72 | The shoulder chain's origin is placed 72 % up the torso. Introduced whole in `6579460`. This is D2's named converter defect, worth 36–47 mm on the arms. A fitted-looking number that nothing fitted. |
| `neck_sigma_m` | `head_orientation.py:443` | 0.010 | `981e437` explains *why* the prior lives at the neck ("where anatomy says the smoothness lives") but not where 10 mm came from. |
| `template_prior` | `head_orientation.py:445` | 0.5 | No stated origin in code, docs or `981e437`. |
| `DEFAULT_WEIGHTS` grid extent | `head_orientation.py:45` | 0 … 1e5 | The *rule* that picks from the grid is documented and reference-free; the grid's endpoints and half-decade spacing are not. A grid too narrow would silently clamp the per-take selection. |
| `velocity_threshold_m_per_s` | `body_projection.py:1120` | 0.10 | Generated-track default. |
| `ground_band_m` | `body_projection.py:1121` | 0.035 | Generated-track default. |
| `minimum_contact_frames` | `body_projection.py:1122` | 3 | Generated-track default. |
| `maximum_root_correction_m` | `body_projection.py:1123` | 0.05 | Generated-track default. |
| foot-contact envelope on the capture path | `commercial_multiview.py:1665–1669` | 0.30 / 0.08 / 2 / 0.08 | The comment states the *reason* for overriding the four defaults ("calibrated capture contains noisier frame-to-frame ankle estimates than a generated track") but not where the numbers came from. Not MAMMA-derived: the delivered foot has been solved from SOMA-77's `ToeBase` since `f6a4973`, and I4 records that MAMMA's feet are **still unscored** — there was no MAMMA foot figure to fit against. |
| `BOX_PADDING` | `soma77_pose.py:58` | 1.25 | Introduced whole in `2eca1f3`. A real accuracy knob — it sets how much context the keypoint model sees — with no stated origin. |

### One meta-level item, recorded rather than flagged

`DEFAULT_WEIGHTS` classifies **clean (own capture)**: the temporal weight is swept *per
take* and chosen by minimum in-frame reprojection, so its execution consults only our own
observations, and `run-report.json`'s `temporal_weight: 100.0` is a measurement on that
take rather than a shipped constant. But CLAUDE.md lists *"the weight-selection rule"*
among the instrument repairs that fixed the head, and the head was scored on a gate whose
reference is MAMMA's head. So an **algorithm** — not a constant — was kept because it
improved a MAMMA-referenced gate. The flag rule tests where a *number* came from and does
not catch this. It is written into the manifest's `meta_level_caveat` and into
`blind_to`, because a reader should not conclude from a clean audit that the head solve
owes the reference nothing.

### What the audit is blind to

* **Inline numeric literals inside function bodies are curated by hand**, from a read of
  the delivery functions — not exhaustively enumerated. An inline magic number nobody
  noticed will not appear.
* **Constants that enter through a data file rather than through source.** That is exactly
  how the one declared MAMMA dependency enters.
* **An asymmetry in the scan:** module-level constants are collected from every scanned
  module regardless of reachability (conservative), while keyword defaults are filtered to
  the reachable call graph (a real gap for anything reached only from outside).
* **Provenance the history never recorded.** `git log -S` finds the commit that introduced
  a value; it cannot recover a reason nobody wrote down.
* The third-party detectors' training data, which is NVIDIA's and Apple's.

---

## Part 2 — re-selecting `THORAX_SMOOTHING_FRAMES` without MAMMA

**Truth** is exact: `_thorax_frames` itself, unsmoothed, on the forward-kinematic joint
positions of `build_synthetic_truth_fixture.py`'s take builder. **Noise** is our own
detector's self-consistency, injected in pixels per view and recovered through
`triangulate_point` on the real rig, so the depth anisotropy is present rather than hidden
behind an isotropic millimetre sphere. Two independent own-detector instruments agree on
the amplitude:

* `run-report.json` median reprojection 2.913 px (a 2D-norm post-solve residual; Rayleigh
  median 1.17741σ, least-squares deflation (2n−3)/2n = 5/8 at n = 4) → **σ = 3.13 px**;
* `sam3d_ladder.py`'s SOMA-77 one-sided cross-view epipolar median 3.11 px (|N(0, 2σ²)|,
  median 0.67449·σ·√2) → **σ = 3.26 px**.

The sweep uses 3.20 px, 12 seeds, and sweeps the amplitude at 0.5× and 2.0×.
`mamma_residuals.py` was not consulted.

### The finding that shaped the instrument: the fixture is far too slow as it stands

Our own capture's thorax turns at **66.0 / 73.2 deg/s median** and **182.5 / 195.8 deg/s
p95** (two performers, `artifacts/handfit-arrays/body-track.npz`). The tracked synthetic
clips at stride 1 turn at **0.0–22.7 deg/s median**. A window is measured in *frames*, so
what it costs and what it buys are entirely a function of how fast the thorax turns — at
that speed smoothing has nothing to lose and the p95 curve runs away to the widest window
on offer. That is not a selection; it is gate G10's defect one lane over. The fix is the
fixture's own mechanism: replay at `stride` times speed. Two of the five clips
(`autoanim_real/autoanim_fixture`, `cpu_smoke/autoanim_fixture`) turn at 0.0–0.9 deg/s and
are excluded from the headline and kept as a **declared degenerate arm**.

### The sweep — pooled over the three moving clips, 12 seeds

Angular error of the smoothed thorax frame against the exact frame, in degrees.
`atten` is the ratio of the candidate's p95 yaw rate to the truth's (1.0 = no attenuation).

| stride | speed vs own capture | window | p50 | **p95** | (seed sd) | lag (frames) | atten |
|---|---|---|---|---|---|---|---|
| 1 | 0.21× | none / 3 | 1.80 | 3.45 | 0.35 | +0.93 | 2.90 |
| | | 5 | 1.75 | 3.37 | 0.35 | +0.67 | 2.37 |
| | | 9 | 1.63 | 3.12 | 0.33 | +0.76 | 1.87 |
| | | 15 | 1.37 | 2.61 | 0.30 | +0.31 | 1.31 |
| | | **21** | 1.21 | **2.33** | 0.28 | +0.42 | 0.91 |
| | | 31 | 1.17 | 2.47 | 0.34 | +0.32 | 0.62 |
| | | 45 | 1.56 | 2.82 | 0.28 | +3.24 | 0.43 |
| 2 | 0.39× | none / 3 | 1.88 | 3.68 | 0.55 | +0.65 | 1.91 |
| | | 5 | 1.85 | 3.61 | 0.54 | +0.16 | 1.65 |
| | | 9 | 1.74 | 3.41 | 0.51 | +0.14 | 1.32 |
| | | **15** | 1.63 | **3.16** | 0.43 | +0.22 | 0.85 |
| | | 21 | 1.81 | 3.22 | 0.42 | +0.48 | 0.62 |
| | | 31 | 2.12 | 3.68 | 0.29 | +0.29 | 0.41 |
| **3** | **0.58×** | none / 3 | 2.03 | 4.10 | 0.64 | −0.46 | 1.49 |
| | | 5 | 1.99 | 3.98 | 0.65 | −0.86 | 1.33 |
| | | **9** | 1.90 | **3.86** | 0.56 | −0.55 | 1.12 |
| | | 15 | 2.01 | 3.91 | 0.51 | +1.77 | 0.68 |
| | | 21 | 2.30 | 4.43 | 0.55 | +2.22 | 0.54 |
| 4 | 0.77× | none / 3 | 2.16 | 4.20 | 0.49 | +1.01 | 1.24 |
| | | 5 | 2.12 | 4.15 | 0.52 | +1.14 | 1.11 |
| | | **9** | 2.16 | **4.11** | 0.49 | +0.13 | 0.97 |
| | | 15 | 2.36 | 4.46 | 0.45 | −1.74 | 0.65 |

Window 3 is byte-identical to *none* throughout, as it must be — a parabola through three
points is exact. That is the harness's self-check that it is really calling the shipped
`_thorax_frames`. Cells past a take's length are reported `n/a`, never silently folded in:
`_thorax_frames` returns the **unsmoothed** frame when the window exceeds the take, so a
too-wide cell would otherwise be "none" wearing a wide label.

### Interior optimum: yes, at every stride — and the argmin moves with speed

| stride | speed ratio | p95 argmin | interior? |
|---|---|---|---|
| 1 | 0.21× | 21 | yes |
| 2 | 0.39× | 15 | yes |
| 3 | 0.58× | **9** | yes |
| 4 | 0.77× | **9** | yes |

And with noise amplitude at the headline stride: **0.5× → 5** (interior), 1× → 9,
**2.0× → 21** (*not* interior — it runs to the end of the covered sweep). The optimum is a
bias–variance trade and it moves with the **ratio** of detector noise to thorax angular
speed, not with either alone. **A single frame count is not a transferable constant**, and
this sweep says so rather than hiding it.

### Is the p95 minimum real? The paired test

Seeds are independent draws (frames within a take are not — lag-1 autocorrelation 0.99),
so the paired sign test over (clip, seed) is legitimate. Share of paired draws on which a
window's p95 was **worse** than the argmin's, with the mean paired difference:

| stride | vs none | vs 5 | vs 9 | vs 15 | vs 21 |
|---|---|---|---|---|---|
| 3 (argmin 9) | 0.83 (+0.15°) | 0.69 (+0.07°) | — | **0.47 (+0.07°)** | 0.72 (+0.43°) |
| 4 (argmin 9) | 0.67 (+0.13°) | 0.61 (+0.06°) | — | **0.69 (+0.38°)** | n/a |

**At stride 3 the p95 rule does not separate 9 from 15** (0.47 of draws, +0.07°). It
separates both from *none* and from 21. At stride 4 — the nearest playback to the real
fixture's speed — 9 does beat 15, on 0.69 of paired draws by 0.38°.

Where p95 is flat, the **reported tiebreaker** (not part of the selection rule) is lag and
peak attenuation, and they are not flat at all: at stride 3, window 15 lags **+1.77
frames** and loses **32 % of the peak yaw rate**, against 9's ≈0 lag and 12 % *gain*. At
stride 4, 15 keeps 65 % of the peak yaw rate against 9's 97 %.

### Per-clip argmins are not a consensus — and that is informative

| stride | amy-cuddy (deg/s) | squat (deg/s) | will-stephen (deg/s) |
|---|---|---|---|
| 3 | 15 (41.5) | **9 (60.9)** | 31 (18.8) |
| 4 | 9 (48.0) | 3 (91.4) | 21 (21.5) |

The pooled 9 is an average over clips that disagree — but they disagree *monotonically in
their own thorax speed*, which is the same relationship the stride sweep shows. The single
data point closest to our own capture's median (squat at stride 3, 60.9 deg/s against the
real 66–73) picks **9**; pushed past it (squat at stride 4, 91.4 deg/s) it picks 3.

### Control arms

| arm | must | result (stride 3, per clip) |
|---|---|---|
| no smoothing at all, raw triangulated frame | show the jitter the docstring describes | p50 3.02–3.88°, **p95 5.92–7.45°** against the windowed 3.86°, with the yaw rate p95 inflated **3.5–8.5×** |
| frozen frame (the take's mean, constant every frame) | fail | fails on the two clips that turn: 5.26 / **10.47°** and 9.53 / **14.21°**, attenuation 0.00 |
| the two static clips as a degenerate arm | run away, not select | argmin 21, **not interior** — the widest window on offer, exactly as a thorax that does not turn should behave |
| noise at 2.0× | — | argmin 21, **not interior** — the rule refusing to select is visible when the trade is pushed |

**The freeze control must be read per clip, never pooled.** On
`will-stephen-acting-body`, which turns at only 18.8 deg/s, the frozen frame scores
1.26 / 2.33° and **beats every window**. A freeze is only a control on a thorax that
moves. The freeze is also built explicitly as the take's mean frame, because a
Savitzky-Golay filter at polyorder 2 fits a parabola and never freezes however wide it
gets — the brief's "a window so wide it freezes the frame" is not constructible.

### Proposal

> **9 frames**, bracketed by the two noise arms at **5–9**, replacing the shipped 15.
> `commercial_multiview.py` is not edited here; the constant moves in lane D.

Both noise arms have an interior optimum at the headline stride — Gaussian **9**,
SOMA-77-heavy-tail **5** (p95 11.73° at 5 against 12.00 at 9 and 12.73 at 15; that arm
prefers narrower because after median-matching a p95 draw is ≈40 px, past
`triangulate_point`'s 14 px inlier gate, and an outlier that survives is then *spread over
the window* by the linear tangent-space filter). The Gaussian arm is the primary because
its amplitude is pinned by two independent estimates of the same σ.

**Two independent biases push the measured optimum wider than the truth**, and they point
the same way:

1. **Speed.** Even the fastest playback the fixture supports at usable length reaches only
   **77 %** of our own capture's median thorax speed, and the argmin falls monotonically
   with speed.
2. **Noise colour.** The injected noise is **white**. A temporal smoother removes white
   noise far better than the temporally correlated error a real detector makes, so
   smoothing is flattered here and a wide window looks cheaper than it is.

So the honest statement is an **upper bound**: at the real fixture's motion the optimum is
**at most 9**, and plausibly 5. The shipped 15 is not supported by any arm at or above
0.58× of real thorax speed.

### Does it agree with the MAMMA arm? No on the level, yes on the shape

The docstring's oracle sweep (transcribed in the report, **reports and never selects**):

| window | oracle P1 median | oracle P1 p95 |
|---|---|---|
| none | 5.46 / 4.87 | 17.22 / 17.59 |
| 5 | **5.42** / 4.87 | 17.00 / 17.21 |
| 15 *(chosen)* | 5.46 / 4.89 | 14.14 / 16.30 |
| 21 | 5.87 / 4.96 | 14.12 / 17.19 |
| 31 | 6.52 / 4.88 | **13.78** / 16.39 |

* **Level: they disagree.** MAMMA's arm chose 15; ours says 9, bracket 5–9.
* **The docstring's justification holds for one performer and not the other.** It says the
  sweep "has an interior optimum". Performer 1's p95 (17.59 / 17.21 / **16.30** / 17.19 /
  16.39) does have an interior minimum, and it is at 15 — the stated choice. Performer 0's
  (17.22 / 17.00 / 14.14 / 14.12 / **13.78**) falls monotonically to 31, the widest window
  it tested, still improving. The table does not say which column the choice was read from,
  and no single window is the p95 argmin on both.
* **What that arm cannot see.** It scores *agreement with a smooth fitted-chain reference*
  and has no truth to charge lag or peak attenuation against. Where its p95 improves with a
  wider window, the improvement may be the reference's own smoothness rather than a better
  frame, and nothing in that arm can separate the two. Ours can, because it scores against
  exact truth — which is what makes lag and attenuation cost something.
* **Shape: they agree.** On *both* performers the median prefers a narrower window than the
  p95 (median argmin 5 each; p95 argmin 31 and 15), and ours does the same at the
  nearest-speed playback. Only the shape is comparable — different quantities, different
  data, never one axis.

### What this instrument is blind to

* **Temporally correlated detector error.** The injected noise is white, and correlated
  error is exactly what a temporal smoother cannot remove, so every window here is
  flattered equally. The *shape* of the curve is the claim, not the absolute degrees.
* Calibration, distortion, sync and soft-tissue error — absent by construction.
* Joint-definition error, which dominated Battles 0 and 1 on real footage.
* **Windows past what the takes cover.** At the nearest-speed playbacks the takes are
  18–32 frames, so 31 and 45 are not measured there at all.
* **Lag resolution.** Cross-correlation lag on an 18–32 frame take is coarse; the estimate
  is meaningful at roughly the 1-frame level, and the small *negative* lags at the
  narrowest windows are estimator noise, not anticipation.
* **A declared MAMMA dependency in the geometry.** The rig that carries pixel noise into
  millimetres *is* MAMMA's `ma_cap`, and the subject placement and speed-match target come
  from our own detector's track on MAMMA's example footage. The mitigation is that the
  selection is an **argmin**, not an absolute, and the amplitude is swept 0.5×–2.0×.

---

## Three things in the plan brief that turned out to be wrong

1. **The fixture's joint order.** The brief said `build_synthetic_truth_fixture.py`'s take
   builder gives `[frame, joint, 3]` in our 19-joint `JOINT_INDEX` order. It returns
   `[frame, 77, 3]` in **SOMA-77 index order**; `SOMA77_TO_AUTOANIM` is applied by
   `observations_for`, not by `build_take`. Handled by `to_19()`, not silently.
2. **"A window so wide it freezes the frame" is not constructible.** Savitzky-Golay at
   polyorder 2 fits a parabola and never freezes, and `_thorax_frames` returns the
   *unsmoothed* frame once the window exceeds the take — so the widest cells are "none"
   in disguise. The freeze control is built explicitly as the take's mean frame.
3. **The tracked synthetic clips cannot select a temporal window as they stand.** At
   stride 1 their thorax turns at 0.0–22.7 deg/s against our own capture's 66–73, so the
   p95 curve is monotone to the widest window on offer and selects nothing. They must be
   played faster first, which costs take length and caps the sweep at 21 (stride 3) or 15
   (stride 4).

## Regenerate

    .venv/bin/python tools/compare/provenance.py            # exits 1 while a leak stands
    .venv/bin/python tools/head/thorax_window_sweep.py      # ~12 min, 12 seeds
    .venv/bin/python -m pytest tests/test_provenance_audit.py
