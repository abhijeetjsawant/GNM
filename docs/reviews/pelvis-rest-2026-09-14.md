# D7c — the pelvis on the rig's own rest. **STOPPED at selector S, twice, before any `src/` change.**

**Date** 2026-09-14 · **Branch** `ladder/D7c` · **Worktree** `.claude/worktrees/ladder-D7c`
**Verdict: the step stopped at the pre-registered stop condition in S at the card's own fixture,
and then again at the amended card's FIXTURE CALIBRATION, which is UNREACHABLE by its own
frozen rule. `src/` was not touched at any point, and the instrument-side estimators are frozen
at `8a82ee4`.**
Hygiene PASSED (8 of 8). The instrument PASSED, committed before `src/` could have moved, and
reproduces the pre-card on the shipped build and on all six D3 bodies. S then reached its own
frozen-pitch-follower clause and failed it on **5 of 6 bodies**, which the card names a STOP —
*"if on any body it does not, the fixture has not shown the discrimination the card claims,
that is a STOP, never a pass."*

Nothing below is an override, a moved band, or a re-chosen constant. The selector was run
once at the fixture the card pre-registers; its verdict is the verdict. A sensitivity sweep is
reported beside it so the coordinator and Astra can see what the stop is attributable to, and
it selects nothing.

---

## 0. What was done, in order, and where it stopped

| stage | what | verdict | commit |
|---|---|---|---|
| 1 | hygiene — today's code rebuilds the shipped delivery byte-identically | **PASS**, 8 of 8 | `d146aa3` |
| 2 | instrument first, on the SHIPPED build and the six D3 bodies | **PASS**, reproduces the pre-card | `7d5f3a3` |
| 3 | S, the selector, at the card's own fixture, run BEFORE the src change was chosen | **STOP** | `ee4b71f` |
| 3A | the amended card's FIXTURE CALIBRATION, and the reread it gates | **UNREACHABLE -> STOP** | this commit |
| 4 | the src change | **not started** — `git diff 7e35dd0 -- src/` is EMPTY | — |
| 5 | the delivery and the bands | **not reached** | — |
| 6 | the gate, the extractor, the tests, the report page | **not reached** | — |

---

## 1. Stage 1 — hygiene. PASS.

`tools/compare/d7c_pelvis_rest_delivery.py`, modelled line for line on D9b's, rebuilt the
delivery through the REAL build script into this step's own directory with `work/` COPIED
(never symlinked). With `src/` unchanged:

* all eight delivered files byte-identical to `artifacts/commercial-multiview-soma77`;
* the cached detections byte-identical before and after the build, and to the shipped build's;
* the raw AND the smoothed triangulations byte-identical on both performers — the
  same-denominator baseline a converter-only step is measured against.

`artifacts/compare/d7c-pelvis-rest/delivery-hygiene-build.json`, verdict PASS, 274.9 s.

**A trap worth recording.** `.venv` is shared with the main checkout and `autoanim_gnm` is
installed **editable there**, so `import autoanim_gnm` inside this worktree resolves to
`/Users/abhi_macbook/Projects/apps/AutoAnim/src` unless `PYTHONPATH=$PWD/src` is set. Every
instrument in this step exits rather than run if the resolved module is outside the worktree,
and the resolved path and `PELVIS_FRAME_SOURCE` are printed into each log and recorded in each
report. Without that guard the tripwire and the delivery would have silently measured the main
repo's unchanged source and passed vacuously.

---

## 2. Stage 2 — the instrument, committed before any `src/` change. PASS.

`tools/compare/d7c_pelvis_rest_gate.py`. The oracle block runs the D3 gate's six
exact-skeleton bodies through the REAL converter under a recording projection watcher, one arm
at a time by module substitution; the take block reads a build's own delivered bytes plus the
`converter-inputs/` dump.

### 2.1 The defect, reproduced on exact truth

Every banded number is on the **ABSOLUTE** row. The D3 gate's `retarget_cost.score` subtracts
the leg-root midpoint per frame and is structurally blind to a root move (CLAUDE.md, D9b); it
is reported and no band reads it.

| arm | tilt median (deg) | 3-point residual | `Spine` miss (mm) | `Hips` miss (mm) | torso, ABS (mm) |
|---|---|---|---|---|---|
| `src_default` = C, shipped | **6.8650 – 6.8651** | 84.5 – 107.2 mm | 20.97 – 28.33 | 9.63 – 11.56 | 8.98 – 12.09 |
| `wrong_origin` (control) | 0.0000 – 0.0001 | 80.4 – 96.5 mm | 0.0001 | 0.0002 | 0.00 |
| `D_rig_rest_hipline` | 0.0000 – 0.0001 | **1.5 – 2.4e-7 m** | 0.0005 | 0.0005 | 0.00 |
| `E_rig_rest_kabsch` | 0.0000 – 0.0001 | **1.5 – 1.8e-7 m** | 0.0001 | 0.0002 | 0.00 |
| `frozen_upright` (must-fail) | 7.2166 | 1.0 – 1.2e-2 m | 22.03 – 29.77 | 10.63 – 12.15 | 5.25 – 6.56 |

with the shipped fit's yaw 0.033 – 0.036° and roll 0.005 – 0.035° beside the 6.865° pitch —
the card's figures, frame for frame.

**The wrong-origin control is the reason the residual is banded in metres and not in degrees.**
Taking the rig's rest offsets from `Hips` instead of the leg-root midpoint is an 80 mm
translation; the rig's rest pelvis is symmetric enough to absorb it entirely into a translation
a rotation-only fit never sees, so it reads **0.0001° of tilt**. A tilt band passes it. A
residual between *normalised* frames passes it. Only the **unnormalised three-point positional
residual** sees it, at 80 – 97 mm. This is the card's pre-registered known blindness, realised.

### 2.2 The take, from the shipped build's own bytes

| | performer 0 | performer 1 |
|---|---|---|
| delivered pelvis +Y vs `Spine1 − hip midpoint`, median | **9.379°** | **9.937°** |
| pre-guard lever median | 131.5768 mm | **125.4928 mm** |
| frames off the 0.15 ceiling | **0** | **29** |

performer 1's demoted set: `24 25 28 29 32 33 38–46 65 68 70 78–81 140 141 144 145 147–149` —
**frame for frame the mask the card froze**, on a median frozen from the unchanged PRE-guard
input. `np.median` over an array carrying NaN returns NaN, which would make the ceiling
infinite and the mask empty — a silent pass — so `pelvis_lever_guard` takes its median over an
explicit finite mask.

---

## 3. Stage 3 — S. The stop, and exactly which clause.

`tools/compare/d7c_pelvis_synthetic.py`, at the fixture the card pre-registers:
`d7_pelvis_synthetic.observe`, I7/I8's heavy-tail frame-correlated noise at
`NOISE_SIGMA_PX = 3.20 px`, injected in pixels through the real `triangulate_point` on the
delivery's own four-camera rig, one frozen draw per body, every arm reading the same draw.
`artifacts/compare/d7c-pelvis-rest/selector.json`.

### 3.1 The clauses S reached, in the card's own order

| clause | measured | verdict |
|---|---|---|
| (a) vs (b) on all three metrics, both populations | **(b) is WORSE in all six cells** | **(a) `E_rig_rest_kabsch` wins** |
| the winner strictly better than C-on-SOMA on (i), both populations; better-or-tied on (ii), (iii) | better in all six cells | **PASS** |
| the frozen-pitch follower ≥ 2× the winner's (i) on the bent tercile, **on every body** | **5 of 6 fail** | **FAIL → STOP** |
| the frozen-pitch follower ≥ 2° on the bent tercile, on every body | 6 of 6 pass | PASS |
| world-vertical against the truth's own tilt range | not reached (the rule stops first) | — |
| G1 missing-only | **not reached** | — |
| G2 finite-only | **not reached** | — |

**(a) beat (b) in every cell**, median over the six per-body medians:

| | (b) `D_rig_rest_hipline` | (a) `E_rig_rest_kabsch` | C-on-SOMA |
|---|---|---|---|
| whole take (i) orientation | 9.749° | **9.549°** | 15.970° |
| whole take (ii) step | 5.071° | **4.831°** | 6.491° |
| whole take (iii) root step | 8.660 mm | **7.637 mm** | 9.014 mm |
| bent tercile (i) | 12.332° | **11.217°** | 15.510° |
| bent tercile (ii) | 5.378° | **5.173°** | 7.039° |
| bent tercile (iii) | 9.519 mm | **8.819 mm** | 10.940 mm |

**The follower clause, per body, bent tercile:**

| seed | winner (a), (i) | follower, (i) | ratio | ≥ 2× |
|---|---|---|---|---|
| 20260903 | 7.769° | 15.469° | 1.991 | **no** |
| 20260904 | 10.968° | 21.110° | 1.925 | **no** |
| 20260905 | 11.889° | 17.221° | 1.448 | **no** |
| 20260906 | 12.221° | 19.217° | 1.573 | **no** |
| 20260907 | 6.894° | 18.900° | 2.742 | yes |
| 20260908 | 11.467° | 20.122° | 1.755 | **no** |

### 3.2 What the stop is attributable to, measured

Two rows, computed by the selector itself (`fixture_attribution`), and neither selects
anything.

**The discrimination is structural and, without observation noise, unbounded.** Every arm on
the truth landmarks with no noise at all:

| arm | whole take | bent tercile |
|---|---|---|
| `D_rig_rest_hipline` | **0.0000°** | **0.0000°** |
| `E_rig_rest_kabsch` | **0.0000°** | **0.0001°** |
| frozen-pitch follower | 6.320° | **14.401°** |
| world-vertical | 7.217° | 16.857° |
| C-on-SOMA | 6.865° | 6.865° |

The follower's 14.401° on the bent tercile is the quantity the clause is about: the performer's
true pelvis pitch about the hip line, relative to gravity. Against a winner at 0.0000° the
ratio is unbounded. **The fixture's noise is the whole of the compression**: it raises the
winner to 6.9 – 12.2° and leaves the follower's structural floor where it was.

**The fixture's noise is 1.8 – 3.0× the take's own measured spread.** D7's own report records
the comparison (`artifacts/compare/d7-pelvis-frame/synthetic.json`,
`noise_calibration_vs_the_real_take`), and this step reproduces it on the D3 bodies:

| | synthetic, this fixture | the real take, D7's measured row |
|---|---|---|
| `midhips → Spine1` spread (sd) | **20.07 mm** (median of six) | 6.61 mm / 11.10 mm |
| per-landmark 3D noise, median | 18 – 23 mm on every landmark | — |

**and the comparison is confounded, in the candidate's favour.** The D3 rig is RIGID, so the
synthetic lever's spread is *pure observation noise*, while the take's mixes observation noise
with the performer's real lever variation. Matching the two would therefore UNDER-noise the
fixture. That is one of the reasons this step does not pick a calibration.

### 3.3 Why this agent did not repair the fixture ON ITS OWN AUTHORITY

**Superseded in part by section 3A**: the coordinator and Astra subsequently amended the card to authorise a repair, on a calibration definition frozen by the reviewer before any reread. That is the correct route and it is the one that was taken. What follows is why this agent did not take it unilaterally, and every reason still holds for a repair chosen by the agent after seeing the verdict.


CLAUDE.md carries a fixture-repair rule (D8b: *"a step's fixture can fail a pre-registered
clause for the fixture's own defects … repair the FIXTURE as a fixture parameter with src
byte-identical across the repair, rerun the same clauses"*). It is deliberately not applied
here, for four reasons:

1. **The brief instructs the stop.** "If S stops (split winner, non-discriminating follower,
   G2 not won), write the report, commit, and STOP: do not touch `src/`." The card says the
   step stops *for the coordinator*.
2. **The noise amplitude is pre-registered, not discovered.** The card names
   `d7_pelvis_synthetic.observe` and I7/I8's noise; D7's own report already carried the
   lever-spread discrepancy and D7 shipped on it; Astra reviewed the 2× follower band across
   four rounds against this same fixture. A fixture property the card author knew about when
   the band was set is a band-calibration question, and only the coordinator can re-pin a band.
3. **There is no principled parameter available.** The only measured target is confounded
   (§3.2), and it is confounded in the direction that favours the candidate.
4. **Choosing sigma after seeing the verdict is selecting on a knob.** It is the same failure
   as "a band the solver regularises is a knob setting, not evidence", wearing a different coat.

Instead the selector gained a `--sigma-scale` flag that is **REPORT ONLY**: any value other
than 1.0 refuses to overwrite `selector.json` and writes
`selector-sensitivity-sigma<scale>.json` carrying an explicit `sensitivity_note`. The sweep is
below.

### 3.4 The sensitivity sweep — REPORT, never a selection

| σ scale | lever sd (mm) | (a) vs (b) | winner | follower ratios, bent | bodies failing ≥ 2× | S verdict |
|---|---|---|---|---|---|---|
| **1.00 (the card's)** | **20.07** | (b) worse in 6/6 | `E_rig_rest_kabsch` | 1.99 1.93 1.45 1.57 **2.74** 1.75 | **5** | **STOP** |
| 0.50 | 13.13 | (b) worse in 6/6 | `E_rig_rest_kabsch` | 2.28 2.12 1.88 2.18 2.65 2.08 | 1 | STOP |
| 0.35 | 9.13 | (b) worse in 6/6 | `E_rig_rest_kabsch` | 2.93 2.73 2.47 2.75 3.08 2.85 | 0 | PROCEED |
| 0.25 | 6.57 | (b) worse in 6/6 | `E_rig_rest_kabsch` | 4.03 3.63 3.30 3.62 4.10 3.77 | 0 | PROCEED |

(the take's own measured lever sd is 6.61 / 11.10 mm, so σ ≈ 0.25 – 0.35 is where the fixture
meets the take on that — confounded — comparison.)

**Two things the sweep establishes, and they are the useful part of it:**

* **The (a)/(b) selection is stable at every noise level tested.** `E_rig_rest_kabsch` wins all
  six cells at σ = 1.00, 0.50, 0.35 and 0.25. Whatever the coordinator decides about the
  fixture's amplitude, the mode that would ship does not change with it. That is worth more
  than the stop itself.
* **The follower ratio moves monotonically with the fixture's noise and nothing else.** The
  follower's own error barely moves (15 – 21° at σ = 1.00 against its 14.40° noiseless floor);
  the winner's error is what the noise inflates.

At σ = 0.35 and 0.25 the guard's own experiments were reached, and they are reported here as
**sensitivity only** because their fixture is not the pre-registered one:

* **G1, missing-only.** Unconditional identity holds on **5 of 6** bodies at sigma 0.25 (seed
  20260904, frame 21) and **4 of 6** at sigma 0.35 (additionally seed 20260903 frame 84 and seed
  20260904 frames 20-22 and 103-104) -- the figures Astra recomputed. Recovery error on the
  missing frames 3.7 - 5.7 deg against 3.3 - 4.3 deg elsewhere. **Reported, never banded** --
  G1 carries no superiority claim, and the CLAIM ITSELF is what was wrong: see section 3A.6,
  where it is amended to the array level.
* **G2, finite-only.** Decisive. The MEDIAN-OVER-SIX figures, which are what the amended card
  asks for (the earlier wording here quoted seed-level ranges and is corrected): at sigma 0.35,
  **7.527 deg guarded against 76.641 deg unguarded** on the corrupted frames and **2.707 deg
  against 8.829 deg** on the transition pairs; at sigma 0.25, 5.524 against 76.314 and 2.123
  against 8.241. Both metrics are won on every body at both sigma. The guard's miss rate is
  0.033 on five bodies and **0.10 on seed 20260906**. **This is not a pass**: G2's verdict at
  the pre-registered fixture was never reached, and a G2 run under a repaired fixture is the
  coordinator's to authorise. Full detail in section 3A.7.

---

## 3A. The amended card — the FIXTURE CALIBRATION, and why it is **UNREACHABLE**

After the σ-1.0 STOP the card gained a FIXTURE CALIBRATION amendment (Astra GPT6 rounds 5
and 6). `selector.json` is untouched and the σ-1.0 verdict stands exactly as it fell; this
section is `selector-calibrated.json`. `src/` was still never touched, and the instrument-side
estimators are frozen at `8a82ee4` — `git diff 8a82ee4 -- tools/compare/d7c_pelvis_estimators.py`
is empty. Only the calibration machinery and the amended G1 test were added.

### 3A.1 The target, measured — not typed

The amendment's target is the take's `|Spine1 − hip midpoint|` sd **at S's own processing
stage** (the converter inputs the delivery watcher dumped: gap-filled, hips smoothed, Spine1
unsmoothed, all 150 frames), over the frames the **0.15 guard KEEPS**, `ddof=0`, larger
performer. `take_calibration_target()` measures it from the hygiene build's own bytes and the
instrument refuses to run if it does not reproduce the frozen constant:

| | frames | guard-kept | median | sd all frames, ddof=0 | **sd guard-kept, ddof=0** | sd guard-kept, ddof=1 |
|---|---|---|---|---|---|---|
| performer 0 | 150 | 150 | 131.5768 mm | 5.9944 | **5.9944** | 6.0145 |
| performer 1 | 150 | **121** | 125.4928 mm | 24.8646 | **8.7636** | 8.8000 |

**target = 8.7636 mm.** The `ddof` difference against round 5's 6.014 / 8.800 is the SD
denominator, not rounding. D7's rigidity row (6.61 / 11.10) is a raw-triangulation,
common-valid-mask statistic on 150 / 138 frames — a different stage, and not the matched
target. The synthetic side now applies the **same keep-rule** before `np.std(ddof=0)`;
`fixture_attribution` carries both the all-frames row (which the σ-1.0 report quoted) and the
matched guard-kept row beside it. S's scoring populations are unchanged by any of this.

**Two qualifications carried, neither softened.** The take's sd bounds its observation noise
only under an additive, uncorrelated length-error model, and a guard-kept sd is a
**conditional** spread — selecting by observed length can break that decomposition — so **no
harshness claim is made in either direction**. And length bounds no direction at all.

### 3A.2 The zero-noise baseline, reported first and never subtracted

σ = 0 through the **same** `observe_body` pipeline — the real cameras, the real
`triangulate_point`, the real gap fill and Savitzky–Golay smoothing, the identical seeded
draws. Guard-kept lever sd **1.8131 mm** (median of six). That is the preprocessing floor:
the 19-contract joints are smoothed and Spine1 is not, so the hip midpoint moves under an
unsmoothed Spine1 even with no pixel noise at all. It is **never subtracted** from either
side — the target is the total post-processed observable, and a variance subtraction would
need a covariance model and would be a different calibration.

### 3A.3 The bisection: a σ was found inside tolerance, and the calibration is still unreachable

One pixel-σ for all six bodies, bracket `[0.10, 1.00]`, tolerance 0.05 mm, ≤ 20 evaluations,
the original seeded draws preserved at every evaluation (`heavy_tail_magnitude` draws its
uniform before it multiplies, so the stream is identical at every amplitude).

| σ | median-of-six guard-kept sd |
|---|---|
| 0.100000 | 3.0777 |
| 0.325000 | 8.4760 |
| 0.332031 | 8.6582 |
| **0.335547** | **8.7495**  ← inside tolerance (\|Δ\| = 0.0141 ≤ 0.05) |
| 0.339063 | 8.8410 |
| 0.353125 | 8.8275  ← **the violation: a 0.0135 mm DECREASE as σ rose** |
| 0.381250 | 9.3457 |
| 0.437500 | 10.6170 |
| 0.550000 | 11.9565 |
| 1.000000 | 12.5108 |

* the bracket **contains** the target (3.0777 … 12.5108) — PASS;
* a σ **inside tolerance was found**: **0.335547 → 8.7495 mm** — PASS;
* the statistic is **not monotone across the evaluations** — **FAIL**, and the frozen rule
  makes that an **UNREACHABLE** calibration.

**The verdict is UNREACHABLE. No reread of S was performed at 0.335547.** Reading S at a σ the
frozen rule rejects, and then reporting it, is how a verdict leaks into a rule change; the
amendment decision has to be made on the calibration, not on what the calibration would have
produced.

### 3A.4 The violation, diagnosed to the frame

Not six bodies wobbling — two rejections:

| seed | σ 0.339063 | σ 0.353125 | Δ | keep-mask change |
|---|---|---|---|---|
| 20260903 | 9.0549 | 9.0353 | **−0.0195** | frame **84** newly rejected (150 → 149 kept) |
| 20260907 | 8.6272 | 8.6198 | **−0.0074** | frame **21** newly rejected (150 → 149 kept) |
| 20260904 | 10.9619 | 11.1511 | +0.1892 | unchanged, 150 kept |
| 20260905 | 9.6138 | 10.0083 | +0.3945 | unchanged, 150 kept |
| 20260906 | 7.2723 | 7.5603 | +0.2880 | unchanged, 150 kept |
| 20260908 | 7.9932 | 8.2987 | +0.3055 | unchanged, 150 kept |

At σ 0.339063 the guard rejects **nothing on any body**. At σ 0.353125 it newly rejects exactly
**one frame on each of two bodies**, and each rejection removes that body's largest-lever
frame — so those two bodies' guard-kept sd **falls** although the pixel noise rose. Those two
bodies are precisely the ones sitting at the 3rd and 4th of six, which is what the
median-of-six is made of; the four bodies that rose are all outside those positions. The
median therefore fell by 0.0135 mm while every unchanged body rose by 0.19 – 0.39 mm.

**That is the mechanism the card names, realised exactly: the synthetic keep-mask moves with
σ, so the statistic is taken over a different population at each evaluation and is not a
smooth function of σ.**

### 3A.5 The one question this hands the coordinator

The frozen rule's *wording* is "the statistic is not monotone in σ **across the evaluations**";
its stated *rationale* is "a bisection on it is not well posed". Here the bisection's own
bracket — 0.325 → 0.332031 → 0.335547 → 0.339063 — **is** strictly monotone, and the single
violation (0.0135 mm, 27 % of the tolerance) lies **above both the accepted σ and the target**,
outside that bracket. Whether the boolean applies to **all evaluations** (the wording) or to
**the final bracket** (the rationale) is the one amendment that would change this verdict, and
it is a rule change only the coordinator and Astra can make. This agent applied the wording,
stopped, and is not arguing the point.

### 3A.6 G1, amended — and not yet exercised at a calibrated σ

The claim is now the reviewer's: **identical effective masks and retained samples ⇒
bit-identical INTERPOLATED ARRAYS**, tested on the arrays `np.interp` produces rather than on
the quaternions downstream of them (`interpolated_array`, which re-executes only the recovery
path; `rig_rest_pelvis_frames` is frozen and was not modified to expose it). The old,
overbroad wording — that the two arms' arrays agree whenever the missing pattern is the same —
is wrong for a stated reason: an additional finite rejection changes the mask, and removing
the 29 samples also changes the finite median the guard compares against. Every additional
rejection is now reported with its **lever, the median it is compared against, the 0.15
threshold, the band in millimetres, and the fraction it is off by**, and none is selected away.

**The amended test has NOT been exercised at a calibrated σ** — the calibration is unreachable,
so G1 and G2 were not rerun. What is recorded remains the σ-0.25 / σ-0.35 sensitivity runs, and
their additional rejections are the ones Astra named: σ 0.25, seed 20260904 frame 21; σ 0.35,
seed 20260903 frame 84 and seed 20260904 frames 20–22 and 103–104.

### 3A.7 G2's sensitivity summary, corrected

The σ-1.0 report quoted seed-level ranges. The recorded **median-over-six** figures, which is
what the amendment asks for:

| σ | (i) corrupted frames, guarded | unguarded | (ii) transition pairs, guarded | unguarded | miss rate | both metrics won on every body |
|---|---|---|---|---|---|---|
| 0.35 | **7.527°** | **76.641°** | **2.707°** | **8.829°** | 0.033, **up to 0.10** on seed 20260906 | yes |
| 0.25 | 5.524° | 76.314° | 2.123° | 8.241° | 0.033, up to 0.10 on seed 20260906 | yes |

**Sensitivity only.** G2's verdict at the pre-registered fixture was never reached, and a G2
run under a repaired fixture is the coordinator's to authorise.

### 3A.8 The (a) restatement, prepared from the pre-card's own (a) arm

Recorded here because the card requires it *before* delivery and because it is the substantive
consequence of (a) having won. Every figure is the pre-card's rebuild of **(a)
`E_rig_rest_kabsch`, UNGUARDED** (`precard-take-candidate`, `precard-take.json`), against the
shipped D9b build, on byte-identical landmarks. **These are predictions to revisit, never
bands**, and the guarded (a) rebuild is still owed.

| quantity, performer 0 / 1 | (a) unguarded | (b) unguarded, the card's original text |
|---|---|---|
| pelvis pitch about the hip line, median | **−8.783 / −9.260°** | −8.784 / −9.283° |
| pelvis change, median (p95) | 8.896 (15.504) / 9.376 (25.516)° | 8.868 (15.454) / 9.311 (23.780)° |
| root move, hoist-subtracted | **12.403 / 13.057 mm** | 12.367 / 12.984 mm |
| root move, fore-aft in today's pelvis frame | **−12.214 / −12.839 mm** | −12.217 / −12.906 mm |
| `Spine` origin move | **30.525 / 30.138 mm** | 30.437 / 29.970 mm |
| `Neck` move | **12.434 / 10.816 mm** | 12.764 / 10.429 mm |
| leg-root midpoint on the captured hip midpoint | **0.000 mm** | 0.000 mm |
| **hip residual, median (p95)** | **4.046 (11.859) / 4.580 (31.853) mm** | 1.185 (4.187) / 1.199 (5.906) mm |
| delivered +Y vs `Spine1 − hip midpoint` | **1.400 / 2.018°** | 3.166 / 3.887° |
| hoist p95, D9b → candidate | 12.54 → **13.124** / 8.718 → **7.369** mm | 12.54 → 11.559 / 8.718 → 9.107 |
| contacts, D9b → candidate | (38, 51) → **(36, 36)** / (11, 18) → **(4, 18)** | (38, 51) → (38, 45) / (11, 18) → (6, 18) |

**What this table settles about the card's (b)-conditional text:**

* **"the leg roots on the captured hips" is NOT (b)-specific.** It reads 0.000 mm under (a)
  too, and it must: `_leg_root_offset` places the leg-root midpoint on the captured hip
  midpoint whatever rotation the pelvis frame carries. The card listed it as conditional; it
  is not.
* **"zero transverse hip residual" IS (b)-specific and does not carry.** Under (b) the hip
  line is an exact axis of the frame and the residual left is only the rig-vs-performer WIDTH
  mismatch (p95 4.2 / 5.9 mm). Under (a) the 197 mm spine lever pulls against the hip line
  inside one un-centred SVD, and the residual is **11.9 / 31.9 mm at p95** — Astra's figures,
  reproduced. Under (a) **B2's hip clause is a REPORT**, its same-denominator equality still
  required, and **no hip-residual band may be manufactured from these numbers.**
* **(a) sits closer to the captured `Spine1` direction than (b)** (1.4 / 2.0° against 3.2 /
  3.9°) and further from the captured hip line. That is the same trade S decided, seen on the
  take: (b) spends its freedom on the hip line, (a) spreads it over all three points.
* **Contacts and the hoist move more under (a)** — performer 0 loses 15 contact frames against
  (b)'s 6, and its hoist p95 rises rather than falls. The card already says contacts may move
  and replaces D9b's identical-contacts clause with P; this is the size of it under (a).


---

## 4. What the card predicted that this run contradicts

**The card's hip-line-specific predictions do not carry.** The card states them explicitly as
conditional: *"every hip-line-specific prediction in this card (zero transverse hip residual,
the leg roots on the captured hips) is CONDITIONAL on (b) winning, and if (a) wins the agent
restates them from (a)'s pre-card arm before delivery."*

**(a) won.** So, for whoever resumes this step:

* the take-level pre-registration in the card (pelvis pitch −8.8 / −9.3°, root 12.4 / 13.0 mm,
  −12.2 / −12.9 mm fore-aft, `Spine` origin 30 mm, `Neck` 12 / 10 mm, hoist p95 12.5 → 11.6 /
  8.7 → 9.1, contacts (38, 51) → (38, 45) / (11, 18) → (6, 18)) was measured on the pre-card's
  rebuild of **(b) unguarded** and is NOT the prediction for (a);
* the pre-card's own (a) arm is `artifacts/compare/d7c-pelvis-rest/precard-take-candidate`
  with `precard-take.json` beside it, and the restatement is to be made from there;
* **the "zero transverse hip residual" clause in B2 does not hold for (a)**: (a) is an
  un-centred Kabsch in which the 197 mm spine lever pulls against the hip line, so the hip line
  is *not* an exact axis of the delivered frame. B2's hip clause must be restated as a
  measured residual, not as zero by construction.

That last point is the substantive one: the card's B2 text was written assuming (b), and a
delivery under (a) would have failed a clause that was true only of the mode that lost.

---

## 5. Every clause, predicted / measured / verdict

| clause | predicted | measured | verdict |
|---|---|---|---|
| hygiene, 8 of 8 byte-identical | 8/8 | 8/8, detections and both triangulations identical | **PASS** |
| instrument: 6.865° on every seed | 6.865 | 6.8650 – 6.8651 | **PASS** |
| instrument: yaw / roll beside it | 0.033–0.036 / 0.005–0.035 | 0.0334–0.0358 / 0.0051–0.0352 | **PASS** |
| instrument: `Spine` 21 – 28 mm | 21–28 | 20.97 – 28.33 | **PASS** |
| instrument: `Hips` 10 mm | ~10 | 9.63 – 11.56 | **PASS** |
| instrument: torso 9.0 – 12.1 ABS | 9.0–12.1 | 8.98 – 12.09 | **PASS** |
| instrument: unnormalised residual 85 – 107 mm | 85–107 | 84.5 – 107.2 | **PASS** |
| instrument: take, +Y vs Spine1−hipmid 9.4 / 9.9° | 9.4 / 9.9 | 9.379 / 9.937 | **PASS** |
| instrument: performer 1's 29 demoted frames, median 125.4928 mm | the frozen mask | identical, frame for frame | **PASS** |
| instrument: performer 0's demoted frames | none | 0 | **PASS** |
| the wrong-origin control reads 0.000° and is caught only by the residual | 0.000° / ~84 mm | 0.0001° / 80.4 – 96.5 mm | **PASS** (blindness realised) |
| `frozen_upright` must-fail | fails O1 | 7.2166°, fails the 0.01° band | **PASS** (fails as required) |
| the D9b build itself must-fail | 6.865° on every oracle frame | 6.8650 – 6.8651 | **PASS** (fails as required) |
| the SOMA template through the new path must-fail | 6.865° | demonstrated on the CURRENT path only: `C_soma_template` reads 6.8650 – 6.8651, an execution identical to `src_default` because `src/` never moved. The card's must-fail is the C execution through the REFACTORED `_pelvis_world_frames` — the tripwire's second reading | **not reached** |
| S: (a) vs (b) decided, not split | either | (a) wins 6/6 cells | **PASS** |
| S: the winner beats C-on-SOMA on (i), both populations | strictly better | 9.549 vs 15.970 and 11.217 vs 15.510 | **PASS** |
| **S: the follower ≥ 2× the winner on every body** | **≥ 2× on 6/6** | **1.45 – 2.74, 5 of 6 below 2×** | **FAIL → STOP** |
| S: the follower ≥ 2° on every body | ≥ 2° | 15.47 – 21.11 | PASS |
| REFACTOR TRIPWIRE | — | **not reached** | — |
| O1, O2, O3 on the candidate | — | **not reached** | — |
| P1, P2, P3 | — | **not reached** (the controls are BUILT in the delivery script and unrun) | — |
| B1 … B6 | — | **not reached** | — |
| G1, G2 at the pre-registered fixture | — | **not reached** | — |

| **THE AMENDED CARD'S FIXTURE CALIBRATION** | | | |
| the target reproduces the frozen constant | 8.7636 mm | 5.9944 / **8.7636** measured from the hygiene build's own converter inputs | **PASS** |
| the synthetic statistic applies the same keep-rule, `ddof=0` both sides | matched | `fixture_attribution` carries the guard-kept row beside the all-frames one | **PASS** |
| the zero-noise baseline reported first, never subtracted | reported | **1.8131 mm** through the same `observe_body` pipeline | **PASS** |
| the bracket [0.10, 1.00] contains the target | contains | 3.0777 … 12.5108 mm | **PASS** |
| a σ inside the 0.05 mm tolerance is found in ≤ 20 evaluations | found | **σ 0.335547 → 8.7495 mm**, \|Δ\| = 0.0141, 10 evaluations | **PASS** |
| the ORIGINAL seeded draws preserved exactly | preserved | two independent runs: all 10 (σ, sd, per-body) triples bit-identical | **PASS** |
| **the statistic is monotone in σ across the evaluations** | **monotone** | **one violation, 0.0135 mm, σ 0.339063 → 0.353125** | **FAIL** |
| the calibration | CALIBRATED | **UNREACHABLE** by the frozen rule | **STOP** |
| the reread of all of S at the calibrated σ | all clauses | **not performed** — reading S at a σ the rule rejects would leak the verdict into the rule change | — |
| G1's amended array-level test | exercised at the calibrated σ | **written and committed; not exercised** — only the σ-0.25 / 0.35 sensitivity runs exist | — |
| G2 at the pre-registered or a calibrated fixture | — | **not reached** | — |

**One FAILED prediction, and its attribution:** the frozen-pitch follower's ≥ 2× separation.
Attributed to the **fixture's noise amplitude**, measured at 1.8 – 3.0× the take's own recorded
spread, against a discrimination that is unbounded without noise (winner 0.0000° vs follower
14.401° on the bent tercile). It is **not** attributable to the estimators, to the guard, or to
the instrument: the same instrument reproduces every pre-card figure exactly, and the (a)/(b)
selection it produces is stable at every noise level tested.

**A second FAILED prediction, from the amended card:** the calibration's monotonicity
precondition. Attributed, to the frame, to the **synthetic keep-mask moving with σ** — at
σ 0.339063 the guard rejects nothing on any body, and at σ 0.353125 it newly rejects one frame
on seed 20260903 (frame 84) and one on seed 20260907 (frame 21), each the largest-lever frame
of a body that happens to sit at the 3rd or 4th of six. Those two bodies' guard-kept sd falls
while the four whose masks did not change all rise by 0.19 – 0.39 mm, so the median-of-six dips
0.0135 mm. It is **not** attributable to the draws (two runs are bit-identical), to the target
(it reproduces the frozen constant exactly), or to the bracket (which contains the target and
whose own final sub-bracket, 0.325 → 0.332031 → 0.335547 → 0.339063, is strictly monotone).
See section 3A.5 for the single question it hands the coordinator.

---

## 6. What every instrument here is blind to

* **Nothing in this step resolves the pelvis CONVENTION.** Every figure is measured in the
  rig's own frame against the rig's own rest. That SOMA-77's `Spine1` lies on the rig's
  `Hips → Spine` axis seen from the hip midpoint is an assumption this step does not test and
  cannot: no landmark instrument resolves a constant change of frame
  (`d7_pelvis_rigidity.py:26`), the exact oracle is fed the rig's own `Spine` and therefore
  agrees by construction, and the photographs judge delivered consequences only. It goes to
  lane H's marker session.
* **S's truth MOTION is D9b's delivered pelvis motion**, produced by the very fit being
  replaced. O1's exactness is a congruence and is unaffected; S's noisy arms inherit the
  shipped fit's smoothness as their truth, so S ranks estimators under the rig convention and
  says nothing about how much real pelvic motion a performer has.
* **`observe` uses every positive-depth camera** and does not reproduce the take's A–C-only
  support on performer 1's lying stretch (frames 100–102, 104, 106 — D8c's unresolved stretch,
  which carries no directional truth either way).
* **A length invariant cannot score direction.** The guard's ceiling is a length rule; a
  same-length rotation of the hip line passes it. That is precisely the trade S was built to
  decide, and S decided it — in (a)'s favour, at every noise level.
* **The fixture's lever-spread comparison is confounded** (§3.2) and confounded in the
  candidate's favour.
* **Neither the delivered bytes nor the photographs have been looked at on any candidate.**
  No claim in this document is about a delivered file other than the shipped one.

---

## 7. What is open

1. **THE ONE DECISION THAT UNBLOCKS THIS STEP: does the calibration's monotonicity
   precondition apply to ALL evaluations (its frozen wording) or to the BISECTION'S OWN FINAL
   BRACKET (its stated rationale, "a bisection on it is not well posed")?** The bracket
   0.325 → 0.332031 → 0.335547 → 0.339063 is strictly monotone and contains the target; the
   single violation is 0.0135 mm — 27 % of the tolerance — and sits above both the accepted σ
   and the target. Under the wording the calibration is unreachable and the σ-1.0 STOP stands,
   which is what this step recorded. Under the rationale σ = 0.335547 is accepted and all of S
   is reread at it. **Only the coordinator and Astra can make that amendment**; this agent
   applied the wording, stopped, and does not argue it. §3A.3–3A.5.
2. **If the precondition is amended, the reread is the next action and nothing else is:**
   all of S at σ 0.335547 — both populations, three metrics, ties, the C-on-SOMA comparisons,
   the every-body follower clause (never normalised per body), the world-vertical report, and
   G1 (with its new array-level test) and G2 — all six bodies kept. One command:
   `PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_pelvis_synthetic.py --calibrate`,
   which will reproduce the identical bisection (proved deterministic over two runs) and then
   proceed once the precondition passes.
3. **Even a calibrated reread may fail the 2× follower clause**, and the amended card is
   explicit about what follows: the failure is recorded, D7c stays undelivered, and there is no
   second reduction, no band change, and no shipping on the remaining conjuncts.
4. **The (a) restatement is prepared but not complete.** §3A.8 restates every (b)-conditional
   prediction from the pre-card's own **unguarded** (a) arm; the card also requires them from a
   **GUARDED (a) rebuild** before delivery, which needs the src change and is therefore unrun.
   Two results already settled there: "the leg roots on the captured hips" is **not**
   (b)-specific (0.000 mm under both), and "zero transverse hip residual" **is** — under (a)
   the hip residual reads p95 **11.9 / 31.9 mm** against (b)'s 4.2 / 5.9, so **B2's hip clause
   is a REPORT under (a)** and no hip-residual band may be manufactured from it.
5. **G1's amended claim is written but not exercised at a calibrated σ**, and G2 has never run
   at the pre-registered fixture. The recorded G1 rejections are σ 0.25 seed 20260904 frame 21;
   σ 0.35 seed 20260903 frame 84 and seed 20260904 frames 20–22 and 103–104.
6. **Everything from the refactor tripwire onward is unrun**: the src change itself, O1/O2/O3
   on a candidate, P1/P2/P3 (whose two controls are built and assert their own non-degeneracy
   in the delivery script but have never been executed), B1 – B6, the containment test, the
   extractor, the tests, the report page.
7. **The instrument-debt items the card hands on are untouched**: the four `SOMA77_REST_*`
   constants' move to `tools/compare/`, D7's moved-by-design clauses, the D3 gate's frozen
   references and its translation-aligned gauge.
8. **The confound in the calibration target stands and is not resolved by it.** A guard-kept sd
   is a conditional spread, and length bounds no direction; the amendment says so and makes no
   harshness claim in either direction. What the calibration matches is a scalar length spread,
   not the detector's directional behaviour.

---

## 8. Files

```
tools/compare/d7c_pelvis_rest_delivery.py     the rebuild, the watchers, the two P1 controls
tools/compare/d7c_pelvis_rest_gate.py         the oracle and take instrument
tools/compare/d7c_pelvis_estimators.py        the two rig estimators, the lever guard, the
                                              controls -- FROZEN at 8a82ee4, unchanged since
tools/compare/d7c_pelvis_synthetic.py         S, the calibration, the amended G1

artifacts/compare/d7c-pelvis-rest/
  delivery-hygiene-build.json                 PASS, 8 of 8
  delivery-hygiene/                           + projection-snapshots/, converter-inputs/
  instrument-shipped.json                     the pre-card reproduced
  oracle-shipped/                             per arm per seed, O2's baseline
  selector.json                               S at the card's own fixture. STOP. Untouched by
                                              the amendment, exactly as it fell.
  selector-calibrated.json                    the FIXTURE CALIBRATION. UNREACHABLE. No reread.
  selector-sensitivity-sigma{0.5,0.35,0.25}.json   REPORT ONLY
  logs/01-hygiene.log 02-instrument-shipped.log 03-selector.log
       04-sensitivity-sigma*.log 05-calibrated.log 06-violation-diagnosis.log
```

`git diff 7e35dd0 -- src/` is empty and
`git diff 8a82ee4 -- tools/compare/d7c_pelvis_estimators.py` is empty: no source file and no
estimator was touched at any point in this step.
