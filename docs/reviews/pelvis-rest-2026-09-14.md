# D7c — the pelvis on the rig's own rest

**Date** 2026-09-14 · **Branch** `ladder/D7c` · **Worktree** `.claude/worktrees/ladder-D7c`
**MERGE.** `E_rig_rest_kabsch` ships. **The step stopped twice in S and then resumed on two
reviewer amendments, each frozen before the reading it gates. Both stops stay recorded exactly
as they fell**, in `selector.json` and `selector-calibrated.json`, which are immutable. S at the card's own
fixture STOPPED on the frozen-pitch-follower clause (§3); the amended card's FIXTURE CALIBRATION
was UNREACHABLE under its own frozen monotonicity rule (§3A); Astra round 7 ruled that STOP
correct, amended the admissibility test post hoc in its own wording, and the same frozen
evaluations then recorded **REACHED** (§3B); the reread of all of S at the calibration's exact
σ **PROCEEDS**, and **`E_rig_rest_kabsch` ships** (§3C).
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
| 3A | the amended card's FIXTURE CALIBRATION, under its frozen monotonicity rule | **UNREACHABLE -> STOP** | `210405d` |
| 3B | the SAME frozen evaluations under Astra round 7's amended admissibility rule | **REACHED**, S pending | `2d8bfcc` |
| 3C | all of S reread at the exact calibrated σ = 0.335546875 | **PROCEED** — `E_rig_rest_kabsch` ships | `2878977` |
| 4 | the `src/` change, the tripwire, the containment test, the tests | **PASS** — tripwire 8 of 8 | `dec1354` |
| 5 | the delivery, O1–O3, P1–P3, B1–B6 | **PASS** — every conjunct | `3e41b7b` |
| 6 | the gate JSON, the extractor stub, the report frames, the review | **MERGE** | this commit |

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

**The fixture's noise is above the take's, and the honest ratio is 1.43×, not the 1.8 – 3.0×
this section first stated.** The original comparison used D7's rigidity row (6.61 / 11.10 mm),
which Astra's round 5 established is **not** the matched target: it is a RAW-triangulation,
common-valid-mask statistic on 150 / 138 frames, a different processing stage. On the MATCHED
statistic — S's own stage, the 0.15 guard's kept frames, `ddof=0` on both sides (§3A.1):

| | synthetic, this fixture at σ 1.0 | the take, matched |
|---|---|---|
| `midhips → Spine1` sd, **guard-kept, ddof=0** | **12.5108 mm** (median of six) | **5.9944** (performer 0) / **8.7636** (performer 1) |
| ratio against the larger performer | **1.43×** | — |
| ratio against the smaller performer | 2.09× | — |
| — the unmatched figures this section first quoted — | 20.07 mm, all frames | 6.61 / 11.10, D7's raw row |
| per-landmark 3D noise, median | 18 – 23 mm on every landmark | — |

**The attribution of the follower failure is unchanged by the correction** — the discrimination
is unbounded without noise and the fixture's noise is what compresses it — but the fixture is
**less inflated than this section first said**, and the sentence is corrected rather than left
standing.

**The confound, in the direction Astra corrected.** The D3 rig is RIGID, so the synthetic
lever's spread is *pure observation noise*, while the take's mixes observation noise with the
performer's real lever variation. Under an additive, uncorrelated length-error model the take's
variance is noise² + true-variation² and the rig's is noise² alone, so matching the two puts
**more** noise in the fixture than the detector has, not less — **the reverse of what this
section originally argued.** And a guard-kept sd is a **conditional** spread: selecting by
observed length can break that decomposition, so **no harshness claim is made in either
direction**, and length bounds no direction at all.

`selector.json`'s own `fixture_attribution.confound` string carries the pre-amendment wording
and is left exactly as it fell, like the rest of that file; it is superseded by §3A.1 and by
the corrected string now in `d7c_pelvis_synthetic.py`.

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

(this table's σ values were chosen before the calibration existed. On the MATCHED statistic
the take's target is 8.7636 mm and the fixture meets it at σ = 0.335547 — §3A.3 — so the σ 0.35
row is very close to the calibrated fixture and the σ 0.25 row is well below it.)

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

## 3B. Astra round 7 — the admissibility rule amended, and the same evaluations **REACHED**

The frozen rule's precondition was global monotonicity, and this record failed it on one
0.0135 mm decrease. Astra round 7 ruled that STOP **correct under the wording it was given**,
kept it recorded, and amended the ADMISSIBILITY test — post hoc, in the reviewer's own words,
reproduced verbatim in `AMENDED_RULE_TEXT` and in the artifact. Nothing was re-observed; the
**same frozen evaluations** were re-assessed.

**Both predecessors are immutable and were not rewritten.** `selector.json` still records the
σ-1.0 STOP; `selector-calibrated.json` still records `status: UNREACHABLE` and
`monotone_across_the_evaluations: false`, and `selector-calibrated-amended.json` carries its
sha256 and states that UNREACHABLE there records the failure of **that** admissibility rule,
not proof that no numerical match exists.

| check | rule | measured | verdict |
|---|---|---|---|
| A | no earlier evaluated statistic exceeds **any** later one by > τ (any pair, not adjacent only — several small decreases could conceal a larger total) | one decrease, **0.0135 mm** (σ 0.339063 → 0.353125) against τ = 0.05 | **PASS** |
| B | the nonzero signs of (statistic − target) change at most once | **1 change**, one sampled crossing: 8.7495 at σ 0.335547 → 8.8410 at 0.339063 | **PASS** |
| C | the unchanged stopping rule finds a value within τ | band [8.7136, 8.8136]; **exactly one** evaluation inside it, 8.7495 mm, residual −0.0141 | **PASS** |

**Verdict: calibration REACHED. S PENDING, not PROCEED.** REACHED is an **observed tolerance
match** at the sampled σ — never global monotonicity, never uniqueness between evaluations,
since even a strictly increasing continuous statistic normally has an *interval* of σ inside a
nonzero tolerance.

**The rounding defect Astra found, fixed and disclosed.** `calibrate()` returned
`round(accepted, 6)` and `main()` passed *that* to S, so S would have been read at 0.335547
while the calibration was evaluated at **0.335546875** — a different fixture from the one
measured. The exact value is now carried and is what S receives. **Separately disclosed and
NOT changed:** the per-body sds are rounded to four places *before* their median is taken;
silently changing that would change the statistic the bisection converged on.

**The exact σ sequence is recovered by replay, not re-derived.** The stopping rule depends only
on the recorded statistics, so replaying it from the frozen bracket reproduces the exact Python
floats — 0.1, 1.0, 0.55, 0.325, 0.4375, 0.38125, 0.353125, 0.33906250000000004, 0.33203125,
**0.335546875** — and every lookup succeeding is itself the proof that the replay *is* the
recorded run.

**The scope of the match, recorded narrowly.** A **conditional length spread** and nothing
else: not directional noise, not detector realism, not camera support, not bias or correlation
structure. Length bounds no direction. The matched fixture ratio at σ 1.0 is **1.4276×**
(Astra's own figure); the zero-noise baseline is **1.8131 mm** and is never subtracted. The
calibration's keep-mask restricts nothing in S's scoring populations.

**The amendment is POST HOC and is recorded as such.** The σ is target-determined — the matched
target and the frozen bisection reach it without consulting S, and the earlier PROCEED at
σ 0.35 enters that arithmetic nowhere — but the coordinator knew that sensitivity result when
proposing the amendment, so agent blindness during the bisection does not make the protocol
independent of earlier outcomes. Every unchanged S clause was accepted in advance even if it
stopped the step again.

---

## 3C. The reread at σ = 0.335546875 — **PROCEED**, and `E_rig_rest_kabsch` ships

All of S, at the calibration's **exact** accepted σ. Median of the six per-body medians:

| arm | whole (i) | (ii) | (iii) | bent (i) | (ii) | (iii) |
|---|---|---|---|---|---|---|
| **(a) `E_rig_rest_kabsch` guarded** | **5.281°** | **2.335°** | **4.356 mm** | **5.460°** | **2.370°** | **4.580 mm** |
| (b) `D_rig_rest_hipline` guarded | 5.568 | 2.655 | 5.018 | 6.078 | 2.793 | 5.622 |
| C-on-SOMA | 9.456 | 3.305 | 5.380 | 8.717 | 3.685 | 5.792 |
| world-vertical | 8.877 | 2.288 | 5.346 | 17.726 | 2.296 | 5.495 |
| thorax-as-pelvis | 15.836 | 2.282 | 4.979 | 37.369 | 2.336 | 5.025 |
| frozen-pitch follower | 9.984 | 2.643 | 5.580 | 15.616 | 2.829 | 5.853 |
| (b) unguarded (ablation) | 5.609 | 2.655 | 5.018 | 6.078 | 2.793 | 5.622 |

* **(a) vs (b):** (b) is worse in **all six cells**. (a) ships — **the same ranking it held at
  σ 1.00, 0.50, 0.35 and 0.25**, so the shipping selection is stable across every fixture this
  step has tested and is not a product of the calibration.
* **the winner beats C-on-SOMA in all six cells** — the constant it removes.
* **THE CLAUSE THAT STOPPED THE STEP NOW PASSES ON EVERY BODY.** Follower ratios **2.559,
  2.831, 2.825, 2.946, 3.048, 3.198**, and 14.61 – 16.13° against the 2° floor. At σ 1.0 five
  of six read below 2×. **The follower's own error barely moved** (15.47 – 21.11 → 14.61 –
  16.13, against its 14.401° noiseless floor); **what moved is the winner**, 6.89 – 12.22 →
  4.80 – 5.71. That is exactly the attribution §3.2 gave for the σ-1.0 failure, confirmed by
  the fixture that repairs it.
* **world-vertical** reads 17.73° against the truth's own bent tilt of 53.73° — not within 2°,
  so the stated limitation does not apply on this fixture.
* **G1, the amended array-level claim, holds on every body.** Unconditional identity does not,
  and is not claimed: seed 20260904 rejects five additional finite frames, each reported with
  its lever, median and threshold. Frames 20, 21, 22 read 151.91 / 134.70 / 147.72 mm against a
  180.02 mm median whose band is [153.02, 207.03]; frames 103 and 104 read 152.90 and 153.01
  against that same 153.02 lower bound — off by 0.15067 and 0.15006 against the 0.15 ceiling,
  a millimetre inside a boundary they cross by six parts in ten thousand. Ordinary noisy
  samples crossing a stated threshold, not an implementation error.
* **G2, where the stop lives, wins BOTH metrics on EVERY body and on the median of six:**
  **7.266° guarded against 76.673° unguarded** on the corrupted frames, **2.590° against
  8.179°** on the transition pairs. Miss rate 0.033 on five bodies, **0.10** on seed 20260906.

The fixture's own guard-kept lever sd at this σ reads **8.75 mm** against the take's 8.7636 —
the match the calibration was selected on, confirmed inside the run that uses it.

**One thing S's main arms cannot show, stated here because the table invites the wrong
reading.** (b) guarded and (b) unguarded are **tied to four places on the bent tercile** —
6.078 / 2.793 / 5.622 both — and identical on the whole take on four of six bodies. The reason
is in the guard's own mask: on the *clean* observation at this σ the guard rejects **nothing on
five of six bodies**, and four frames (20, 21, 22, 24) on seed 20260904 alone. So **the guard's
cost in S's main arms is close to zero by construction on this fixture, and its win there
proves nothing about it** — a guard that cannot lose cannot be said to have won. The guard is
tested in **G2**, on a corruption built for it, and there it wins both metrics on every body by
10.6x on the corrupted frames (7.266 against 76.673 deg) and **3.16x** on the transition pairs (2.590 against 8.179) -- the aggregate STEP improvement is 3.16x, not an order of magnitude, and the strict per-body wins are what stand. That separation, not the main table, is the
evidence for shipping it.

---

## 4A. Stage 4 — the `src/` change, and the tripwire's two readings

`_pelvis_world_frames` gains a `rest` parameter (the call site passes the converter's own
dict, exactly as `_leg_root_offset` and `_joint_origin` already read it), the two rig-rest
modes, and the pelvis lever guard. **`PELVIS_FRAME_SOURCE = "E_rig_rest_kabsch"`.** No constant
enters either rig mode and neither reads SOMA-77's `root` landmark.

### 4A.1 The refactor tripwire: ONE execution, TWO references, TWO verdicts

| reference | result | verdict |
|---|---|---|
| against **D9b**, with the mode held at `C_kabsch_pelvis` | all **8 of 8** delivered files byte-identical to `artifacts/commercial-multiview-soma77`, 274.0 s; the mode-C report reads `lever_guard.applied: false` | **PASS** |
| against **exact rig truth**, the same six-body C execution | **6.8650 – 6.8651°** on every seed | **PASS as a must-fail** |

Never counted as two demonstrations. The guard's scope is what makes the first true: A, B and
C read the spine array untouched.

### 4A.2 O1, O2, O3 — the shipping mode, through the src path

| clause | band | measured (worst of six seeds) |
|---|---|---|
| O1 pelvis vs truth | ≤ 0.01° (from 6.865) | **0.0001°** |
| O1 `Spine` origin, hoist-subtracted | ≤ 0.01 mm (from 21–28) | **0.0001 mm** |
| O1 `Hips` origin, hoist-subtracted | ≤ 0.01 mm (from 10) | **0.0002 mm** |
| O1 torso, ABSOLUTE, unhoisted frames | 0.00 (from 8.98–12.09) | **0.00** |
| O1 unnormalised 3-point residual | ≤ 1e-6 m | **1.53 – 1.83e-7 m** |
| O2 legs, feet, toes vs the shipped FK | ≤ 0.1 mm | **0.054 – 0.077 mm** |
| O2 contacts on the oracle bodies | identical | identical on all six |
| O2 hoist change | ≤ 0.05 mm | **0.008 – 0.032 mm** |
| O3 the D3 gate's ALIGNED gauge, arms | REPORT | **1.32–2.72 → 0.07–0.60** |

**A reporting defect found and fixed inside this stage:** `arm_geometry` assumed `src_default`
meant the SOMA template. Once `src/` shipped a rig mode that was false, and the shipping arm's
residual was being scored under geometry it does not use — 0.10 m instead of 1.7e-7. It now
resolves the effective mode from `PELVIS_FRAME_SOURCE`. O1's residual clause would have failed
for a reporting reason.

### 4A.3 Bit-parity with what the selector evaluated — 12 of 12

On S's own frozen draws at σ = 0.335546875, the shipped branch and
`tools/compare/d7c_pelvis_estimators.py` agree **bit for bit** on the quaternions and on the
demoted-frame list, for six bodies × two modes. Without it the selector could choose one
estimator and production ship another — a defect **no band in this step could see**, because
every band scores the delivery. `git diff 8a82ee4 -- tools/compare/d7c_pelvis_estimators.py`
is empty: the estimator was frozen through the calibration, the reread **and** this change, and
the src branch was written to match it.

### 4A.4 Containment, proved with a positive control

`tests/test_pelvis_rest.py` DELETES the four `SOMA77_REST_*` constants and rebuilds: both rig
modes bit-identical on the resolved path **and on the missing-data path**, while mode C
**raises**. Without that second half the test cannot tell an unused constant from a constant on
a path it never took.

### 4A.5 The four superseded pins, in two classes

`tests/test_pelvis_frame.py` is D7's record and is not edited (the D9b precedent). Four of its
tests fail:

* **one is MOVED BY DESIGN and the card pre-registered it** — `round_trips_the_pelvis_frame`
  poses SOMASKEL77, whose truth pelvis **is** the convention D7c removes, and reads **7.568°**.
  Re-pinned twice in the new file: the same exactness against the rig's own rest (< 0.01° for
  both modes), and the SOMA-posed reading held as a NUMBER so a future change that moves it for
  a different reason is visible;
* **three are SIGNATURE pins** — they call `_pelvis_world_frames` with no `rest` and a rig mode
  raises rather than inventing one. Behaviour unchanged; re-pinned with the rest supplied.

`pytest tests/test_pelvis_frame.py tests/test_pelvis_rest.py tests/test_arm_origin.py
tests/test_hoist_reaim.py -q` → **4 failed, 47 passed**.

---

## 5A. Stage 5 — the delivery and the bands

The delivery rebuilt through the real build script, `work/` copied, **both landmark arrays
byte-identical** on both performers (the same-denominator baseline a converter-only change
must not move). The run-report records `mode: E_rig_rest_kabsch` and the guard's demoted
frames: **0 on performer 0, and on performer 1 the frozen 29-frame mask, frame for frame**,
with resolved fraction 0.807 against the 0.5 fallback.

### 5A.1 P — three contracts, kept apart

| contract | result |
|---|---|
| **P1** on the take, both performers | **PASS**, no failing channel. The delivered track is AUTHENTICATED against the GLB's own `body_track_sha256` first |
| **P1** on **every oracle body** | **PASS**, 6 of 6 |
| **P2** anchor lock, GLB's own arrays, keyed samples | **PASS**; worst travel **4.5e-7 / 2.9e-7 m** against the 1e-5 m band, 18 and 4 runs |
| **P3** travel on the frozen union | REPORT, 51 intervals |

**P1's control 1 cannot be built, and that is a stronger result than P1 catching it.** The
build asserts the projection really did change a foot local (it refuses to run as a no-op),
restores the pre-projection rotations — and `BodyTrack.__post_init__` runs
`validate_body_track`, which raises **`left foot contact moved 0.00884243 m (limit
0.00001000 m)`** before a single file is written. The mutation cannot reach a delivered
artifact at all. The same thing D9b found of its lock-without-correction degenerate.

So P1's *detection* of that mutation is exercised directly on the delivered bytes
(`d7c_p1_controls.py`): applied offline, P1 fails on `local::LeftFoot` and `local::RightFoot`
on both performers, and control 2's mask-clearing fails on `foot_contacts` on both — while the
unmutated delivery passes the identical comparison.

**AN INSTRUMENT DEFECT FOUND BY ITS OWN CONTROL, and fixed.** The first built
`control-clear-contacts` read P1 **PASS** — because the watcher saved the snapshot *after* the
control's mutation, so the control cleared the mask on the delivered track **and** on the
snapshot and P1 compared two copies of the same mutation. A control its own instrument cannot
see is worse than no control. The snapshot is now taken from the function's own return
**before** any mutation, and the control was rebuilt. The **delivery's** P1 result is unaffected,
and that is asserted as a measurement rather than as an argument: the delivery's snapshots
were written by the OLD watcher, and `d7c_p1_controls.py` compares the delivered track against
those very snapshots and reads **PASS on both performers**, while detecting both mutations
applied to the same bytes. (In `shipped` mode there is no mutation for the ordering to
matter to — but the reading is what settles it, not the argument.)

### 5A.2 B1 the photographs — PASS on all eight cells, and a rise nobody predicted

| performer | part | cut | difference | ci95 | verdict |
|---|---|---|---|---|---|
| 0 | arms | whole take | −0.00019 | [−0.00138, 0.00115] | PASS |
| 0 | arms | bent tercile | +0.00079 | [−0.00094, 0.00167] | PASS |
| 0 | torso+legs | whole take | **+0.00715** | **[0.00486, 0.01162]** | PASS |
| 0 | torso+legs | bent tercile | **+0.00792** | **[0.00557, 0.01331]** | PASS |
| 1 | arms | whole take | −0.00068 | [−0.00125, 0.00142] | PASS |
| 1 | arms | bent tercile | −0.00081 | [−0.00213, 0.00125] | PASS |
| 1 | torso+legs | whole take | **+0.00303** | **[0.00066, 0.00752]** | PASS |
| 1 | torso+legs | bent tercile | +0.00192 | [−0.00066, 0.00360] | PASS |

The clause is `ci95[1] >= 0` — **worsening not established**, which does *not* establish
non-worsening. It is met on all eight cells. But three cells show the torso **rising with the
interval clear of zero**, and **improvement was not predicted**: the card's reasoning is that
a constant change of pelvis frame rotates the trunk about the hip line and a mesh can rotate
inside its own outline. A rise therefore needs its own explanation and does not have one here.
Two candidates, neither settled by this instrument: the root moves 12.4 / 13.1 mm
hoist-subtracted, which translates every skinned vertex; or the trunk's new tilt happens to
sit better inside the outline on this footage. And the card's reasoning for "not predicted"
was too strong **on its own terms**: D7's world-vertical control *lost* 0.218 IoU on this
instrument by rotating the pelvis to vertical — a ROTATION, not a translation — so these masks
demonstrably *can* see a pelvis rotation of this order. A ~9° rotation moving them is
therefore not surprising; what is unexplained is the *direction*. **It is reported as
unexplained and no credit is taken for it.**
The MAMMA mesh oracle agrees with the committed unsplit run to 0.0.

### 5A.3 B2, and the same denominator

`delivered_vs_capture.py --reference smoothed`: **same denominator TRUE**. The landmarks are
byte-identical, so the change did no more than refit the pelvis.

### 5A.4 The (a) restatement, from the GUARDED delivery

The card requires every (b)-conditional prediction restated from the pre-card's (a) arm **and
then from a guarded (a) rebuild**. Both columns, side by side — **predictions to revisit,
never bands**:

| performer 0 / 1 | pre-card, (a) UNGUARDED | **the delivery, (a) GUARDED** |
|---|---|---|
| pelvis pitch about the hip line, median | −8.783 / −9.260° | **−8.783 / −9.219°** |
| pelvis change, median | 8.896 / 9.376° | **8.896 / 9.376°** |
| root move, hoist-subtracted, median | 12.403 / 13.057 mm | **12.403 / 13.057 mm** |
| root move, fore-aft | −12.214 / −12.839 mm | **−12.214 / −12.810 mm** |
| `Spine` origin move | 30.525 / 30.138 mm | **30.525 / 30.138 mm** |
| `Neck` move | 12.434 / 10.816 mm | **12.434 / 9.735 mm** |
| leg-root midpoint on the captured hip midpoint | 0.000 mm | **0.0002 mm max** |
| **hip residual, full positional p95** | **11.859 / 31.853 mm** | **12.069 / 14.897 mm** |
| hoist p95, D9b → candidate | 12.54 → 13.124 / 8.718 → 7.369 | **12.54 → 13.124 / 8.718 → 8.094** |
| contacts, D9b → candidate | (38,51) → (36,36) / (11,18) → (4,18) | **(38,51) → (36,36) / (11,18) → (5,18)** |
| pelvis step p95 | — | **14.08 / 13.33°** |
| frames over 800°/s | — | **0 / 1** |

**What this settles, and what it does not:**

* **"the leg roots on the captured hips" was never (b)-specific.** It reads 0.0002 mm under
  (a) too, and it must: `_leg_root_offset` places the leg-root midpoint on the captured hip
  midpoint whatever rotation the pelvis carries. The card listed it as conditional in error.
* **"zero transverse hip residual" IS (b)-specific and does not carry.** Under (a) the hip
  line is not an exact axis of the frame, and the hip residual is a **REPORT**: angular
  1.68 / 1.85° median, transverse 3.10 / 3.52 mm median, full positional p95 **12.1 / 14.9 mm**
  against D9b's 6.4 / 12.0. **No hip-residual band is manufactured from these numbers**, and
  B2's same-denominator equality remains required and passes.
* **The guard is worth its place on the take, and the figure is performer 1's hip residual:**
  the pre-card's unguarded (a) read **p95 31.9 mm** and the guarded delivery reads **14.9 mm**.
  The 29 demoted frames are where that difference lives.
* **One frame over 800°/s on performer 1**, against unguarded (b)'s 2 and guarded (b)'s 0 in
  the card. It is a REPORT quantity by the card's own words — nothing in the merge predicate
  scores it — and it is stated rather than smoothed away.

### 5A.7 The B1 attribution — the two performers rise for opposite reasons

A third build was rendered through the identical pixel path, masks and frozen draws: the
candidate's LOCAL rotations and REST with **D9b's per-frame root translation**, so the
translation's share and the articulation's share separate. (Its contact mask is cleared,
because `validate_body_track` refuses a track whose asserted contacts do not hold once the
root is swapped — the same refusal that made P1's first control unbuildable. It is **not** a
delivery.)

| | both effects | = articulation | + root translation |
|---|---|---|---|
| performer 0, torso, whole take | +0.00715 | **+0.01191** | −0.00476 |
| performer 0, torso, bent tercile | +0.00792 | **+0.00747** | +0.00045 |
| performer 1, torso, whole take | +0.00303 | −0.00249 | **+0.00552** |
| performer 1, torso, bent tercile | +0.00192 | −0.00205 | **+0.00397** |
| performer 0, arms, whole take | −0.00019 | −0.00693 | +0.00674 |
| performer 1, arms, whole take | −0.00068 | −0.00721 | +0.00654 |

**Performer 0's torso rise is the articulation** — the new trunk tilt genuinely fits that
outline better. **Performer 1's is the rigid root shift, and its articulation slightly hurts**,
so that rise is not evidence about the pelvis. And **the arm cells were hiding two effects of
opposite sign**: ~0 in B1, in fact −0.005…−0.009 of articulation cancelling +0.006…+0.008 of
root. DIAGNOSTIC ONLY — it cannot change B1's verdict, and an attribution is not a
justification.

### 5A.5 B5 and B6 — the delivered bytes

`tools/head/head_gate.py` rerun: candidate 6.38 / 15.37 / 16.08 / 4.73 **PASS**, the two
controls FAIL as they must, the gated arms PASS with 19 of 150 frames flagged (12.7 %, under
the 25 % ceiling). It scores the INPUT solve and **cannot prove the exporter preserved it**,
which is why `d7c_delivered_bytes.py` reads the GLB:

* **the delivered `Head` WORLD rotation is IDENTICAL between the two builds** — 105.0914° on
  performer 0 and 83.5315° on performer 1, to four decimals. The converter places the whole
  head-on-torso rotation on `Head` as an ABSOLUTE target, so the chain compensates for the
  pelvis and the exporter carried it through. The input gate could not have shown this.
* LINEAR samplers, 150 frames, 4.9667 s, one translation channel and 55 rotation channels
  sharing one time array; quaternion norms 1 ± 4e-8; **zero negative adjacent dots** (the
  hemisphere walk holds through the pelvis change); full rotation increments median 0.029°.
* **track → GLB positional closure max 0.0005 mm** — the float32 floor.
* between-key playback, two populations kept apart because they mean different things:
  **inside** a contact run the chord is 0.0001 mm median / 0.0003 max (both keys planted, so
  this is the sampler's own error); **at a run boundary**, where the next key is already
  moving, 5.2 mm median / 15.5 max — playback, not a lock failure, and B6's report rather
  than P2's clause.
* **the three invariants hold bit-identically on both performers**: `Root`, both eyes, and
  the finger proximals. Everything else below `Root` moved, which the card states up front.

**Still owed and handed to D6:** the mesh-deformation reading on the pelvis / hip / thigh
region (inverted or collapsed triangles, edge-length and area change against the rest). IoU
can rise while the skin tears, and §5A.2's three rising torso cells are exactly where that
question bites.

### 5A.6 B3, and what the two hoist recoveries say

Hoist p95 12.54 → 13.12 mm on performer 0 and 8.72 → 8.09 on performer 1; the lowest delivered
foot 16.66 → 16.93 and 12.96 → 12.25 mm; contact runs listed per side. The hoist is recovered
both by the converter's own root line and by D9's arm fit, and the two agree to 0.0001 mm at
the median.


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

## 5. Every clause: predicted / measured / verdict

**This is the CURRENT table**, after Astra's merge review at `9dda9ac` (NO MERGE) and the work
that answered it. It carries both recorded STOPs, both post-hoc amendments, and the corrected
numbers. The delivery figures are in §5A; the machine-readable version is
`artifacts/compare/d7c-pelvis-rest/gate.json`, which also records, for every passing clause,
whether flipping it to FAIL turns the merge rule.

| clause | predicted | measured | verdict |
|---|---|---|---|
| hygiene: today's code rebuilds the shipped delivery | 8 of 8 | 8 of 8 | **PASS** |
| **tripwire (i)** mode C held reproduces D9b | 8 of 8 | 8 of 8, 274.0 s, `lever_guard.applied: false` | **PASS** |
| **tripwire (ii)** the SAME execution vs exact truth | 6.865° | 6.8650 – 6.8651° | **PASS** (as a must-fail) |
| O1 pelvis vs truth | ≤ 0.01° (from 6.865) | max **0.0001°** | **PASS** |
| O1 `Spine` origin, hoist-subtracted | ≤ 0.01 mm (from 21–28) | max **0.0001 mm** | **PASS** |
| O1 `Hips` origin, hoist-subtracted | ≤ 0.01 mm (from 10) | max **0.0002 mm** | **PASS** |
| O1 torso, ABSOLUTE, unhoisted frames | 0.00 (from 8.98–12.09) | **0.00** | **PASS** |
| O1 unnormalised 3-point residual | ≤ 1e-6 m | **1.53 – 1.83e-7 m** | **PASS** |
| must-fail: wrong-origin template | 0.000° tilt, ~84 mm residual | 0.0001°, 80.4 – 96.5 mm | **PASS** (blindness realised) |
| must-fail: pelvis frozen upright | fails O1 | 7.2166° | **PASS** |
| O2 legs, feet, toes vs the shipped FK | ≤ 0.1 mm | **0.054 – 0.077 mm** | **PASS** |
| O2 contacts on the oracle bodies | identical | identical, 6 of 6 | **PASS** |
| O2 hoist change | ≤ 0.05 mm | **0.008 – 0.032 mm** | **PASS** |
| O3 the D3 gate's ALIGNED gauge, arms | REPORT | 1.32–2.72 → **0.07–0.60** | REPORT |
| **S at the card's own fixture (σ 1.0): follower ≥ 2× every body** | ≥ 2× on 6/6 | **1.45 – 2.74×; 5 of 6 below** | **FAIL — a recorded STOP** |
| **the calibration under its frozen monotonicity precondition** | monotone | **one 0.0135 mm decrease** | **FAIL — a recorded STOP** |
| the same frozen evaluations under round 7's amended rule | REACHED | A 0.0135 ≤ 0.05, B 1 sign change, C σ = 0.335546875 | **PASS** (POST HOC) |
| S reread: (a) vs (b), 3 metrics × 2 populations | decided, not split | (b) worse in **6 of 6**; ships `E_rig_rest_kabsch` | **PASS** |
| S reread: winner vs C-on-SOMA on (i) | strictly better, both | 5.281 vs 9.456; 5.460 vs 8.717 | **PASS** |
| S reread: follower ≥ 2× **and** ≥ 2°, every body | 6 of 6 | **2.56 – 3.20×**, 14.6 – 16.1° | **PASS** |
| S reread: G1's AMENDED array-level claim | holds every body | holds 6/6; unconditional identity 5/6, rejections reported with lever/median/threshold | **PASS** |
| S reread: G2 both metrics, every body | both, 6 of 6 | (i) **7.266 vs 76.673°**, (ii) **2.590 vs 8.179°**; miss rate ≤ 0.10 | **PASS** |
| the world-vertical control vs the truth's own tilt | report; limitation if within 2° | **17.726° vs the truth PELVIS's 16.823°** | **LIMITATION APPLIES, stated** |
| the delivery: BOTH landmark arrays byte-identical | identical | identical, both performers | **PASS** |
| the run-report records the mode and the demoted frames | E + 0 and 29 | `E_rig_rest_kabsch`; 0 and the frozen 29-frame mask | REPORT |
| **P1** on the take, both performers | PASS | PASS, no failing channel, GLB-authenticated | **PASS** |
| **P1** on every oracle body | PASS | PASS, 6 of 6 | **PASS** |
| **P2** on the take | ≤ 1e-5 m | **4.5e-7 / 2.9e-7 m**, 18 and 4 runs | **PASS** |
| **P2** on every oracle body, from each exported GLB | ≤ 1e-5 m | **worst 4.86e-7 m**, 10–20 runs per seed | **PASS** |
| P1 control 1 (foot locals overwritten) | must FAIL P1 | **cannot be BUILT** — `validate_body_track` refuses it; P1 detects it offline on both performers | **PASS** |
| P1 control 2 (mask cleared) | must FAIL P1 | FAILs P1 on `root_translation_m` and `foot_contacts`, both performers | **PASS** |
| P3 travel on the frozen union | REPORT | 51 intervals | REPORT |
| B1 photographs, 8 cells, `ci95[1] ≥ 0` | worsening not established | met on all 8 | **PASS** |
| B1 the MAMMA mesh oracle | bit-identical | 0.0 | **PASS** |
| B2 `delivered_vs_capture --reference smoothed` | same denominator | TRUE | **PASS** |
| B1 attribution (diagnostic) | — | performer 0's rise is the ARTICULATION (+0.0119 vs −0.0048); performer 1's is the ROOT (−0.0025 vs +0.0055) | REPORT |
| B3 the hoist and the contacts | REPORT | hoist p95 12.54→13.12 / 8.72→8.09 mm; contacts (38,51)→(36,36) / (11,18)→(5,18) | REPORT |
| B4 the pelvis and root motion | REPORT | pitch −8.783 / −9.219°; root 12.40 / 13.06 mm; step p95 14.08 / 13.33°; **0 / 1** frame over 800°/s | REPORT |
| B5 the head gate rerun | REPORT | candidate PASS, both controls FAIL; flagged **19/150 performer 0, 23/150 performer 1** | REPORT |
| B5b the delivered `Head` WORLD rotation, from the GLB | REPORT | between-build difference **4e-6° median, 1.3e-5 max — NOT zero** | REPORT |
| B6 sampler times, channels, quaternions | REPORT | LINEAR, 150 frames, 4.9667 s, 1+55 channels, norms 1±4e-8, **zero** negative adjacent dots; normalised increment median **0.0°** | REPORT |
| B6 track→GLB **positional** closure | REPORT | max **0.0005 mm** | REPORT |
| B6 track→GLB rotational comparison | REPORT | 32.0° median — **NOT a closure**: the GLB channel is the rig local composed with the node's rest rotation | REPORT |
| B6 hierarchy and bone lengths vs the sized skeleton | REPORT | hierarchy matches joint for joint; bone-length error **0.0 mm** on all 54 | REPORT |
| B6 inverse binds / mesh deformation | REPORT | **NOT ESTABLISHED** — skin-at-node-rest misses POSITION by 590/156 mm on **both** builds; this reader or the exporter, and Blender renders the same files fine. Same size on D9b ⇒ **predates D7c** | REPORT |
| B6 between-key playback (quaternions interpolated, then FK) | REPORT | inside a run **5.2 – 8.6 mm median**; the earlier 0.0003 mm was averaged composed positions and is withdrawn | REPORT |
| B6 the `Root` / eye / finger invariants | REPORT | bit-identical, both performers — a **TRACK-ARRAY** claim | REPORT |
| the provenance audit | no unaudited constant | `RIG_REST_PELVIS_MODES` registered; `PELVIS_FRAME_SOURCE` rewritten keeping its history | **PASS** |
| **merge rule, twelve conjuncts, each enforced** | all PASS | all PASS; every one turns the verdict when flipped | **MERGE** |

**Tests.** `tests/test_pelvis_rest.py` 14 passed. The full suite reads **7 failed, 1216 passed,
16 skipped**: the four superseded `test_pelvis_frame` pins (re-pinned here, §4A.5; the
coordinator re-pins that file in place at the merge), and `test_body_export::…hash_bound` and
`test_phase4_app::test_home_and_health`, which **fail identically on the D9b worktree and are
not this step's**. `test_provenance_audit` now passes.

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
* **The fixture's lever-spread comparison is confounded, and the direction is the opposite of
  what this document first claimed** (§3.2, §3A.1): under an additive, uncorrelated
  length-error model matching a rigid rig's spread to a moving performer's puts MORE noise in
  the fixture, not less — and a guard-kept sd is a conditional spread, so no harshness claim is
  made either way. What the calibration matches is a scalar LENGTH spread; it says nothing
  about the detector's directional behaviour.
* **Neither the delivered bytes nor the photographs have been looked at on any candidate.**
  No claim in this document is about a delivered file other than the shipped one.

---

## 7. What is open

1. **B1's three rising torso cells are now ATTRIBUTED, and the attribution is not one story.**
   Performer 0's rise is the ARTICULATION (+0.0119 against the root's −0.0048); performer 1's
   is the ROOT TRANSLATION (+0.0055 against the articulation's −0.0025), so performer 1's rise
   is **not** evidence that the pelvis is better there and is not claimed as such. Both
   performers' arm cells, which read ~0 in B1, are −0.005 to −0.009 of articulation cancelling
   +0.006 to +0.008 of root. §5A.7.
2. **The world-vertical control's limitation APPLIES and is stated.** Its bent-tercile error
   (17.726°) is within 2° of the truth pelvis's own median tilt from vertical (16.823°), so on
   this fixture it is doing little more than reporting how far from upright this motion's
   pelvis is. S's stops are unchanged; the winner's separation from the **frozen-pitch
   follower** — the control built for exactly this, and not upright — is what carries that
   argument.
3. **The inverse binds and the mesh-deformation reading are NOT ESTABLISHED.** Skinning at the
   nodes' own rest TRS misses the `POSITION` attribute by 590 / 156 mm, identically on the
   shipped build. Either the exporter's node rest pose is not its bind pose, or this reader's
   glTF convention is wrong — and Blender renders the same files to sensible silhouettes, so
   the reader is the likelier suspect. The attribution is established (it predates D7c); the
   measurement is not. Handed to D6 **with the reader**.
4. **The hip residual under (a) is a REPORT and stays one** — full positional p95 12.1 / 14.9
   mm against D9b's 6.4 / 12.0. No band may be made from it.
5. **One frame over 800°/s on performer 1**; a REPORT quantity by the card's own words.
6. **Two amendments in this step are POST HOC** and recorded as such: the fixture calibration
   (proposed after the σ-0.35 sensitivity was known) and the admissibility rule (amended after
   the frozen precondition failed). The σ is target-determined and the (a)/(b) ranking is
   stable at σ 1.00, 0.50, 0.35, 0.25 **and** 0.3355; that does not make the protocol
   independent of earlier outcomes.
7. **Nothing in this step resolves the pelvis CONVENTION** (§6) — lane H's marker session.
8. **The instrument-debt items are untouched**: the four `SOMA77_REST_*` constants' move to
   `tools/compare/` (containment proved, move not made), D7's moved-by-design clauses, the D3
   gate's frozen references and its translation-aligned gauge, and the four superseded pins
   left in `tests/test_pelvis_frame.py` for the coordinator to re-pin in place.
9. **The calibration matches a CONDITIONAL LENGTH SPREAD and nothing else** — not directional
   noise, not detector realism, not camera support. Length bounds no direction.
10. **Two pre-existing test failures** (`test_body_export::…hash_bound`,
    `test_phase4_app::test_home_and_health`) fail identically on the D9b worktree and are not
    this step's.

---

## 7A. The report frames, and why they are not published here

`d7c_pelvis_rest_frames.py` draws the one thing this step is about: the pelvis's own up axis,
from the hip midpoint through the rig's `Spine`, extended, against the captured `Spine1` the
step claims it should point at. Performer 1, cameras A001 and D001, the demoted run 38–46
(frame ids 98–106) with unaffected frames either side. **No magnification** — 9° of pelvis
rotation and 30 mm at `Spine` are visible at native scale, unlike D9b's one-pixel re-aim. The
captured landmarks and the Spine1 array are asserted identical between the columns before a
pixel is drawn.

The axis error on those frames, before → after: **9.80 → 3.40°, 13.89 → 6.44°, 14.70 → 4.09°**
and so on across the 21 frames.

**Nothing is published.** Two sets are written and the choice is the coordinator's:

* `report/frames/` — 21 frames, 1.9 MB, plus `d7c-pelvis-rest.mp4` (496 KB) for SendUserFile;
* `report/frames-page/` — **11 frames, 268 KB**, half scale at q38, which is what a v9 tab can
  actually carry.

"The Solve So Far" is at **9.27 of its ~9.5 MB cap**, so even the small set needs an older
tab's frames shrunk or dropped first. That is a page edit and a publish, and this agent does
neither.

---

## 8. Files

```
src/autoanim_gnm/commercial_multiview.py      `_pelvis_world_frames(rest=...)`, the two rig
                                              modes, `_pelvis_lever_guard`,
                                              PELVIS_FRAME_SOURCE = E_rig_rest_kabsch
tests/test_pelvis_rest.py                     14 tests: bit-parity, containment with its
                                              positive control, the guard's scope and frozen
                                              denominator, the four superseded pins re-pinned

tools/compare/d7c_pelvis_rest_delivery.py     the rebuild, both watchers, `--pelvis-mode`
tools/compare/d7c_pelvis_rest_gate.py         the oracle and take instrument
tools/compare/d7c_pelvis_estimators.py        FROZEN at 8a82ee4, unchanged since
tools/compare/d7c_pelvis_synthetic.py         S, the calibration, the amended G1
tools/compare/d7c_projection_preservation.py  P1 / P2 / P3
tools/compare/d7c_p1_controls.py              P1's two controls, one of which cannot be built
tools/compare/d7c_pelvis_rest_silhouette.py   B1
tools/compare/d7c_delivered_bytes.py          B5b and B6
tools/compare/d7c_gate_report.py              the clause table and the merge rule
tools/compare/d7c_pelvis_rest_frames.py       the report frames and the mp4
tools/compare/extractors/d7c_pelvis_rest.py   the ladder extractor STUB (never a registry edit)

artifacts/compare/d7c-pelvis-rest/
  gate.json                                   every clause, and the merge rule: MERGE
  delivery-hygiene-build.json                 8 of 8
  tripwire-mode-c-build.json                  8 of 8, the mode held at C
  delivery-build.json                         the shipped candidate
  instrument-shipped.json instrument-d7c.json instrument-take.json
  selector.json                               S at the card's fixture. STOP. IMMUTABLE.
  selector-calibrated.json                    the frozen-rule UNREACHABLE. IMMUTABLE.
  selector-calibrated-amended.json            REACHED, S pending
  selector-reread-sigma0.335546875.json       PROCEED, E_rig_rest_kabsch ships
  projection-preservation*.json p1-controls.json silhouette-partwise.json
  b2-delivered-vs-capture.json b3-hoist-and-contacts.json b6-delivered-bytes.json
  report/frames/ report/frames-page/ report/d7c-pelvis-rest.mp4
  logs/01 … 22
```

`git diff 8a82ee4 -- tools/compare/d7c_pelvis_estimators.py` is empty: the estimator the
selector evaluated was frozen through the calibration, the reread **and** the src change, and
the shipped branch was written to match it bit for bit.
