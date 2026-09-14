# D7c — the pelvis on the rig's own rest. **STOPPED at selector S, before any `src/` change.**

**Date** 2026-09-14 · **Branch** `ladder/D7c` · **Worktree** `.claude/worktrees/ladder-D7c`
**Verdict: the step stopped at the pre-registered stop condition in S. `src/` was not touched.**
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
| 3 | S, the selector, on synthetic truth, run BEFORE the src change was chosen | **STOP** | this commit |
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

### 3.3 Why this agent did not repair the fixture, and what it did instead

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

* **G1, missing-only.** The equivalence holds on **5 of 6** bodies, not 6. On seed 20260904 the
  guard demotes **one additional frame (21)** beyond the injected 29-frame pattern, so the
  guarded and unguarded interpolated arrays are not bit-identical there. Astra verified
  bit-identity on the *shipped code path*; on this fixture's own noisy draw one frame's lever
  lands outside the 0.15 ceiling on its own. Recovery error on the missing frames 3.7 – 5.7°
  against 3.3 – 4.3° elsewhere. **Reported, never banded** — G1 carries no superiority claim.
* **G2, finite-only.** Decisive: guarded **3.2 – 5.7°** against unguarded **62 – 77°** on the
  corrupted frames, and 1.6 – 2.6° against 3.6 – 5.8° on the transition pairs, on every body
  and on the median of six. The guard's miss rate is 0.033 (1 of 30 corrupted frames not
  rejected). **This is not a pass**: G2's verdict at the pre-registered fixture was never
  reached, and a G2 run under a repaired fixture is the coordinator's to authorise.

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
| the SOMA template through the new path must-fail | 6.865° | `C_soma_template` arm reads 6.8650 – 6.8651, identical to `src_default` | **PASS** (fails as required) |
| S: (a) vs (b) decided, not split | either | (a) wins 6/6 cells | **PASS** |
| S: the winner beats C-on-SOMA on (i), both populations | strictly better | 9.549 vs 15.970 and 11.217 vs 15.510 | **PASS** |
| **S: the follower ≥ 2× the winner on every body** | **≥ 2× on 6/6** | **1.45 – 2.74, 5 of 6 below 2×** | **FAIL → STOP** |
| S: the follower ≥ 2° on every body | ≥ 2° | 15.47 – 21.11 | PASS |
| REFACTOR TRIPWIRE | — | **not reached** | — |
| O1, O2, O3 on the candidate | — | **not reached** | — |
| P1, P2, P3 | — | **not reached** (the controls are BUILT in the delivery script and unrun) | — |
| B1 … B6 | — | **not reached** | — |
| G1, G2 at the pre-registered fixture | — | **not reached** | — |

**One FAILED prediction, and its attribution:** the frozen-pitch follower's ≥ 2× separation.
Attributed to the **fixture's noise amplitude**, measured at 1.8 – 3.0× the take's own recorded
spread, against a discrimination that is unbounded without noise (winner 0.0000° vs follower
14.401° on the bent tercile). It is **not** attributable to the estimators, to the guard, or to
the instrument: the same instrument reproduces every pre-card figure exactly, and the (a)/(b)
selection it produces is stable at every noise level tested.

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

1. **The fixture's noise amplitude against the 2× follower band — the coordinator's and
   Astra's call.** Either the band is re-pinned against a fixture measured at 1.8 – 3.0× the
   take's spread, or the fixture is repaired as a parameter (with `src/` byte-identical across
   the repair, which it trivially is: `git diff 7e35dd0 -- src/` is empty on this branch) and the SAME clauses rerun.
   The sweep in §3.4 is the input to that decision and nothing more.
2. **The card's hip-line-specific predictions must be restated from (a)'s pre-card arm**
   before any delivery (§4), including B2's hip clause, which is zero by construction only
   under (b).
3. **G1's equivalence is 5 of 6, not 6 of 6** on this fixture (one extra demoted frame on seed
   20260904). Whether that matters depends on the fixture the coordinator settles.
4. **G2 was never run at the pre-registered fixture.** Its sensitivity result is strong
   (guarded 3.2 – 5.7° vs unguarded 62 – 77°) and is not a pass.
5. **Everything from the refactor tripwire onward is unrun**: O1/O2/O3 on a candidate, P1/P2/P3
   (whose two controls are built and asserted non-degenerate in the delivery script but never
   executed), B1 – B6, the containment test, the extractor, the tests, the report page.
6. **The instrument-debt items the card hands on are untouched**: the four `SOMA77_REST_*`
   constants' move to `tools/compare/`, D7's moved-by-design clauses, the D3 gate's frozen
   references and its translation-aligned gauge.

---

## 8. Files

```
tools/compare/d7c_pelvis_rest_delivery.py     the rebuild, the watchers, the two P1 controls
tools/compare/d7c_pelvis_rest_gate.py         the oracle and take instrument
tools/compare/d7c_pelvis_estimators.py        the two rig estimators, the lever guard, the controls
tools/compare/d7c_pelvis_synthetic.py         S

artifacts/compare/d7c-pelvis-rest/
  delivery-hygiene-build.json                 PASS, 8 of 8
  delivery-hygiene/                           + projection-snapshots/, converter-inputs/
  instrument-shipped.json                     the pre-card reproduced
  oracle-shipped/                             per arm per seed, O2's baseline
  selector.json                               S, at the card's own fixture. STOP.
  selector-sensitivity-sigma{0.5,0.35,0.25}.json   REPORT ONLY
  logs/01-hygiene.log 02-instrument-shipped.log 03-selector.log 04-sensitivity-sigma*.log
```
