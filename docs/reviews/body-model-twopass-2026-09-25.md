# D4d: a second calibration pass on the landmark start. The measurement (2026-09-25)

**VERDICT: PASS. The combined fitter (D4c's landmark start plus a second calibration pass) supersedes D4's O1 under
the card's registered band. Nothing is merged from here: the coordinator carries the card's merge rule.**

The gate prints one line:

    VERDICT: PASS (every conjunct holds)

**What this PASS does not show, first.** The twelve acceptance bodies do not contain the class D4c failed on. D4c's
misses (four, across its development and acceptance) were all strongly shortened spines on donor 0 (draws −0.75 to
−0.955). Phase 2's uniform draw gave donor 0 spines of +0.367, −0.269, +0.743, −0.315, −0.397 and +0.775: none below
−0.40. The one strongly negative draw is on donor 1 (20261206/d1, −0.959), where D4c's start was never the problem.
Consistent with that, the REPORTED one-pass arm, which is D4c's fitter, also passes L on 12 of 12 (worst trunk
0.883×). **So the acceptance set cannot tell the candidate from D4c's fitter.** The evidence that the second pass
repairs the failing class lives only in Phase 1, on burned fixtures. There, all four one-pass misses close:

| fixture | spine draw | one pass (D4c) | two passes |
|---|---|---|---|
| 20261106/d0 | −0.911 | 1.103× | 0.676× |
| 20260924/d0 | −0.955 | 1.273× | 0.784× |
| 20260925/d0 | −0.750 | 1.018× | 0.617× |
| 20260927/d0 | −0.885 | 1.145× | 0.722× |

The card licenses this PASS. It registered a uniform draw, "not re-stratified, which would change the fixture
distribution", with results reported by spine tercile. The gap is stated here, not repaired.

| stage | commit | what |
|---|---|---|
| 1a | `f04bd9b` | Merge tag `ladder/D4c-fail-1a89cc7` (D4c's landmark start and tooling). One add/add conflict, D4c's review: main's copy kept, which is the tag's text plus the disposition header |
| 1b | `48eb884` | Provenance and precondition 0. D4c's drawn set was reused by sha256: eight drawn, the spine among them, so no STOP. The retained D4c cells match D4c's records (72/72 and 108/108) |
| 2 | `1efdf08` | The ONE code change and the tripwire. Hygiene HOLDS: 216/216 tripwire files, `--body rig` 8/8, the source diff is the second pass only |
| 3 | `8dda2ea` | Phase 1 on 30 burned fixtures. The decision JSON selects TWO-PASS and was committed before any Phase-2 fixture existed |
| 4 | `4fb2e18` | Phase 2: 12 × 7 arms, each cell stamped with the decision's sha256. Closure on every candidate cell. The manifest |
| 5 | `9be28b9` | The real take rebuilt through `--body mhr`; B1, B2 and the delivery closure |
| 6 | `614512e` | The gate (ONE verdict) and its fuzz, with CRASH kept as its own class |
| 7 | this commit | The tests, this review, the extractor stub and the report frames |

The records are in `docs/reviews/body-model-twopass-records/`. The logs are in `artifacts/compare/d4d-twopass/logs/`
(01–26). Nothing under `src/` changed, and neither did the build script.

---

## 1. The pre-registration

The card is the "D4d a second calibration pass" row of `docs/LADDER_EXECUTION_PLAN.md` §2 (lines 160–206).
`docs/reviews/body-model-twopass-card-2026-09-25.md` is byte-identical to those lines (checked with `cmp`). It is not
repeated here. The Astra card review is `docs/reviews/body-model-twopass-astra-review-2026-09-25.md`, and every finding
in it was already in the card.

## 2. The clause table (`artifacts/compare/d4d-twopass/gate.json`)

| clause | predicted (the card) | measured | verdict |
|---|---|---|---|
| precondition 0 | (carried) the spine drawn | D4c's frozen set, reused by sha256 `430f1f67…`. Eight drawn, the trunk scored | HOLDS (held) |
| stage 0b | the spine control fails L as scored on every Phase-1 fixture | 30/30 | HOLDS (held) |
| **Phase 1: WARM** | holds on 12/12 | **12/12** on D4c's 12. Worst trunk 0.214×. **20261106/d0: 0.214× (0.41 mm against 1.94 mm)** | HOLDS (held) |
| **Phase 1: TWO-PASS** | brings D4c's 1.10× below 1, every segment within tolerance on all 12 | **12/12**. Worst segment 0.726×. 20261106/d0: 1.103× → 0.676× | **TWO-PASS selected** (held) |
| validity | exact pooled ≤ 1.0 mm on every fixture | 0.549–0.848 mm, 12/12 | PASS (held) |
| **L** | PASS, the worst trunk on a strongly negative spine | **12/12**. Worst trunk 0.726× on **20261201/d1, spine −0.206 (middle tercile)**. Worst limb or width 0.316× | **PASS** (the "strongly negative" half FAILED, §4) |
| closure | ≤ 1e-4 m, bound by content | worst 2.6e-6 m over 12 × 127 joints × 150 frames. Every row matches the cell's GLB and track sha256 and the files | PASS (held) |
| must-fail (i), mean body | misses L | 12/12. The closest fixture is 36.4× | PASS (held) |
| must-fail (ii), spine −0.149 | misses L at the trunk | 12/12. The closest fixture is 6.0× | PASS (held) |
| must-fail (iii), exact identity | reads 0 and passes | 12/12 at 0.000000 mm | PASS (held) |
| must-fail (iv), init-only | misses L (STOP if it passes) | 12/12. The closest fixture is 4.9×. It misses at the trunk on 3/12, and misses through shoulder width on every fixture (4.9–20.2×), as in D4c | PASS (held) |
| **B1** | combined − D7c lower CI > 0 on both performers; combined − D4c within ±0.01 | **+0.1517 [+0.1295, +0.1591]** and **+0.1164 [+0.0836, +0.1337]**, 600/600 cells each. The frozen-pose control is below the candidate 8/8. MAMMA is bit-identical 8/8. The mesh and GLBs are bound by sha256. Reported, combined − D4c: −0.0039 [−0.0058, −0.0004] and −0.0014 [−0.0148, +0.0028] | PASS (held; see §5) |
| B2 | (carried) same denominator | Every numbered check is re-derived by the gate from the delivery's own files, against this step's `--body rig` build. Both performers pass. D4's instrument agrees | PASS (held) |
| hygiene | `--body rig` 8/8; passes = 1 reproduces D4c's delivery and cells; the source diff is the second pass only | 8/8. Tripwire 216/216: 106 byte-identical, 110 equal under the named normalisations only. Source diff: one kwarg (default 1) and exactly one statement | PASS (held) |
| **overall** | **PASS** | **PASS** | **PASS** |

The falsifier ("WARM drifts on 20261106/d0 → the basin → STOP") did not fire. WARM reads 0.214× there.

Reported: the allowable rest-length intervals have an **empty intersection on every scored segment**, so no common
constant body passes all twelve fixtures. The legacy zero-start arm passes L on 1/12 (worst trunk 31.7×). J, the
candidate minus the paired floor, is at most +0.09 mm on any joint.

## 3. The Phase-1 record (`phase1-decision.json`, committed at `8dda2ea` before any Phase-2 path existed)

These are 30 burned fixtures. The D4c start, the paired floors and the spine controls are D4c's retained cells,
bound by content to D4c's manifest and development JSON. WARM, TWO-PASS and SW* are new. The gate verifies how each
arm was built:

* every start is recomputed from the consumed landmarks;
* each pass's two stages were handed the same identity: the start, then pass 1's stage-B return;
* **two-pass's first pass equals D4c's one-pass fitted identity, bit for bit, on all 30.**

L passes / worst trunk ratio / worst scored segment ratio:

| population | D4c start (one pass) | WARM | TWO-PASS | SW* | spine control |
|---|---|---|---|---|---|
| D4c's 12 (the fork) | 11/12 · 1.103 · 1.103 | 12/12 · 0.214 · 0.214 | **12/12 · 0.726 · 0.726** | 12/12 · 0.877 · 0.877 | 0/12 · closest 6.09 |
| D4's 6 | 3/6 · 1.273 · 1.273 | 6/6 · 0.219 · 0.219 | 6/6 · 0.784 · 0.784 | 6/6 · 0.929 · 0.929 | 0/6 · closest 7.05 |
| D4b's 12 | 12/12 · 0.903 · 0.903 | 12/12 · 0.144 · 0.180 | 12/12 · 0.839 · 0.839 | 12/12 · 0.537 · 0.537 | 0/12 · closest 6.25 |

**The spine terciles.** The rule is the configured limit [−1.1, 1.1] cut in thirds, frozen in `d4d_fixture.py`
before any reading. On D4c's 12, the short tercile holds two fixtures. The one-pass worst there is 1.103×, TWO-PASS
reads 0.676×, WARM 0.214× and SW* 0.877×. The middle and long terciles are within tolerance for every arm.

**The failing fixture, per stage (20261106/d0).** Stage A (locators only) does not move the identity. Stage B does.

| | spine error (start 0.0324) | trunk ratio |
|---|---|---|
| after A1 | +0.0324 | 1.673× |
| after B1 | +0.0214 | 1.103× (D4c's reading) |
| after A2 | +0.0214 | 1.103× |
| after B2 | +0.0131 | 0.676× |

The second pass removes about another 40 % of what the first left.

**Donor 1 and SW*.** On donor 1 the landmark start is exact for the spine (≤ 0.0004), and one pass drifts it off to
−0.005 … −0.019. The second pass pulls it back only part of the way (−0.004 … −0.017). With ONLY shoulder width started
at the truth (SW*), donor 1's spine error falls to +0.001 … +0.004, the same as WARM's on every one of the six. This
is D4c's coupling hypothesis ("a wrong companion channel") read directly: shoulder width's chord start (−0.05 … −0.13
off on donor 1) is what pulls the spine. SW* is truth-seeded, a report and never a candidate. Under the card, it is
the lead for a landmark-derived shoulder-width start that would need its own registration.

**Cap exhaustion (reported).** Every calibration call still stops at or above the configured cap on most solves, in
both passes. For TWO-PASS in Phase 1, per stage:

* A1: 598–608 at the cap, 255–387 above it, of 1054;
* B1: 605–608 at, 364–413 above, of 1056;
* A2: 603–608 at, 351–403 above;
* B2: 604–609 at, 369–417 above.

The same pattern holds in Phase 2. The second pass is not a converged fit either.

**Instrument checks (reported).** WARM's fitted identity equals D4b's retained WARM cells (D4b's proxy on the D4
fitter) on 18/18. That is a second reading of determinism across fitter versions.

## 4. The FAILED prediction, attributed

**"Phase 2: the worst trunk on a strongly negative spine."** The worst trunk is 0.726× on 20261201/d1 (spine −0.206,
middle tercile).

* **Attribution: the population.** The uniform draw produced no donor-0 spine below −0.40. The strongly negative
  class that produces the largest start error (the p90 chord reads a shortened spine LONG on donor 0's motion) was
  not in the acceptance set.
* On donor 1, where the one strongly negative draw fell, the landmark start is exact. The residual there is the
  calibration's drift off an exact start, which is largest at middle-tercile draws.
* This is not a fitter finding, and it is the same fact as the first paragraph: the acceptance set did not exercise
  the class Phase 1 repaired.

Every other prediction held: WARM 12/12, TWO-PASS selected, PASS, and B1 within ±0.01 of D4c.

## 5. Where the second pass reads worse (reported; it feeds what is open)

* **Phase 2.** Two passes read worse than one at the trunk on 2 of 12: 20261203/d0 (0.039 → 0.188×) and 20261205/d1
  (0.644 → 0.706×). Phase 1 shows the same on 3 of 30: 20260922/d0 (0.066 → 0.160×), 20260923/d0 (0.075 → 0.214×)
  and 20261103/d0 (0.020 → 0.075×), all long-spine draws where pass 1 was already near. It is D4c §5's signature,
  "started at the truth, the fit leaves it", with a second pass adding a little more of it. All stay well inside
  tolerance.
* **The real take (no truth).** The second pass moves the fitted spine further:
  * performer 0: 0.959 → **1.102**, at and just over the configured limit 1.1 (the limits are soft penalties);
  * performer 1: 0.420 → **0.823**.

  The landmark residual rises from 16.7 to 17.7 mm and from 16.7 to 17.1 mm. B1 against D4c reads −0.0039, with the CI
  entirely below zero on performer 0, and −0.0014 on performer 1. The card registered a regression against D4c that
  still beats D7c as a PASS, and B1's band holds by a wide margin (lower CI +0.130 / +0.084). But on the real take the
  second pass moves the identity by up to 0.4 units, and the silhouette cannot say which spine is right. The take's
  own `neck`→`nose` convention gap (D4c §6, lane H) is the likeliest thing a longer spine is absorbing. That is a
  hypothesis.

## 6. What each instrument is blind to

* **Phase 2 is blind to the failure class** (the opening section and §4). The one-pass arm passing 12/12 is the proof.
* **L scores lengths, not the parameter vector.** Compensating channels can hold every scored length with a wrong
  identity. Donor 1's spine drift under a wrong shoulder-width start is the measured example.
* **The tolerance is an engineering rule, not a bound**: the sum of two marginal medians.
* **Every fixture is exact, noiseless, with pinned zero offsets.** The real take carries the landmark-to-joint
  convention gap, which no fixture has.
* **Only two donor motions exist.** "Held-out" means the identity, never the pose (lane H).
* **Attribution is to the implementation only** (the card). Two passes do not separate extra iterations from the
  restart and stage effects. The PASS does not show that cap exhaustion caused D4c's miss: both passes stop at the cap
  on most solves.
* **WARM decided the fork and selects nothing.** It holding shows no drift under this procedure from the truth. It
  does not show that the objective's minimum is the truth.
* **B1** scores the mesh's outline against clothed masks. It cannot see a 0.4-unit spine change on performer 1.
* **The delivery's `passes: 2` label** is read from `calibration-passes.json`, the producer's own file. The gate
  binds it by elimination and REPORTS the binding without banding it. The consumed arrays are byte-identical to
  D4c's delivery's, D4c's delivery is the one-pass fit of those bytes (the tripwire), and the delivered identity
  differs from D4c's on both performers. What remains unbound (that the difference is the second pass and nothing
  else) rests on the source diff.
* **The fuzz** (§7) proves that what the gate reads, it depends on, not that it reads the right things.

## 7. The gate and its fuzz

`tools/compare/d4d_twopass_gate.py` derives every clause from the files. Two pieces of D4c's instrument debt are
repaired, as the card required:

* the tripwire's equality is re-derived from the bytes, byte equality first and then ONLY the named normalisation
  for that file kind (`body_model.assets` in a track JSON, `provenance.fitter_sha256` in a cell JSON, momentum's
  wall-clock timestamps in a debug log);
* B2's nine numbered checks are re-derived from the delivery's own arrays; the producer's booleans are never read.

`tools/compare/d4d_twopass_gate_fuzz.py` (log 22, `fuzz.json`) does two things.

* **Targeted mutations: 54/54 turn and 0 crash.** Each mutation must move the measured PASS to the verdict the card
  gives, AND the targeted conjunct itself must fall, so a mutation cannot turn through an unrelated failure. Three
  NEGATIVE controls must not move it, and they do not: a tripwire cell differing only in `provenance.fitter_sha256`,
  a debug log differing only in timestamps, and a reported value.
* **The leaf walk.** It is bounded to one Phase-2 fixture's seven cells, its closure row, the B1 paired rows, and one
  Phase-1 fork fixture's three new cells. The Phase-1 cells are walked twice: as committed, and with the decision
  re-bound. **3420 leaves enforced, 1801 reported with a named justification, 0 crash and 0 unjustified.**
  Walking it found two holes, both fixed in the gate before this reading:
  * a malformed Phase-1 cell crashed the gate's construction check; malformed inputs now fail closed;
  * the non-calibrating arms' `start` fields were unchecked; they now must match the identity each arm held.

CRASH is its own class, never counted as ENFORCED.

## 8. What is open

* **D4's O1 closes as "superseded by the combined fitter", on the card's band.** The consequence the card registers
  on PASS: D4, D4c and D4d close. D4c's deferred tooling merge is discharged. `--body mhr` carries the combined
  fitter, and the default stays `rig`. **Nothing is merged or pushed from this branch; the coordinator runs the
  close-out.**
* **The failure class was not re-tested on untouched bodies** (the opening section). A registration that wants acceptance evidence on
  it needs fixtures that contain it, for example a stratified or conditioned draw, registered as a changed
  population.
* **The real-take identity moves 0.14 / 0.40 units under the second pass** (§5), with no truth to referee it. Lane H's
  markers or a `Spine1` feed are the instruments that could.
* **SW* is the lead** for a landmark-derived shoulder-width start (§3). That needs its own registration.
* **Two of D4c's tests pin D4c's fitter and now fail on this branch, by design** (log 26). They are not edited: this
  step may not touch an existing test.
  * `test_the_source_diff_accepts_only_the_two_starts` asserts the fitter differs from `3136befb` in exactly D4c's
    two starts.
  * `test_the_gate_reproduces_its_committed_verdict_from_the_artifacts` re-runs D4c's gate, which checks every
    acceptance cell's recorded fitter sha256 against the CURRENT fitter.

  Both are true of D4c's tag and false of any later fitter. The merge that the PASS licenses needs the coordinator
  to pin them to the tag's fitter or retire them. The rest of the suite reads 1261 passed, 43 skipped and 6 failed:
  these two, plus D4c's four pre-existing (`test_body_compositor` unified preview, `test_body_export` GLB hash-bound,
  and two `test_phase4_app`).
* **Cap exhaustion is unchanged in kind.** Neither pass converges. That is an attribution limit, not a clause.
* **The detection cache.** Both real-take builds (`--body rig` hygiene and `--body mhr` delivery) reused the shipped
  delivery's cached detections (copied in). The SOMA-77 detector is restored on this machine but was not re-run.

## 9. Decisions an executor made

1. **WARM and SW* run ONE pass.** The card defines WARM as "calibration started at the truth identity", D4b's WARM,
   which is one pass. SW* modifies only the start. Both go through `fit_one(start_identity=…)`, never D4b's proxy
   override, which would also replace pass 2's start. TWO-PASS is the only two-pass arm in Phase 1.
2. **The Phase-1 D4c-start, floor and spine-control arms are D4c's retained cells.** They are not rerun. They are
   bound by content to D4c's records, and the stage-2 tripwire shows the new code reproduces them. Two-pass's first
   pass equalling D4c's fitted identity on all 30 is a second reading of that.
3. **The pass count is written to `subject-XX.calibration-passes.json`**, a new file, so passes = 1 leaves every D4c
   delivery file byte-identical. The CLI defaults to 2 (what `--body mhr` runs); `fit_one` defaults to 1.
4. **The tripwire reran D4c's own fixture** (`d4c_fixture.py`, unchanged) on the new fitter, for all 72 acceptance
   cells.
5. **The silhouette instrument's cache was seeded** with the MHR mesh from `blender_export_mesh_momentum.py`, as D4c
   did. `silhouette.py` exports with the rig exporter, which fails on an MHR GLB (log 16 records the failed first
   attempt).
6. **The spine-tercile rule** is the configured limit in thirds, population-independent, frozen before any reading.

## 10. Reproduce

```
git merge ladder/D4c-fail-1a89cc7                                       # stage 1a
/tmp/momenv/bin/python tools/fitter/d4d_fixture.py --provenance --out docs/reviews/body-model-twopass-records/provenance.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate.py --precondition-0 --out docs/reviews/body-model-twopass-records/precondition-0.json
for s in 0 1; do /tmp/momenv/bin/python tools/fitter/mhr_delivery.py --inputs artifacts/compare/d4c-start/delivery/converter-inputs \
    --out artifacts/compare/d4d-twopass/tripwire/delivery --lod 2 --landmarks smoothed --subject $s --passes 1; done
/tmp/momenv/bin/python tools/fitter/d4c_fixture.py --drive --population acceptance --out artifacts/compare/d4d-twopass/tripwire/acceptance
cp -Rp artifacts/commercial-multiview-soma77/work artifacts/compare/d4d-twopass/hygiene/
PYTHONPATH=$PWD/src .venv/bin/python scripts/build_commercial_multiview_comparison.py --videos .cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos \
    --calibration-yaml .cache/mamma/configs/examples/calib/iphones_outdoors.yaml --detector soma77 --body rig \
    --body-run artifacts/compare/d1-fix/body-run-regenerated --output artifacts/compare/d4d-twopass/hygiene
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate.py --hygiene --out docs/reviews/body-model-twopass-records/hygiene.json
for P in d4c d4 d4b; do /tmp/momenv/bin/python tools/fitter/d4d_fixture.py --drive --population $P --out artifacts/compare/d4d-twopass/phase1; done
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate.py --phase1 --out docs/reviews/body-model-twopass-records/phase1-decision.json
git commit ...                                                          # the decision, BEFORE Phase 2
/tmp/momenv/bin/python tools/fitter/d4d_fixture.py --drive --population phase2 --arms A --out artifacts/compare/d4d-twopass/phase2   # per arm
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_glb_closure.py --pair SEED_dD=GLB,TRACK ... --out artifacts/compare/d4d-twopass/closure.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate.py --manifest --out docs/reviews/body-model-twopass-records/acceptance-manifest.json
cp -Rp artifacts/commercial-multiview-soma77/work artifacts/compare/d4d-twopass/delivery/
PYTHONPATH=$PWD/src .venv/bin/python scripts/build_commercial_multiview_comparison.py ... --body mhr --mhr-lod 2 --mhr-landmarks smoothed \
    --output artifacts/compare/d4d-twopass/delivery
Blender --background --python tools/compare/blender_export_mesh_momentum.py -- artifacts/compare/d4d-twopass/work-delivery/delivered-mesh.npz \
    artifacts/compare/d4d-twopass/delivery 30
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/silhouette.py --delivery artifacts/compare/d4d-twopass/delivery \
    --work artifacts/compare/d4d-twopass/work-silhouette --out artifacts/compare/d4d-twopass/silhouette-delivery.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_silhouette_paired.py --out artifacts/compare/d4d-twopass/b1-paired.json \
    --arm baseline_D7c_rig=artifacts/compare/i6/delivered-mesh.npz \
    --arm D4c_fitted_MHR_lod2=artifacts/compare/d4c-start/work-delivery/delivered-mesh.npz \
    --arm D4d_fitted_MHR_lod2=artifacts/compare/d4d-twopass/work-delivery/delivered-mesh.npz \
    --pair D4d_fitted_MHR_lod2,baseline_D7c_rig --pair D4d_fitted_MHR_lod2,D4c_fitted_MHR_lod2
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_b2_same_denominator.py --delivery artifacts/compare/d4d-twopass/delivery \
    --rig-build artifacts/compare/d4d-twopass/hygiene --out artifacts/compare/d4d-twopass/b2-same-denominator.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate.py --out artifacts/compare/d4d-twopass/gate.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate_fuzz.py --out artifacts/compare/d4d-twopass/fuzz.json
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_d4d_twopass.py
```

`tests/test_d4d_twopass.py` passes 16 tests (log 25). The full suite (log 26) reads 1261 passed, 43 skipped and 6 failed: D4c's four pre-existing failures and the two D4c tests that pin D4c's fitter (§8).

The report frames are in `artifacts/compare/d4d-twopass/report/`: 25 JPEGs, 480 px, q40, every 6th frame, camera
A001. The panels are stacked: the D4c fit (aqua) sits above the D4d fit (blue), over the SAM2 mask outline. The
full-rate `d4d-twopass-A001.mp4` is beside them. They are rendered by D4's own `d4_report_frames.py`, imported and
re-pointed by the driver below (kept as `artifacts/compare/d4d-twopass/report-driver.py`; log 23):

```python
"""D4d report frames: D4's own d4_report_frames.py, imported and re-pointed (D4c's precedent, its log 23).
Stacked panels: the D4c fit (aqua, the alternative) above the D4d combined fit (blue, ours), over the SAM2 outline."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/abhi_macbook/Projects/apps/AutoAnim/.claude/worktrees/ladder-D4d")
sys.path.insert(0, str(ROOT / "tools/compare"))
import d4_report_frames as rf  # noqa: E402

rf.FRAMES_DIR = ROOT / "artifacts/compare/d4d-twopass/delivery/work/frames"
rf.ARMS = {"D4c fit, one pass": ROOT / "artifacts/compare/d4c-start/work-delivery/delivered-mesh.npz",
           "D4d fit, two passes": ROOT / "artifacts/compare/d4d-twopass/work-delivery/delivered-mesh.npz"}
rf.COLOURS = {"D4c fit, one pass": (225, 225, 90), "D4d fit, two passes": (255, 150, 40)}
rf.PANEL_WIDTH, rf.JPEG_QUALITY = 480, 40


class _NP:
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def hstack(panels):
        return np.vstack(panels)


rf.np = _NP()
sys.argv = ["d4_report_frames.py", "--out", str(ROOT / "artifacts/compare/d4d-twopass/report"), "--cameras", "A001",
            "--mp4-camera", "A001"]
raise SystemExit(rf.main())
```

It was run as `PYTHONPATH=$PWD/src .venv/bin/python report-driver.py`, and the mp4 was renamed to
`d4d-twopass-A001.mp4`. The extractor stub is `tools/compare/extractors/d4d_twopass.py`, and the
registry is not edited.
