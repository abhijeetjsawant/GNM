# D4 — the body model in the delivery path

**Date** 2026-09-21/22 · **Branch** `ladder/D4` · **Worktree** `.claude/worktrees/ladder-D4`
**STOPPED AT STAGE 3.** O1, the exactness oracle, **FAILED** its pre-registered band on one of
six seeds — max 1.030 mm against `<= 1 mm` — and the card's merge rule is
`hygiene AND the reproduction AND O1 AND B1 on both performers AND B2`. The brief's stage order
puts O1 *before the delivery is scored* for exactly this reason, so **the delivery was not built
and B1–B5 were not run.** No band was moved, no clause was overridden, and nothing MHR-fitted was
delivered.

Hygiene **PASSED** 8 of 8. The reproduction **PASSED exactly** — and the path it reproduced was
then measured and found to carry a defect that had silently degraded performer 1 in the pre-card.
O1's own must-fail (the mean body) **behaves as predicted** and rejects by 9.6–39.8 mm; O1's
closure band **PASSED** at 2.9e-6 m and rejects three input mutations.

---

## 0. What was done, in order, and where it stopped

| stage | what | verdict | commit |
|---|---|---|---|
| 1 | hygiene — `--body rig` rebuilds the D7c delivery byte-identically | **PASS**, 8 of 8 | `6980c71` |
| 2 | the reproduction — `--body mhr`, lod6, the RAW array, through the build script | **PASS**, exactly (0.0 on 600 cells) | `6fb4288` |
| 3 | O1, the exactness oracle, its must-fails and its closure band | **FAIL → STOP** | this commit |
| 4–6 | the delivery, B1–B5, the gate, the tests, the report | **NOT RUN** (the order exists for this) | — |

---

## 1. Stage 1 — hygiene. PASS, 8 of 8.

`scripts/build_commercial_multiview_comparison.py` gained `--body {rig,mhr}` (default `rig`) plus
`--mhr-lod`, `--mhr-landmarks` and `--mhr-python`. The selector and the whole MHR adapter landed
in the same commit deliberately: hygiene's only job is to prove the changed script innocent of
disturbing the rig path, and it does.

Rebuilt into `artifacts/compare/d4-body/hygiene` with `work/` **copied** from the shipped
delivery. All eight delivered files byte-identical to `artifacts/commercial-multiview-soma77`
(`subject-00.glb 1bca2b500aa1fb69`, `subject-01.glb 3765d7455afbf622`, the two tracks, the two
JSONs, the two mappings — `logs/02-hygiene-byteidentity.log`).

**Two facts the machine forced.**

* `.cache/autoanim_gnm` was wiped between 2026-09-15 and 2026-09-21 (the pre-card records the
  clean-up), so `DEFAULT_BODY_RUN` no longer resolves. The rebuild ran with
  `--body-run artifacts/compare/d1-fix/body-run-regenerated`, the 2026-09-02 regeneration under
  the corrected joint map. **The 8 of 8 byte-identity is itself the proof** that this is the asset
  the shipped delivery was built from; nothing else establishes it, and the body-run path is now
  a fact this lane has to carry until `.cache` is restored.
* The four restored fixture videos hash to `run-report.json`'s `input_sha256` exactly, so the
  frame-extraction stamps held and no frame was re-extracted and no detection re-run.
* The build now prints the resolved `autoanim_gnm` path. Every run in this step read the
  worktree's own `src`, not the main checkout through `.venv`'s editable install.

---

## 2. Stage 2 — the reproduction. PASS, exactly. And the path it reproduced is defective.

| arm | performer 0 | performer 1 |
|---|---|---|
| pre-card fitted MHR (retained meshes) | 0.7894 | 0.7295 |
| `--body mhr --mhr-lod 6 --mhr-landmarks raw` through the build script | 0.7894 | 0.7295 |
| paired difference, 600 frame-camera cells, block 15 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

The clause asked for 0.001; the measurement is 0.0000 on every cell, so the integration path IS
the pre-card's path. `tools/compare/d4_silhouette_paired.py` gained `--arm NAME=PATH` and
`--pair CAND,REF` and reproduces the pre-card's own `paired.json` bit for bit on its default arms
(all six intervals identical), so the CLI did not move the statistic.

**Then the path was measured.** Three findings, all new, all in `tools/fitter/mhr_delivery.py`'s
docstrings with their numbers:

1. **A momentum `calibrate_markers` call mutates whatever a LATER
   `Character.load_fbx(...).load_model_definition(...)` returns in the same process.** The same
   204 model parameters then put the skeleton somewhere else: skeleton-state sum −53291.87 against
   −57778.49, the skinned mesh up to 0.76 m away, the glTF animation 79 channels instead of 98 —
   and the file still imports cleanly at the right 0..149 frame range. A character loaded *before*
   the first calibration is unaffected (its export is byte-identical before and after).
2. **The pre-card fitted both performers in one process**, so performer 1 was fitted on a model
   performer 0's calibration had moved. One process per performer: performer 1's calibrated height
   170.73 → 178.58 cm, its locator residual 27.44 → 16.72 mm, its pooled IoU 0.7295 → **0.7503**
   (+0.0208 [0.0091, 0.0403] paired). Performer 0, fitted first and so uncontaminated, is
   unchanged to 0.0 on all 600 cells.
3. **`Character.with_locators([])` does not clear locators** (17 in, 17 out), so the marker spheres
   cannot be stripped that way. The export character is loaded separately, before any calibration,
   and a guard compares its skeleton state against the fitted character's **on every frame** and
   refuses to write otherwise. That guard is what caught (1): it fired on performer 1.

The delivered GLB now reproduces momentum's own `skin_points` to 4.3e-6 m on both performers (the
pre-card's path fails that on performer 1) and carries one mesh, not eighteen.

Which number is "the reproduction"? The exact one. The clause asks whether the build script walks
the pre-card's path; it does. The repaired arm is reported beside it.

---

## 3. Stage 3 — O1. FAIL. The step stops here.

`tools/fitter/d4_o1_exactness.py`, one process per (seed, arm) — see §2(1), which makes any
multi-cell process invalid. Six seeds, the identity a draw on the card's named `scale_*` channels
each uniform within its configured limit in `compact_v6_1.model`, the other identity channels zero,
the donor motion the pre-card's own tracked pose for performer 0, the 26 `*_flexible` channels at
zero in the truth and removed from the fitter's transform (`simplify_parameter_transform`,
204 → 178), truth landmarks the 17 mapped joints at the pinned zero offsets, the fitter given
landmarks only.

### 3.1 The clause table

| clause | predicted | measured | verdict |
|---|---|---|---|
| **O1 exactness** | max over six seeds ≤ 1 mm | 0.853 / 0.985 / **1.030** / 0.967 / 0.799 / 0.898 mm, **max 1.030** | **FAIL** |
| O1 must-fail, mean body | misses it on every seed, 5–60 mm | 31.8 / 32.8 / 24.7 / **9.6** / 39.8 / 32.2 mm | PASS (fails on all six; the 9.6 mm seed is inside the predicted range) |
| O1 closure | the delivered GLB's FK = the track, ≤ 1e-4 m | worst 2.9e-6 m on eight files | PASS |
| closure mutations | a mutated input must fail it | crossed seeds 1.6e-1 m, crossed performers 1.9 m, mean-body track 1.1e-1 m — all reject | PASS |
| must-fail, free locator offsets | offsets median > 50 mm **and** the identity channels at zero | offsets **84.0 / 143.0** mm (max 177 / 317), residual 81.6 / 195.8 mm — but **13 identity channels nonzero on both** | the ARM fails as it must; the PREDICTION's second half is **FALSE** |
| *reported, not a card arm* — exact-identity floor | — | 0.506 / 0.561 / 0.749 / 0.757 / 0.593 / 0.635 mm, **max 0.757** | — |

The FAIL is robust to how the card's statistic is read. Max over six seeds, oracle arm:
median-of-per-frame-medians **1.030**, pooled median over joint-frames **1.259**, worst per-joint
median **3.924**, mean **2.245** mm. Every reading exceeds 1 mm.

### 3.2 Attribution of the FAIL

**It is not the instrument, and it is not the flexible channels.** Scored one cell per process:
the simplified 178-parameter character and the full 204-parameter character with
`TrackingConfig.active_params` masking the flexible channels agree to four decimals (0.8777 vs
0.8777, 1.1604 vs 1.1603). Astra's finding 1 — that `spine_length_flexible` duplicates
`scale_spine_length` — is real and handled, and it is worth **nothing** numerically here:
excluding the 26 channels moved the first seed by 0.012 mm. What it does buy is that the oracle
cannot be exact for the wrong reason.

**Part of it was the fixture, and that part was repaired.** The card calls the donor "natively
representable". It is FK-representable — but measured against `compact_v6_1.model`'s own
`limit ... minmax` entries the donor violates **23 pose parameters on up to 150 frames each**
(`r_clavicle_rx` has limit [0, 0] and is nonzero on every frame; `head_lean` has [−0.3, 0.3] and
reaches 0.924). momentum's limits are soft — the pre-card's own tracker produced this motion — but
they are a penalty in the solve, so a truth outside them is not a pose the solver will return, and
the fixture was charging the fitter for its own infeasibility. Clamping the donor's pose to those
configured limits (1025 parameter-frames) is a **fixture parameter**, applied with the fitter's
source byte-identical across the repair (`mhr_delivery.py` sha256 `8ff8d973…`), and the same
clauses were rerun:

| | pre-repair | post-repair |
|---|---|---|
| oracle, max over six seeds | **1.248 mm** | **1.030 mm** |
| exact-identity floor, max over six seeds | 1.160 mm | 0.757 mm |

**The pre-repair reading stands on record** (`artifacts/compare/d4-body/o1-prerepair.json`), is
reproducible with `--no-clamp` (verified: 1.2484 mm on the worst seed), and the repair is recorded
as post hoc, per D7c's rule.

*What the repair is blind to.* `configured_limits()` parses the model file's `limit ... minmax`
lines. Every one of the 198 `limit` lines in `compact_v6_1.model` is `minmax`, so nothing declared
there is missed; the character reports 172 `parameter_limits` (fewer, because some named
parameters are not in this transform), and any limit momentum synthesises rather than reads from
the file was not checked against the donor.

**What remains is the fitter's own cost, against a floor the band barely clears.** With the
*truth* identity handed in and only the pose re-solved, the tracker still leaves **0.51–0.76 mm**
on exact, noiseless, fully visible data. So the 1 mm band allows the identity fit **0.24 mm** of
headroom on the worst seed, and the identity fit costs **0.21–0.35 mm**. The excess is
concentrated in two channels:

* `scale_spine_length` is **shrunk toward zero** on every seed. The sign of the error is the
  opposite of the sign of the draw in six of six cells (draw +1.082 → −0.189, +1.078 → −0.209,
  −0.955 → +0.170, −0.750 → +0.153, +0.743 → −0.151, −0.886 → +0.149), i.e. 14–20 % of the drawn
  value pulled back to the mean body, with `scale_neck_length` compensating in the trunk on the
  seeds where the spine is stretched (+0.019 to +0.033) and idle where it is shortened (±0.001).
  That is a **regulariser**, not an unexplained bias: `scale_spine_length` is the one drawn
  channel whose `limit` line in `compact_v6_1.model` carries no trailing weight
  (`scale_neck_length`'s carries 0.1), so it sits on the default soft-limit weight. It costs
  2–4 mm at `c_neck` and `root`, the two worst-scoring joints on five of six seeds. **This is the
  lane's own rule in a new place: a quantity the solver regularises is a knob setting, and O1's
  1 mm band does not accommodate that knob's current value.**
* `scale_foot_length` is **unidentifiable from this landmark set** and is recovered at ≈ 0 on
  every seed. `l_foot`/`r_foot` are the ankles and the adapter maps no toe, so nothing in the 17
  landmarks moves with foot length. The card's `scale_hip_height` is a second dead channel for a
  different reason: its *configured* limit is the point [0.0, 0.0], so "uniform within its
  configured limit" draws identically zero. Both are listed in the report rather than swapped out.
* `scale_feet` does not exist in this release; the card's "nearest named channel" rule put
  `scale_foot_length` in its place, listed in `o1.json`.

*The tail.* Even with the exact identity the worst frames read 3.4–5.2 mm (per-frame median over
the 17 joints), and they are the **same frames on every seed** — 0, 30, 32/33, 52/53, 57/58 — so
it is pose-dependent, not noise, and frame 0 being among them on four of six seeds is the tracker
starting from the rest pose on a performer who is bent over the ground. Reported; it is the
delivery's first frames too, and stage 4 should expect it.

The locator offsets are not the explanation: with `limit_weight 10` they stay at 0.0000 mm median
and 0.0009 mm max after the full two-stage fit, so the card's "the same offsets the fitter uses"
premise holds exactly.

### 3.3 What O1 did prove

The clause it was built for — *reject the mean body, which already beats the rig on B1 by
+0.097 / +0.070 and would otherwise pass that band on its own* — it discriminates by a factor of
**9 to 39**: 9.6–39.8 mm against the candidate's 0.8–1.0 mm. There is no reading of this fixture
under which a constant identity passes.

---

## 4. The free-offset must-fail: half of a pre-registered prediction is false

Run on the take (raw array, lod6, one process per performer):

| | offsets median | offsets max | residual median | nonzero identity channels |
|---|---|---|---|---|
| pinned (limit_weight 10) | 0.0 mm | ~0 mm | 11.7 / 16.7 mm | 13 / 13 |
| **free** | **84.0 / 143.0 mm** | 177 / 317 mm | **81.6 / 195.8 mm** | **13 / 13** |

The arm is pathological and fails as a must-fail must — by offsets six to nine times over the
50 mm clause and by a residual five to twelve times the pinned fit's. But the prediction's second
half, "*and* the identity channels at zero", is **false**: 13 channels are nonzero on both
performers and several sit on their configured limits (`scale_shoulder_width` 0.200 of 0.2,
`scale_r_hands`/`scale_l_hands` 0.200 of 0.2, `scale_spine_length` 1.101 of 1.1). This is exactly
Astra's debt item — the "identity at zero" claim was carried over from FITTER_PLAN §7's
**locator-only stage A** and was never established for the full two-stage control. It is recorded
here as FAILED-as-a-prediction, with the arm's rejection intact on the other half.

---

## 5. Findings worth carrying out of this step

1. **A momentum calibration mutates later model loads in the same process.** Any instrument that
   fits and then loads MHR again is measuring a different model. One process per fit. The cheap
   guard is comparing skeleton states between the two characters before writing anything — it cost
   nothing and it caught a wrong delivered file.
2. **The pre-card's performer-1 number was produced on a contaminated model** and is 0.021 IoU
   low. Every MHR figure quoted for performer 1 before 2026-09-22 carries that.
3. **A GLB can import cleanly, at the right frame range, with the right vertex count, and be
   wrong.** The Blender action range 0..149 was correct in both the good and the bad export; only
   comparing the file against momentum's own skinning separated them.
4. **`scale_foot_length` and `scale_hip_height` are dead channels for this landmark set** — one
   for want of a toe landmark, one because its configured limit is a point. Any identity gate over
   MHR must say which of its channels the input can see.
5. **The 1 mm band sits at ~1.3× the fitter's own floor on exact data.** That is not a reason to
   move it — it is a fact the next card needs before it re-registers one.

---

## 6. What is open, and what is owed

* **The band.** O1 fails at 1.030 mm against 1.000 mm, with a measured floor of 0.757 mm on the
  same fixture. Whether the band is re-registered against that floor, or the fitter's spine-length
  estimate is improved, or the fixture's draw range is narrowed, is **the coordinator's and
  Astra's call, not this agent's**. Nothing here may move it.
* **`scale_spine_length` is regularised toward the mean by ~15 % of the draw**, by a
  default-weight soft limit in MHR's own model definition. It is not the flexible duplicate, not
  the locator offsets, not the iteration count (the identity estimate is unchanged from max_iter
  30 to 300) and not the fixture's limits. Whoever re-registers the band has to decide whether
  that weight is part of the shipped fitter or a knob — and if it is a knob, it may not be
  selected on a MAMMA arm, and selecting it on this fixture makes O1 a knob setting rather than
  an oracle.
* **B1–B5 are unrun**, the delivery is unbuilt, and the card's `gate.json`, extractor stub, report
  frames, mp4 and new tests belong to the stages that were not reached.
* **The MHR path consumes 17 of the 19 body landmarks and none of the rig's auxiliary feeds** (the
  head solve, the toes, `Spine1`). B2 was written to log the consumed array
  (`converter-inputs/`, produced by every `--body mhr` build) but its comparison was not run.
* The pelvis and hip conventions, and the identifiability of the locator offsets, remain lane H's
  marker session. O1 is a **model-consistency** oracle and says nothing about them.

---

## 7. Reproduce

```
PYTHONPATH=$PWD/src .venv/bin/python scripts/build_commercial_multiview_comparison.py \
  --videos .cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos \
  --calibration-yaml .cache/mamma/configs/examples/calib/iphones_outdoors.yaml \
  --detector soma77 --body rig --body-run artifacts/compare/d1-fix/body-run-regenerated \
  --output artifacts/compare/d4-body/hygiene                       # stage 1

... --body mhr --mhr-lod 6 --mhr-landmarks raw --output artifacts/compare/d4-body/repro-v2
/Applications/Blender.app/Contents/MacOS/Blender --background --python \
  tools/compare/blender_export_mesh_momentum.py -- OUT.npz REPRO_DIR 30
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_silhouette_paired.py --out R.json \
  --arm ... --pair ...                                             # stage 2

/tmp/momenv/bin/python tools/fitter/d4_o1_exactness.py --drive \
  --out artifacts/compare/d4-body/o1 --lod 2                       # stage 3
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_glb_closure.py --pair NAME=GLB,TRACK --out C.json
```

Logs, every line, under `artifacts/compare/d4-body/logs/`; reports beside them under
`artifacts/compare/d4-body/`.
