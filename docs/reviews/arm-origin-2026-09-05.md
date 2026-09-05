# D9 — aim the arms from their own origins, so the delivered arm sits on the captured points

**Branch** `ladder/D9`, from `26bef65` (D8 merged). **Date** 2026-09-05.
**Verdict: MERGE.** Every banded clause passes; two predictions are refuted and recorded,
two committed tests now assert a contract this step supersedes and are handed to the
coordinator untouched.

Files: `src/autoanim_gnm/commercial_multiview.py` (pass C, arms only),
`tests/test_arm_origin.py` (new), `tools/compare/delivered_vs_capture.py` (the arm-aim
floor), `tools/compare/d9_arm_delivery.py`, `tools/compare/d9_arm_silhouette.py`,
`tools/compare/d9_arm_gate.py`, `tools/compare/extractors/d9_arms.py`.
Reports: `artifacts/compare/d9-arms/gate.json`, `.../delivered-vs-capture.json`,
`.../silhouette-partwise.json`, `.../delivery-hygiene-build.json`, `.../delivery-build.json`,
`artifacts/compare/scoreboard-d9-arms.json`.

---

## 0. The pre-registration, restated verbatim

From `docs/LADDER_EXECUTION_PLAN.md` section 2, the D9 card, written 2026-09-05 and frozen
before any number below existed. It is reproduced in `d9_arm_gate.py`'s `PRE_REGISTRATION`
and copied into `gate.json` on every run.

> **floor, measured 2026-09-05 from the files for the ACTUAL operation** (elbow on the ray
> from the real UpperArm origin at rest length, wrist on the ray from that placed elbow):
> elbow 5.5 / 5.7 (performer 0 L / R) and 8.8 / 6.7 (performer 1) mm median, p95 20–32;
> wrist chained 7.3 / 9.2 and 10.4 / 11.9, p95 22–36; the window (85–125) 4–15. **Frame 118
> is length-limited, not aim-limited:** the captured upper arm there is 333 mm against a
> rest of 277 and the captured shoulder width 270 against the performer's 361 — a D8
> leftover on the yellow — so the floor there is 36 / 41 mm and D9's visible gain on that
> frame is partial · **instrument first:** `delivered_vs_capture.py` on the D8 delivery
> (reference = its own smoothed points) must reproduce 11 / 14 / 15 (performer 0 L:
> UpperArm / elbow / wrist) and the per-frame floor above before any src change · **B1
> (placement; the candidate optimises it, so paired with the floor and the photographs):**
> elbow and wrist medians within 3 mm of the real floor on both performers and both sides,
> whole take; bent-tercile and window cuts reported; predicted elbow 14 → 6–9, wrist 15–20 →
> 7–12; must-fail: the shipped D8 build and a root-translation degenerate that zeroes a
> wrist (B2 catches it) · **B2 (untouchable):** root, Hips, Spine/Chest/UpperChest, Neck,
> Head, Shoulder (clavicle) and every LEG local bit-identical to D8; the UpperArm-origin
> miss UNCHANGED (10–13 mm; it is D5's); the leg dry run REPORTED from the closed form
> (aiming the thigh from its own origin would move the knee 3.1–4.2 mm median, p95 9–26) and
> NOT shipped · **B3 (the photographs):** part-wise silhouette, ARMS, whole take and the
> window, not worse on either performer with the CI clear on the D7b/D8 predicate (upper
> bound ≥ 0), improvement predicted on both · **B4:** D3 closure ≤ 1e-6 m and
> same-denominator PASS; head gate rerun (predicted byte-equal figures: it reads landmarks
> and the head solve, neither touched); canonical round trip arms reported before / after
> (0.55 / 0.08 may close toward zero: the re-solve is exact by construction) · **B5
> synthetic (SOMASKEL77 posed clips, I7 noise):** elbow / wrist placement of
> aim-from-own-origin vs landmark-direction under an injected clavicle-length mismatch of
> ±30 mm; clean arm reaches the length floor to 1e-6; oracle: with the clavicle exactly
> right the two aims agree to 1e-9 (must hold, or the change did more than aiming) · **MAMMA
> per joint, window and whole take: report only** · **merge rule, fixed before numbers:** B1
> on both performers and both sides AND B2 exact AND B3 on both performers AND B4's closure
> and denominator clauses; the rest reports

---

## 1. The instrument first, and the floor reproduced

Run on the **shipped D8 delivery**, read only, reference = its own smoothed points:

| performer 0, left | card | measured |
|---|---|---|
| `UpperArm` vs `left_shoulder` | 11 | **10.93** |
| `LowerArm` vs `left_elbow` | 14 | **13.74** |
| `Hand` vs `left_wrist` | 15 | **14.67** |

The floor — the new `arm_aim_floor` block in `delivered_vs_capture.py` — reproduces every
figure the coordinator measured:

| | elbow L / R | wrist L / R | elbow p95 L / R | wrist p95 L / R |
|---|---|---|---|---|
| performer 0, card | 5.5 / 5.7 | 7.3 / 9.2 | 20–32 | 22–36 |
| performer 0, measured | **5.50 / 5.75** | **7.27 / 9.25** | 20.1 / 27.6 | 22.4 / 33.9 |
| performer 1, card | 8.8 / 6.7 | 10.4 / 11.9 | 20–32 | 22–36 |
| performer 1, measured | **8.77 / 6.66** | **10.41 / 11.87** | 31.9 / 30.8 | 35.8 / 35.2 |

Frame id **118** (array index **58**; the take's own ids run 60..209, so `id − 60` is the
index — `d8_occlusion_silhouette.py` carries the same two constants) on performer 1's left
arm: floor **36.22 / 41.42 mm**, captured upper arm **332.5 mm** against a rest bone of
**277.3 mm**. Length-limited, exactly as the card says.

One small departure from the card's text: the WINDOW figures read 3.58–17.75 mm, not
"4–15". The elbow window medians are 4.00 / 10.79 and 9.66 / 14.52; the wrist window
medians are 3.58 / 10.92 and 15.01 / 17.75. The card's range covers the elbow and clips
the wrist at both ends. Recorded, not a band.

---

## 2. The defect, and the change that closes it

Pass C built each arm bone's direction from two LANDMARKS — `at("left_elbow") −
at("left_shoulder")` — and handed it to a rotation that turns the bone about the rig's own
`LeftUpperArm` **origin**. Those are two different points. D2 aims the clavicle from its own
pivot and that is right, but the clavicle has a **fixed rest length**, so its child's origin
cannot in general reach the captured shoulder: 10.9 / 11.0 mm median on performer 0 and
10.2 / 12.8 on performer 1, read from the delivered file's own bytes. A direction measured
from a displaced origin misses everything below it by the displacement, and the delivered
elbow and wrist carried exactly that.

This is **D7b's mechanism one chain further out** — there the `Spine` origin rode the pelvis
pitch and the delivered neck went 14 → 59 mm off its landmark — and it is D7b's fix. In
pass C, arms only, both sides:

```
direction = at(target_name) - _joint_origin(world, frame, root_translation, rest, joint_name)
```

with the origin computed **inside** the loop, per bone, so `LowerArm`'s origin is read after
`UpperArm`'s world has been written. That ordering is the D2c lesson: a placed parent, then
the chain below it re-solved. Hands and fingers inherit. **The legs are deliberately
unchanged** and stay landmark-to-landmark; §7.3 carries their dry run.

No constant, no window, no provenance entry, nothing from MAMMA. The whole change is one
loop split in two.

---

## 3. Every band, with its number and its verdict

| band | what it asks | number | verdict |
|---|---|---|---|
| floor | reproduce the card's floor to 0.1 mm; D9's floor equals D8's exactly | all eight match; floors bit-equal | **PASS** |
| **B1** placement | elbow and wrist within **3 mm** of the real floor, both performers, both sides, whole take | worst gap **1.59 mm** | **PASS** |
| **B2** untouchable | root, Hips, trunk, Neck, Head, both clavicles, every leg local bit-identical; the `UpperArm` miss unchanged; the leg dry run reported | every banded array bit-identical; every non-arm joint unchanged from the file to 1e-9 mm | **PASS** |
| **B3** photographs | arms not worse, whole take AND window, both performers, CI upper bound ≥ 0 | all four cells positive; performer 1's window +0.0132 with the interval clear of zero | **PASS** |
| **B4** closure etc. | D3 closure ≤ 1e-6 m; same denominator PASS | 4.69e-7 / 4.71e-7 m; denominator PASS on both arrays | **PASS** |
| **B5** synthetic | clean arm reaches its length floor to 1e-6 m; oracle agrees to 1e-9 | worst gap **4.75e-7 m**; oracle gap **2.22e-16** | **PASS** |
| hygiene | today's code reproduces the shipped 8 files byte-identically | 8 / 8 | **PASS** |

**Merge rule (B1 ∧ B2 ∧ B3 ∧ B4): MERGE. Failed clauses: NONE.**

### B1 in full, from the delivered files' own bytes

Median millimetres against the delivery's own smoothed landmarks. The floor is bit-equal on
both arms, because D9 does not move the `UpperArm` origin.

| | D8 | **D9** | floor | gap | D9 − D8, 95 % block CI |
|---|---|---|---|---|---|
| performer 0, left elbow | 13.74 | **7.05** | 5.51 | 1.54 | −3.94 [−6.32, −2.26] |
| performer 0, right elbow | 13.91 | **6.81** | 5.75 | 1.06 | −2.79 [−8.28, −0.70] |
| performer 0, left wrist | 14.67 | **8.86** | 7.27 | 1.59 | −3.89 [−7.95, −1.67] |
| performer 0, right wrist | 15.61 | **10.39** | 9.25 | 1.14 | −2.88 [−7.28, −0.75] |
| performer 1, left elbow | 13.44 | **9.11** | 8.77 | 0.34 | −2.42 [−10.43, −1.16] |
| performer 1, right elbow | 15.09 | **8.09** | 6.66 | 1.42 | −4.46 [−7.55, −1.77] |
| performer 1, left wrist | 17.59 | **10.90** | 10.41 | 0.49 | −4.81 [−11.05, −1.73] |
| performer 1, right wrist | 20.09 | **11.87** | 11.87 | 0.00 | −5.11 [−7.58, −1.74] |

Eight of eight improve with the interval clear of zero. Eight of eight land within 1.6 mm
of a floor no aim can move.

### B2 from the file, every joint this step must not touch

Bit-identical track arrays on both performers: `root_translation_m`, `foot_contacts`,
`ticks`, `rest_translations_m`, `joint_names`, and the LOCAL rotations of `Hips`, `Spine`,
`Chest`, `UpperChest`, `Neck`, `Head`, both eyes, **both clavicles**, both `UpperLeg`,
both `LowerLeg`, both `Foot`, both `Toes`, and every finger. And from the delivered file:
`LeftUpperArm` 10.933 / 10.248 mm, `Neck` 18.589 / 15.440, `LeftLowerLeg` 8.521 / 11.477,
`hips_joint_vs_hip_midpoint` 80.0 — every one identical to D8 to below 1e-9 mm.

The four arm bones DID move on both performers, which is the half of B2 that stops the
list proving nothing.

The two `Hand` locals are **not** bit-identical and cannot be: `_set_world` gives a hand its
parent's world, so its local is a quaternion composed with its own inverse and the rounding
follows the parent. The worst component gap is **1.11e-16** and the delivered local is the
identity to within it. Reported, never banded.

---

## 4. The degenerates, and which band catches which

* **The shipped D8 build.** Its own elbows and wrists sit 4.7–8.4 mm above their own floor
  on all eight cells — every one outside the 3 mm band. B1 rejects it.
* **A root translation that zeroes a wrist.** Translate the root, per frame, by
  `left_wrist − LeftHand_origin` (forward kinematics on the D9 track; nothing exported, the
  D2 "translate the same rendered body" precedent). It scores a **perfect** left wrist —
  0.000135 mm median — and B1 alone would wave it through. B2 catches it: its root is not
  bit-identical to D8 by construction, and its hips move 9.2 / 12.5 mm, its knees 9.5 / 15.3
  and its ankles 14.4 / 16.8, against a D9 build whose hip reads 4.0 / 5.0. Its **right**
  wrist also goes to 11.6 / 21.9 mm — a degenerate that buys one landmark with all the
  others.
* **The wrong-sign control in B5.** The clavicle mismatch is injected at **both** +30 mm and
  −30 mm, because an aim that only works on one sign is not an aim. The clean arm reaches
  its length floor on every clip at both signs.

---

## 5. What each instrument here is blind to

* **`delivered_vs_capture.py`** scores the SKELETON in the delivered file against *our own*
  triangulated landmarks. It cannot see orientation about a joint that is already on its
  landmark; it cannot see the mesh; a common-mode detector error is inside its reference; and
  it cannot say whether a change is right *in the world* — only whether the delivered
  skeleton sits closer to the points that delivery was solved onto. **B1 is a band the
  candidate optimises directly**, which is why it is quoted only beside the floor (which no
  aim can move and which is bit-equal across the arms) and the photographs (which the
  candidate cannot optimise).
* **The arm-aim floor** is a floor on PLACEMENT. A bone that reaches its landmark can still
  be rolled about its own axis, and nothing here sees that.
* **The part-wise silhouette** scores the MESH. The elbow and wrist move by 3–7 mm and an
  arm can move that far *inside its own outline* without moving a pixel — so a level reading
  is this instrument's resolution, not evidence against B1. It is also blind to depth, to a
  left/right mirror of a fore-aft symmetric pose, and to where inside the outline a limb
  sits; and it reads a mesh bound to an asset whose proportions are not the performer's, so
  every figure is a CHANGE against a large standing shape mismatch. Precision and recall are
  printed beside every IoU for the dilated-blob pattern.
* **The canonical round trip** re-solves the arms from landmarks its own first pass emitted,
  and after this change those landmarks are the FK'd joint origins the second pass aims
  from — so on the arms it is now closed **by construction**. Its fall from 0.55 → 0.17 and
  0.08 → 0.02 mm is a property of the instrument and is reported, never claimed as evidence.
* **`retarget_cost.py`** re-solves the track through the converter and is structurally blind
  to what the exporter wrote (D7b's finding). Unchanged, and still labelled.
* **MAMMA's per-joint scoreboard** measures AGREEMENT with an instrument, not accuracy. It
  reports; it selects nothing, and nothing from it enters src or a constant.

---

## 6. The reported arms

### 6.1 The photographs, part-wise, arms only (B3)

IoU against MAMMA's SAM2 person masks, median over the frames every camera scored, paired
moving-block bootstrap, block 15, 2000 draws, seed 20260905, identical draws for both arms.
Both landmark arrays byte-identical across the arms; MAMMA's own mesh oracle agrees with the
committed unsplit run to 0.0.

| performer | cut | n | D8 | **D9** | D9 − D8 [95 % CI] | MAMMA mesh |
|---|---|---|---|---|---|---|
| 0 | whole take | 150 | 0.20675 | **0.20922** | +0.00247 [−0.00075, +0.00600] | 0.24293 |
| 0 | window 85–125 | 41 | 0.11282 | **0.11336** | +0.00055 [−0.00265, +0.00447] | 0.15055 |
| 0 | bent tercile | 50 | 0.21749 | **0.22360** | +0.00610 [+0.00127, +0.01046] | 0.26074 |
| 1 | whole take | 150 | 0.21507 | **0.21587** | +0.00079 [−0.00138, +0.00628] | 0.24900 |
| 1 | window 85–125 | 41 | 0.15716 | **0.17037** | +0.01322 [+0.00081, +0.02334] | 0.22240 |
| 1 | bent tercile | 50 | 0.22316 | 0.22197 | −0.00120 [−0.00259, +0.00212] | 0.25091 |

All four banded cells pass. Three cells have intervals clear of zero, all positive:
performer 0's bent tercile and everything outside the window, and performer 1's window —
which is D8's occlusion stretch, where the captured arm was worst. Performer 1's bent
tercile is level within its interval and is reported. On performer 1's window the arm's
**precision rises 0.708 → 0.773 with recall also rising 0.169 → 0.180**, so it is not the
dilated-blob pattern. The torso is unmoved to ±0.0002 IoU everywhere, as B2 requires.

### 6.2 MAMMA per joint, report only

Through `tools/head/subject_map.py`; MAMMA's `body_id-00` is **our subject 1** on this
fixture, and both scoreboards resolve it that way. Delivered column, median mm:

| joint | performer 0 D8 → D9 | performer 1 D8 → D9 |
|---|---|---|
| left elbow | 36.9 → 37.7 | 63.5 → 63.1 |
| right elbow | 21.8 → **18.4** | 37.4 → **36.0** |
| left wrist | 29.0 → **24.2** | 37.3 → **32.7** |
| right wrist | 27.2 → **24.9** | 34.0 → **31.5** |
| left shoulder | 30.8 → 30.8 | 46.5 → 46.5 |
| neck | 89.5 → 89.5 | 92.6 → 92.6 |
| left knee | 26.1 → 26.1 | 41.0 → 41.0 |

Seven of eight arm cells move closer to MAMMA; performer 0's left elbow moves 0.8 mm
further. Every joint outside the arms is identical, which is the check that says the
scoreboard is reading the same two builds. **It selects nothing.**

### 6.3 The head gate, rerun

Reran `tools/head/head_gate.py` and compared its report before and after. **Every figure is
byte-equal.** The single difference is `absolute_facing_not_a_band.source`, an absolute path
(the committed run was made from the main checkout, this rerun from the worktree); it is
excluded by name in the gate and no figure is normalised. Both reports are kept under
`artifacts/compare/d9-arms/head-gate-shipped-{before,after}-d9.json` (`artifacts/` is gitignored), and
`artifacts/head-lane/head-gate-shipped.json` was restored to the committed version so the
shared artifact keeps the main checkout's provenance. The prediction — byte-equal, because
the gate reads the head solve and the landmarks and neither is touched — holds.

### 6.4 The canonical round trip, before and after

Arms 0.55 → **0.17** mm on performer 0 and 0.08 → **0.02** on performer 1; legs and torso
0.00 on both, before and after. The card predicted it "may close toward zero: the re-solve
is exact by construction", and it did. See §5 for why this is not evidence.

### 6.5 B5, synthetic truth

Five SOMASKEL77 posed clips, MAMMA-free. Both arms run the REAL converter: the D8 arm is
reached by substituting `_joint_origin` so `*UpperArm` reports the captured shoulder and
`*LowerArm` the captured elbow — the shipped D8 directions at the identical call site, never
a re-implementation. Measured **before** the ground projection, for the reason in §7.1.

* **Clean, ±30 mm clavicle mismatch:** the aim-from-own-origin arm reaches its length floor
  on every clip, both signs, both sides — worst gap **4.75e-7 m** against a 1e-6 band.
* **D9 beats D8 on the elbow in 20 of 20 clean cells**, by 0.1 to 40.4 mm, and on the wrist
  in **18 of 20**. The two exceptions are on `cpu_smoke/autoanim_fixture`, and they are
  honest: see §7.4.
* **Under I7's own measured noise** (three seeds, pooled, +30 mm mismatch): elbow 13.2 vs
  16.5, 27.3 vs 29.8, 44.5 vs 54.0, 44.0 vs 51.8, 17.2 vs 22.1 mm (D9 vs D8); wrist 10.7 vs
  22.0, 27.3 vs 41.7, 22.7 vs 62.3, 18.9 vs 64.3, and 76.1 vs 73.8.
* **The oracle:** on a fixed-point fixture where the captured shoulder IS the `UpperArm`
  origin and the arm bones are at the rig's own lengths, the two aims are handed identical
  directions and the tracks agree to **2.22e-16** of a quaternion component, against the
  card's 1e-9. `tests/test_arm_origin.py` carries the same construction as a unit test.

---

## 7. What the coordinator should know

### 7.1 The delivered elbow does NOT sit exactly on its floor frame by frame, and the reason is the ground projection

A prediction made during the pass — that after D9 the delivered elbow's miss would EQUAL the
floor to storage precision on every frame — is **refuted at the frame level and holds at the
median**. The per-frame excess has median 0.0 mm and p95 1.4–6.8 mm, with a worst frame of
17.8 mm.

The cause is measured, not assumed. `positions_to_body_track` ends in
`project_generated_foot_contacts`, which **translates the root after every aim has been
taken** (`maximum_root_correction_m=0.08`); D7b's B7 records the same fact for the trunk. A
translation leaves every rotation alone, so the delivered bone still points along
`elbow − origin_before_the_hoist` while the gate measures the ray from the origin after it.

The gate recovers that hoist from the delivered file alone and the recovery is
**over-determined**: each of the four arm bones contributes two linear equations in the same
three unknowns, so if one rigid translation explains them all the residual is zero. It does:
residual **1.2e-4 mm at worst**, hoist median 0.0002 / 0.0001 mm, p95 12.54 / 8.79 mm, max
19.71 / 9.68 mm, and **67 of 150** frames on performer 0 and **19 of 150** on performer 1
carry a hoist above 0.5 mm (`gate.json` →
`B1_placement.the_ground_projection.figures.*.recovered_root_shift_mm`). That is the second
independent measurement the lane's rule asks for before a number is explained: a first fit
that included the LEGS left 5.8 / 13.2 mm of residual and was discarded, because the legs
are aimed landmark-to-landmark and do not satisfy the constraint the arms do. It is present on the D8 build identically and it is
not this step's to fix.

### 7.2 Two committed tests now assert a contract this step supersedes — they were NOT edited

Both are red on `ladder/D9` and green on D8. Neither was touched, per the brief.

1. **`tests/test_clavicle_origin.py::test_joint_origin_as_the_converter_actually_calls_it`**
   asserts `set(recorded) == set(CLAVICLES)` — that `_joint_origin` is called *only* for the
   two clavicles. D9 calls it for the four arm bones as well, which is the step. The
   assertion is a statement of D2's scope, and D2's own claim (the origin used is the one the
   rotation turns about) is unchanged. Suggested replacement: assert the recorded set is
   `CLAVICLES | ARM_BONES`, keeping the rest of the test as it is.
2. **`tests/test_clavicle_temporal.py::test_a_single_frame_pop_is_rejected_and_the_arm_is_re_solved_onto_its_landmarks`**
   asserts that on the replaced frame `LeftUpperArm → LeftLowerArm` points along
   `elbow − shoulder`, the LANDMARK direction, to 0.05°. Under D9 it points from the
   `UpperArm` origin instead, and on that fixture's replaced frame the two differ by 20.3°.
   **The property the test protects is not lost — it is improved.** On that same frame,
   measured through the same code path: the delivered elbow sits **26.9 mm** from the
   captured elbow under D9 against **99.9 mm** under D8, and the wrist **35.1 mm** against
   **103.0 mm**; whole-take medians 6.1 vs 11.6 and 9.2 vs 10.8. The arm is re-solved, not
   swung rigidly; the test measures the old idiom (direction) rather than the claim
   (placement). Suggested replacement: assert the delivered elbow and wrist land on their
   landmarks to the bone's own length floor on the replaced frame.

Four further failures in the full suite (**1161 passed, 6 failed, 16 skipped**) are
**pre-existing on D8** and unrelated: `test_body_compositor.py::test_unified_preview_...`,
`test_body_export.py::test_export_animated_body_glb_...` (the uncommitted-file item already
recorded in `LADDER_STATUS`), `test_phase4_app.py::test_home_and_health`, and
`test_retarget_cost.py::test_converter_rotations_depend_on_bone_lengths_only_through_the_clavicle`.
Verified by stashing the src change and rerunning those six.

### 7.3 The leg dry run — reported, not shipped

Closed form from the D9 delivery's own bytes: if the thigh were aimed from its own FK origin
at the captured knee, the delivered knee would move

| | median | p95 | max | delivered knee error → would become |
|---|---|---|---|---|
| performer 0 left | 3.09 | 9.23 | 16.8 | 8.52 → 6.88 mm |
| performer 0 right | 3.05 | 11.05 | 19.1 | 6.56 → 4.58 mm |
| performer 1 left | 3.46 | 25.96 | 36.8 | 11.48 → 8.94 mm |
| performer 1 right | 4.25 | 26.09 | 36.7 | 15.90 → 12.32 mm |

The card said 3.1–4.2 median, p95 9–26. Measured 3.05–4.25 and 9.2–26.1 — reproduced, with
performer 0's left knee 0.01 mm below the stated range. The legs are spared the arm's defect
because D2b puts the leg-root **midpoint** on the captured hip midpoint: the delivered
`UpperLeg` origin sits 3.7–5.1 mm from its own hip landmark, against the arm's 10–13 mm.
**Not shipped**, and `tests/test_arm_origin.py` asserts the legs are still aimed
landmark-to-landmark so a later change cannot arrive unnoticed.

### 7.4 Refuted predictions

1. **"Predicted elbow 14 → 6–9 mm"** — holds on three of four arms (7.05, 6.81, 8.09) and
   fails on performer 1's left elbow at **9.11 mm**. It could not have held: that arm's own
   floor is 8.77 mm, so no aim can reach 9.00. The band — within 3 mm of the floor — passes
   there at **0.34 mm**, the best margin of the eight. **The band is not moved**; the
   prediction is recorded as refuted, and the lesson is that a prediction written in absolute
   millimetres is in tension with a floor that varies by arm.
2. **"The delivered elbow will sit on its floor frame by frame"** — refuted; §7.1.
3. **"With the clavicle exactly right the two tracks will be bit-identical"** — refuted, and
   the card's own 1e-9 form holds. The fixed-point fixture converges to ~1e-18 m rather than
   exactly, and the hands' locals round with their parent, so the tracks agree to 2.22e-16
   instead of bitwise. This is the D7b hazard read the other way round: 1e-9 was **reachable**
   here because the quantity is a quaternion component and not an angle at the float32 floor.
   Stated in the test's own docstring so nobody tightens it later.
4. **Aiming from the origin does not reduce the wrist error on every body.** On B5's
   `cpu_smoke/autoanim_fixture` clip the D9 wrist is worse than D8's (62.5 vs 57.0 and 61.9
   vs 55.0 mm clean; 76.1 vs 73.8 noisy) while its elbow is essentially exact (0.14 / 0.91
   mm). That clip's forearm length mismatch is enormous — the wrist floor there is 62.5 mm —
   and a **displaced parent can accidentally compensate for a bone of the wrong length**. D9
   reaches its floor exactly on that clip too; it is the floor that is high. On the real
   delivery all four wrists improve. This is the honest limit of the claim: D9 makes the aim
   correct, not the arm the right length, and where length dominates the answer can be worse
   in absolute millimetres.

### 7.5 Assumptions taken without asking, as the brief directs

* The window figures in the floor block are reported as measured (3.58–17.75 mm) rather than
  forced into the card's "4–15"; §1.
* B3's primary cut is the **whole take**, with D8's window and D7b's tilt terciles beside it.
  The card names "whole take and the window" and both are banded; the terciles report.
* `mamma_scoreboard.py` was run on the D9 rebuild with `--label d9-arms`, writing
  `artifacts/compare/scoreboard-d9-arms.json` beside the committed D8 one. Report only.
* `artifacts/head-lane/head-gate-shipped.json` was rerun and then **restored** to its
  committed bytes; the rerun is kept under `artifacts/compare/d9-arms/`. §6.3.
* `arm_aim_floor` reads the delivery's **smoothed** landmarks whatever `--reference` is
  given, exactly as `trunk_length_floor` has since D7b. Under `--reference raw` the joint
  rows are scored against the raw array and the floor against the smoothed one; they are the
  same points wherever a landmark triangulated, and the difference is stated here rather
  than left for the next reader of that mode to discover.
* No `ladder.py` edit, no `status.py` call, no publish — the brief reserves all three for the
  coordinator. The extractor is a stub with its `VISUALS` block and a self-check that reports
  `VISUALS keys missing from the extractor: NONE`.

### 7.6 What the coordinator must do

1. Register `from extractors.d9_arms import x_arm_origin` in `tools/compare/ladder.py` and
   route the `arm_*` keys to rung 7 and the `silhouette_*` keys to rung 1. Until then the
   rung prints `NO VISUAL`.
2. `status.py set D9 done --report artifacts/compare/d9-arms/gate.json`, then `ladder.py`,
   then republish the three pages.
3. Decide the two superseded tests in §7.2 — they need an owner, and this branch may not
   edit them.
4. On merge, rebuild the delivery in place and rerun the instruments, as D7b and D8 did. The
   rebuild here wrote **only** to `artifacts/compare/d9-arms/`; the shipped delivery was
   never written, and the hygiene arm proves today's code reproduces its 8 files
   byte-identically (`subject-00.glb` `84263800485b6224…`, `subject-01.glb`
   `0448e330d6c86b78…`, and the six others in `delivery-hygiene-build.json`).
5. **Handed to D5**, unchanged by this step: the `UpperArm` origin's own 10–13 mm miss —
   8.4 / 7.9 mm of trunk chord plus 6.2 / 7.3 mm of missing shoulder translation — and the
   arm's LENGTH error, which is what the floor is made of and what now dominates every arm
   figure on the delivery.
