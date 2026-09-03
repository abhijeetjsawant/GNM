# D2: the clavicle origin — the direction measured from the joint that carries it

Branch `ladder/D2`, from `461e743`. One part changed: where the clavicle's direction is
measured **from**. Nothing else in the converter moves.

---

## 0. Pre-registration — written and committed BEFORE the numbers existed

This section is committed on its own, ahead of the fix and ahead of the gate, so that the
git history is the proof that these were predictions and not descriptions. Everything
below this line was written with no measurement in hand beyond the committed
`artifacts/compare/retarget-cost.json` (arms 41.57 / 47.05 mm, legs 0.00, on the canonical
round-trip oracle) and a reading of the source.

**P1 — the arms drop to a band, not merely improve.** The canonical round trip's six arm
landmarks (`GROUPS["arms"]` in `tools/swap-harness/retarget_cost.py` — both shoulders,
both elbows, both wrists; the older plan text says four and is wrong) must each land at
**≤ 5 mm median on both subjects**. Legs and torso must stay at **0.00 exactly**.
36 → 30 mm is a failure, not a partial success.

**P2 — the remainder is not zero, and here is the mechanism, derived before measuring.**
On the way back the round trip re-estimates the torso frame's secondary axis
(`shoulder_across`) from the **UpperArm origins**, because `landmarks_from_fk` writes the
reference FK's `LeftUpperArm`/`RightUpperArm` origins into the `left_shoulder` /
`right_shoulder` slots. The reference solve built its torso frame from the **capture's
shoulder landmarks**. Those two across-vectors differ whenever the two clavicle vectors
have unequal length or unequal foreshortening, so the torso **roll about `torso_up`**
differs by a fraction of a degree.

The rest of the torso chain is roll-invariant: every rest offset from `Hips` up to `Neck`
is a pure `+Y` vector, and `_frame_alignment` sends `+Y` to `unit(torso_up)` whatever the
roll, so `Spine`, `Chest`, `UpperChest`, `Neck` origins do not move (which is why the
torso group can stay at 0.00). `LeftShoulder`'s rest offset is `(0.11, 0.10, 0)`; the
`0.11` in X is the only part that turns with the roll. So a roll difference δ displaces
the shoulder origins by ≈ 0.11·δ — 1–3 mm for δ under a degree.

The brief predicts the arm residual **equals** that displacement. It does not, exactly,
and the corrected form is registered here. Write `d` for the root-relative displacement of
the FK `LeftShoulder` origin between the reference solve and the round-trip solve, and `ê`
for the unit upper-arm direction. In the round trip `‖landmark − origin_ref‖ = 0.16 m`
exactly (the landmark *is* the reference's UpperArm origin, one rest length out). Then the
round trip places the UpperArm origin at `origin_ref + d + 0.16·unit(e − d)` with
`e = 0.16·ê`, and to first order in `d`

&nbsp;&nbsp;&nbsp;&nbsp;`residual ≈ (d · ê) ê`,&nbsp;&nbsp;so&nbsp;&nbsp;`‖residual‖ ≤ ‖d‖`.

**The residual is the component of the shoulder-origin displacement along the upper arm**,
not the displacement itself. The transverse part of `d` is absorbed by re-aiming the
clavicle, which is precisely what the fix makes it free to do.

**P2b — the elbow and the wrist carry the same vector, exactly.** The forearm and hand
chains are measured landmark-to-landmark and are untouched, and in the round trip
`‖elbow − shoulder‖` is exactly the `LeftLowerArm` rest length, so the whole arm below the
UpperArm origin is a **parallel offset** of the reference arm. Prediction: shoulder, elbow
and wrist medians agree to the last printed digit, before and after. (The committed
before-report already shows this: 36.37 / 36.37 / 36.37 on subject 0.)

Gate obligation: measure `‖d‖`, `‖(d·ê)ê‖` and `‖residual‖` per frame and report the
agreement. **If the residual does not match the projection, the fix has a defect** and
that is what the report will say.

**P3 — arm B (MAMMA `pred_joints` in) drops in step.** Same mechanism, and its input is a
rigid SMPL-X skeleton, so if anything it should be cleaner. Before: 21.96 / 36.33 mm on
`ORACLE_roundtrip_from_mamma_solve`. Reported before/after on its own reference; MAMMA
reports and never selects, and nothing in this step reads a MAMMA number to choose
anything.

**P4 — rung 11 `canon` may NOT drop, and may get worse at the shoulder.** The canonical
rig's half shoulder span is 110 mm of clavicle plus 160 mm of upper arm against ~175 mm
measured on these performers. Aiming the clavicle correctly on a rig that is too wide puts
the arm root further out laterally while lifting it less. **A flat or slightly worse canon
figure is NOT a failure of this step; it is the proportion mismatch (D5).** The drop should
show on the SIZED arms (rung 7 `sized_arms_*`, rung 11 `sized_*`) and on arm B.

**P5 — scale invariance breaks by design.** Today the converter returns *identical*
rotations on a canonical and a sized skeleton (`converter_is_scale_invariant` in the I1
report, max rotation difference 5.6e-16), because every rotation aims a rest **direction**
and sizing scales rest offsets without turning them. After D2 the clavicle direction is
measured from the **sized rig's own origin**, which sizing does move, so
`performer_sized_resolved` ≠ `performer_sized_fk_only_scoreboard_method` and the
scoreboard's `sized` arm (canonical rotations replayed on a sized skeleton) stops being
equivalent to a re-solve. Both are reported, before and after. The scoreboard's method is
not changed; the difference is described.

**P6 — the differing-joint set in the bit-identity theorem, corrected before measuring.**
The brief says the local rotations that change are the two clavicles "and their
descendants (… and the finger joints under it)". That is wrong about the fingers and
probably wrong about the hands, and the corrected prediction is registered here:

* the finger locals are `_finger_rest_local(joint.name)`, a **constant** per joint name, so
  their *local* rotations are bit-identical before and after (their *world* rotations do
  move — the fingers ride the arm);
* `LeftHand` / `RightHand` are set to their parent's world, so their local is
  `inv(world[parent]) · world[parent]` — either exactly identity-bit-equal or different at
  the 1e-16 rounding level, and it is not predictable which.

Registered prediction: the set of joints whose `local_rotations_xyzw` differ **contains**
{Left,Right}×{Shoulder, UpperArm, LowerArm} and is **contained in** that set plus
{LeftHand, RightHand}. Torso, head, eyes, legs, feet, toes and every finger are identical.

**P7 — twist is the gate's blind spot, and no instrument in this lane scores it.**
`_world_for_bone` is the minimal rotation from the parent's frame, so the upper arm,
forearm and hand inherit the clavicle's roll. A 15–19° change in clavicle direction changes
arm twist by some fraction of that, and the delivered hand orientation with it. Nothing
here measures it. The only instrument for it is a picture, and a picture is what will be
produced.

---

## 1. What shipped

One expression, twice, in `src/autoanim_gnm/commercial_multiview.py`:

```python
("LeftShoulder",  "LeftUpperArm",  at("left_shoulder")  - left_shoulder_origin),
("RightShoulder", "RightUpperArm", at("right_shoulder") - right_shoulder_origin),
```

where the origins come from a new module-level helper, `_joint_origin(world, frame,
root_translation, rest, joint_name)`, which walks Root → Hips → Spine → Chest → UpperChest
→ Shoulder using **exactly** `autoanim_gnm.body.forward_kinematics_positions`'s recursion
and the caller's own `rest` dict, so a patched or per-performer-sized skeleton is honoured.
It is module level and called by bare name so an instrument can substitute it and run every
control through the identical code path.

The constant `0.72` **leaves and nothing replaces it.** The origin is the rig's own
geometry. No reference — MAMMA's least of all — enters it.

## 2. The mechanism

`_world_for_bone` turns a bone about the rig's own Shoulder origin, which sits at
`Hips + 0.12 + 0.16 + 0.15` (UpperChest) `+ (±0.11, 0.10, 0)` — 530 mm up and 110 mm
lateral on the canonical rig. The direction it was handed was measured from
`pelvis + 0.72·torso_up`, 418 mm up the torso axis and 110 mm inboard. **Two origins for
one direction.** Whatever the rest of the pipeline did, the arm root could not land on the
requested shoulder, and did not: 41.57 / 47.05 mm off on a body with canonical proportions
by construction.

Measuring from the point the rotation actually pivots about is right unconditionally. It
does not depend on the rig being well placed — and, as section 4 shows, this rig is not.

## 3. The pre-registered expectations, and whether each was met

| | prediction | outcome |
|---|---|---|
| **P1** | round-trip arms ≤ 5 mm on all six landmarks; legs and torso 0.00 | **NOT MET on the arms (67.25 / 79.32 mm, worse than before).** Legs and torso 0.00 exactly, under every variant. The band's *premise* was false — see §4. |
| **P2** | residual = `(d·ê)ê`, the shoulder-origin displacement projected on the upper arm | **MET, and only visible once §4's term is removed.** Under the corrected control: residual 0.604 mm vs projection 0.581 mm (subject 0), 0.222 vs 0.227 (subject 1) — agreement to **13 µm and 6 µm**. As shipped, the projection explains 0.58 of a 61.49 mm residual and the other 60.87 mm is §4. |
| **P2b** | shoulder, elbow and wrist carry the identical residual | **MET exactly.** After: 61.49 / 61.49 / 61.49 and 69.68 / 69.68 / 69.68 (subject 0); 79.76 ×3 and 77.87 ×3 (subject 1). The arm below the UpperArm origin is a rigid parallel offset. |
| **P3** | arm B drops in step | **MET on the delivered configuration** (200.32 → 130.52, 219.51 → 89.55 mm), and its round trip rises the same way ours does (21.96 → 66.42, 36.33 → 78.74), for the same reason: with the hip drop out of the re-solve it reads 4.36 / 1.68 mm. |
| **P4** | rung 11 `canon` may not drop, and may worsen at the shoulder | **MET, and stated in the plan's own words: a flat canon figure is NOT a failure of this step; it is the proportion mismatch (D5).** Subject 0's `canon` is unchanged at 151.58 mm (margin 0.00, CI [0.00, 32.17]); subject 1's improves 137.38 → 103.91. The **sized** arm improves on both, as predicted. |
| **P5** | scale invariance breaks by design | **MET.** Max per-joint rotation difference between a canonical solve and a re-solve on a sized rig: 0.045° before, 144.24° / 167.50° after, and confined to exactly the six clavicle-chain joints. |
| **P6** | differing locals ⊇ clavicle chain, ⊆ that plus the two hands; no finger local moves | **MET exactly.** Differing set on both subjects: `LeftShoulder, LeftUpperArm, LeftLowerArm, LeftHand, RightShoulder, RightUpperArm, RightLowerArm, RightHand`. Zero finger locals moved. The brief's version of this list was wrong about the fingers; the pre-registered correction was right. |
| **P7** | twist is the blind spot | **MET and now measured, not asserted** — §6. |

## 4. Why the round trip got worse, and why it is not the clavicle

**The brief's premise is false.** It said "the clavicle does not share [the legs'
root-placement] problem". It shares it exactly. What differs is that measuring from the
rig's own origin is right anyway, because that is where the bone pivots.

`retarget_cost.landmarks_from_fk` writes the rig's **UpperLeg origins** into the
`left_hip` / `right_hip` slots. That is anatomically right — those are the femoral joint
centres, which is what a `left_hip` landmark is. The converter then places **`Hips`** on
that midpoint, 80 mm higher. So the round trip's *second* solve sits 80 mm low against the
same landmark cloud, and the clavicle direction — the first direction in this converter
ever measured in the **rig's** frame rather than between two landmarks — picks up
`0.08·û_up` and turns by ~27°.

Every other chain is measured landmark-to-landmark and is blind to a translation; the
score is root-relative, another translation. **The legs reading 0.00 mm through all of this
was never evidence the root placement is right. It is evidence that a translation-invariant
chain scored root-relatively cannot see a translation.**

The attribution is a control, not an argument. Swap the origin helper for the **re-solve
only** — the delivered path untouched — so that it returns the origin the rig would have if
its leg roots, rather than `Hips`, sat on the hip-landmark midpoint (no constant: the offset
is the skeleton's own `LeftUpperLeg` rest translation):

| | as the instrument builds it | with the rig's own hip drop out of the re-solve |
|---|---|---|
| subject 0, arms | 67.25 mm | **0.60 mm** |
| subject 1, arms | 79.32 mm | **0.22 mm** |
| arm B, subject 0 | 66.42 mm | 4.36 mm |
| arm B, subject 1 | 78.74 mm | 1.68 mm |

**100 % of the round trip's rise is the root-placement convention — open D-lane work — and
0 % is the clavicle.** Applied to both passes it reads 0.51 / 0.08 mm.

The corollary is uncomfortable and should be said: **the pre-D2 figure of 36–47 mm was
never "the clavicle-origin cost" either.** It was the clavicle-origin cost *measured through
an instrument that dodges the root-placement offset because its anchor lived in the landmark
frame*. The oracle was miscalibrated in the direction that flattered the old code.

## 5. The gate

`tools/compare/d2_clavicle_gate.py` → `artifacts/compare/d2-clavicle/gate.json`.
Regenerate: `PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d2_clavicle_gate.py`.
Bootstraps are moving-block, block 15, 2000 draws, seed 20260902, **both arms on identical
drawn frames**; lag-1 autocorrelation on the scored series is 0.96 / 0.95, so ordinary
resampling would be invalid. The "before" per-frame series come from the legacy-anchor
**swap**, not the committed JSON (which carries only medians); the swap reproducing the
committed medians to 0.00 mm is what licenses that substitution.

| band | before | after | 95 % interval (before − after) | verdict |
|---|---|---|---|---|
| **A.** round-trip arms, all six landmarks ≤ 5 mm, subj 0 | 41.57 | 67.25 | [−31.17, −14.64] | **FAIL** |
| **A.** round-trip arms, all six landmarks ≤ 5 mm, subj 1 | 47.05 | 79.32 | [−37.42, −29.27] | **FAIL** |
| **A.** round-trip legs and torso = 0.00 exact, both subjects | 0.00 | 0.00 | — | PASS |
| **ATTRIBUTION.** round-trip arms, hip drop out of pass 2 ≤ 1 mm, subj 0 | 67.25 | 0.60 | — | PASS |
| **ATTRIBUTION.** same, subj 1 | 79.32 | 0.22 | — | PASS |
| **B(a).** legacy-anchor swap reproduces the committed report ≤ 0.01 mm, subj 0 | 41.57 | 41.57 | — | PASS |
| **B(a).** same, subj 1 | 47.05 | 47.05 | — | PASS |
| **B(b).** best of 17 on-axis scalars must still fail, subj 0 | — | 12.08 (at 1.00) | — | PASS |
| **B(b).** same, subj 1 | — | 10.62 (at 1.05) | — | PASS |
| **B(c).** the UpperChest origin must fail, subj 0 / subj 1 | — | 45.37 / 65.05 | — | PASS |
| **B(d).** legs = 0.00 under EVERY variant, both subjects | — | 0.00 | — | PASS |
| **DELIVERY.** arms on our own capture, canonical rig, subj 0 | 181.33 | 124.39 | [25.79, 94.88] | PASS |
| **DELIVERY.** same, subj 1 | 217.50 | 89.56 | [96.98, 156.90] | PASS |
| **TEMPORAL.** clavicle-chain frames over a human's peak joint rate, subj 0 | 32 | 48 | — | **FAIL** |
| **TEMPORAL.** same, subj 1 | 11 | 49 | — | **FAIL** |
| **TEMPORAL.** legs, neck, head step-angles bit-identical, both | — | identical | — | PASS |
| **C(arm B).** MAMMA's joints in, canonical arms, subj 0 | 200.32 | 130.52 | [34.21, 103.95] | PASS |
| **C(arm B).** same, subj 1 | 219.51 | 89.55 | [108.47, 164.75] | PASS |
| **C.** bit-identity theorems | — | held | — | PASS |
| **E.** scoreboard `capture` arm identical, both subjects | — | 0.00 | — | PASS |
| **E.** scoreboard `canon`, all joints, subj 0 | 151.58 | 151.58 | median 0.00, [0.00, 32.17], p(wrong sign) 1.00 — **unchanged, not improved** | PASS (P4) |
| **E.** scoreboard `canon`, all joints, subj 1 | 137.38 | 103.91 | median 29.54, [0.00, 67.35], p(wrong sign) 0.06 — **interval touches zero** | PASS |
| **E.** scoreboard `canon`, six arm joints, subj 0 | 231.21 | 186.93 | [20.70, 77.73] | PASS |
| **E.** scoreboard `canon`, six arm joints, subj 1 | 241.47 | 117.27 | [91.57, 143.50] | PASS |
| **E.** scoreboard `sized`, all joints, subj 0 | 136.58 | 110.06 | [4.94, 41.36] | PASS |
| **E.** scoreboard `sized`, all joints, subj 1 | 113.04 | 88.93 | [5.11, 35.96] | PASS |
| **E.** scoreboard `sized`, six arm joints, subj 0 | 144.54 | 117.85 | [5.95, 48.49] | PASS |
| **E.** scoreboard `sized`, six arm joints, subj 1 | 139.70 | 98.16 | [12.08, 64.72] | PASS |

**Read the two `canon` rows carefully; they are the weakest numbers here and must not be
overstated.** Subject 0's is a median difference of *exactly* 0.00 with p(wrong sign) 1.00 —
that arm did **not move**, which is P4 met, not an improvement. Subject 1's median is
29.54 mm but its interval reaches 0.00 at p(wrong sign) 0.06, so it is suggestive and not
established. **The six arm-joint rows and both `sized` rows are the ones whose intervals
clear zero** (p(wrong sign) 0.00 on all four arm rows, 0.007 and 0.0005 on the sized rows).
Six of eight scoreboard arms are clear of zero; the two `canon`-all-joint arms are not.

**Overall: FAIL**, on band A as pre-registered and on the temporal band. Neither failure is
a reason to revert; both are reasons to sequence. See §9.

### The controls

* **The swap is faithful.** Putting the legacy anchor back through the same helper
  reproduces the committed I1 report exactly — 41.57 and 47.05 mm — so candidate and
  control differ in one expression and nothing else.
* **No gate a constant can pass.** Sweeping the legacy scalar 0.40 → 1.20 in steps of 0.05,
  the best is 1.00 on subject 0 (12.08 mm) and 1.05 on subject 1 (10.62 mm), and **no
  scalar reaches 5 mm on either subject at any value.** The true origin is 110 mm *off* the
  torso axis; no point on that axis can land it, however the scalar is chosen.

  The cross figures are reported because they refute a tidier story this report was one
  draft away from telling. Subject 0's best scalar on subject 1 gives 11.84 mm, and subject
  1's best on subject 0 gives 12.27 mm — against 12.08 and 10.62. **The tuning transfers
  almost perfectly between the two performers**, so "it does not generalise" would have been
  a false claim. What disposes of the "tune 0.72 until a subject improves" degenerate is not
  a transfer failure; it is that the whole family, tuned to its own optimum on its own
  subject, still misses the band by more than a factor of two, and does so because the
  parametrisation is wrong — a point on a line cannot reach a point 110 mm off it.
  (For scale: the shipped 0.72 sits near the sweep's *worst* value, 41.39 / 47.38 mm.)
* **The plausible off-by-one fails.** The clavicle's parent origin (UpperChest) scores
  45.37 / 65.05 mm.
* **The legs are untouched by construction, not by luck.** 0.00 mm under the legacy anchor,
  all 17 swept scalars, the UpperChest origin and the hip-drop variant alike, and their
  step-angles are bit-identical.

### Bit identity (§C)

Between the delivered build and the branch rebuild, on both subjects: `root_translation_m`,
`triangulated_world_positions_z_up_m`, `raw_triangulated_world_positions_z_up_m`, `ticks`
and `foot_contacts` are **byte-identical**, and so are all four
`work/*-observations.jsonl` and `work/*-soma77-observations.jsonl` — nothing was
re-extracted or re-detected. The `local_rotations_xyzw` that differ are **exactly**
`{Left,Right}×{Shoulder, UpperArm, LowerArm, Hand}` and nothing else: not one torso, head,
eye, leg, foot, toe or finger local moved.

**Therefore the head gate and the facing instrument cannot move under this change, by
construction rather than by re-measurement** — they read the triangulated positions and the
torso/Neck/Head rotations, every one of which is the same bytes.

Measured anyway (§F): every forward-dot median and every handedness sign in
`facing-after.json` matches the committed `facing-location.json` to four decimals,
**including the mesh nose**, which is read off the rebuilt GLB through the real skinning —
the nose vertices carry no arm weights. The only numeric differences are the `sha_chain`
booleans (a rebuild in a different directory), `i6_reconciliation`'s yaw-180 displacement
for `LeftHand`, `RightHand` and `LeftLowerArm` (arm joints — expected), and one frame in
150 flipping a near-zero `fraction_of_frames_positive` on subject 1's triple product. No
sign median moved.

## 6. What it cost, and no instrument here was built to see it

**Twist.** `_world_for_bone` is the *minimal* rotation from the parent's frame, so the arm
below the clavicle inherits the clavicle's roll. Measuring the world reorientation of each
arm bone between the two builds, and the angle between that rotation's axis and the bone's
own long axis:

| | UpperArm median / p95 | axis offset from the bone |
|---|---|---|
| subject 0, left / right | 12.49° / 37.69°, 16.00° / 50.10° | **0.0000°** |
| subject 1, left / right | 4.97° / 57.99°, 8.12° / 41.95° | **0.0000°** |

The axis offset is zero on every bone, so **the entire change below the clavicle is a pure
twist about each bone's own long axis, carried rigidly down to the hand.** A twist about a
bone displaces nothing on that bone's chain, so *no* joint-origin score in this lane can
see it — the same lesson D1 learned on the finger curl. The hand close-ups at frame 75
(`artifacts/compare/d2-clavicle/hand-s{0,1}-{Left,Right}-{BEFORE,AFTER}.jpg`) show the hand
in the same pose relative to the forearm, with the whole arm placed differently in the
world; at that frame the twist is small enough not to read. It is not small on every frame.

**Jitter, and this one is a regression.** Per-frame local rotation step, at 30 fps:

| joint | before median / p95 / max | after median / p95 / max |
|---|---|---|
| subj 0, LeftShoulder | 0.95° / 7.73° / 38.87° | 2.58° / 13.90° / **88.94°** |
| subj 0, RightShoulder | 0.96° / 19.92° / 44.73° | 3.07° / 29.56° / **109.22°** |
| subj 1, LeftShoulder | 1.36° / 12.35° / 44.48° | 2.46° / 20.06° / **160.57°** |
| subj 1, RightShoulder | 1.16° / 15.18° / 25.24° | 2.97° / 42.04° / **150.47°** |
| subj 0/1, LeftLowerLeg | 1.85° / 8.43° / 14.48° · 4.40° / 13.09° / 20.27° | **identical** |

Frames exceeding a human's peak joint rate (~800°/s, 26.67°/frame here) over the six
clavicle-chain joints: **32 → 48** and **11 → 49**. 160.57° in one frame is 4817°/s, six
times what a human shoulder can do.

The cause is geometric and was predictable from the change itself: the old anchor sat
~400 mm from the shoulder landmark, the rig's own origin sits **60–170 mm** from it, and a
short lever arm turns the same landmark noise into far more direction noise. It is also why
sizing now swings the clavicle so hard (P5): moving the origin a few centimetres is a large
fraction of that lever.

**The arm root lands far better and travels worse. Both are true, and neither is evidence
about the other.** The positional gate was structurally incapable of noticing the second;
it is in the gate now because the question "what is this blind to?" was asked before the
conclusion was written, not after.

## 7. What the gate is blind to

* **Twist**, above — measured here only because a second instrument was built for it; no
  standing rung scores it.
* **Proportions.** The canonical rig's shoulder is far wider than these performers'. Aiming
  the clavicle correctly on a rig of the wrong width puts the arm root further out; that is
  D5 and this gate cannot separate it. It is why subject 0's `canon` scoreboard figure does
  not move.
* **Absolute placement.** Every own-capture figure is root-relative, so the deliberate
  ~90 mm ground projection *and* the 80 mm root-placement offset are outside the score by
  construction. The round trip sees the latter only because one direction is now measured in
  the rig's frame.
* **Self-reference.** The round trip scores the converter against its own output, so it
  cannot see detector error, triangulation error, or any convention shared between input and
  reference. Its own hip convention is exactly such a defect, and it took a change of
  reference frame to expose it.
* **The scoreboard** is agreement with an instrument, not accuracy. Neither side has ground
  truth, and a change that improves agreement could be moving toward MAMMA's error.
* **Arm B and the own-capture arms are not differenceable** — different body, different
  poses, different joint convention.

## 8. Strings that are now stale

Edited on this branch (`tools/swap-harness/retarget_cost.py`): the module docstring's
oracle note and the `ORACLE_roundtrip_canonical` note, both of which asserted "36-47 mm on
the arms (clavicle origin)"; and `converter_is_scale_invariant.finding`, now conditional on
the computed value, stating the D2 reason when it is FALSE and warning that the recorded
maximum is a **quaternion component** difference which reads ~2 for a mere sign flip.

Left for the registry owner, deliberately not touched:

* `tools/compare/extractors/i1_retarget.py` — `oracle_arms_*`'s note *"if this DEPARTS from
  36-47 mm the instrument is broken, not the pipeline"* (it now reads 67–79 and the
  instrument is not broken, it is mis-specified), and `sized_arms_*`'s note *"sizing is
  bit-identical re-solved or replayed"* (false since D2).
* `tools/compare/provenance.py` — the CURATED entry `"0.72 * torso_up (clavicle origin)"`.
  `provenance.scan()` no longer finds the literal in the source, but the curated registry
  still lists it as an audited unknown, and `tests/test_provenance_audit.py`'s pinned list
  still contains it. That test **passes** today; removing the entry is the deliberate edit
  its own docstring describes, and it is the owner's.

## 9. What to do with this

D2 is **correct by construction and order-independent with the root placement**: the helper
reads the rig, so fixing the root carries the clavicle with it and needs no further change
here. Two routes, and the choice is not this step's to make:

1. **Merge now.** The delivered arms improve 57 mm and 128 mm against our own capture, the
   scoreboard improves on six of its eight arms with intervals clear of zero (the two
   `canon`-all-joint arms are not: one is flat, one touches zero), an unaudited constant
   leaves, and the head and facing cannot move. The costs are that the round-trip oracle
   reads 67–79 mm until root placement ships, the i1 labels go stale, and the arm jitters
   more.
2. **Hold and ship with the root-placement fix**, after which the oracle passes at ~0.5 mm
   in one go and the jitter question can be answered on a rig that is where it belongs.

Either way the jitter regression wants its own answer, and the honest one is not smoothing:
the direction is noisy because the lever is short, and the lever is short because the rig's
shoulder is in the wrong place for these performers. That is D5.

## 10. In plain language

The character's collarbone was being aimed from the wrong place. The code worked out which
way the collar should point by measuring from an invented spot in the middle of the chest,
and then swung the bone from the shoulder — two different places, about 15 to 19 degrees
apart. The result was that the top of the arm never landed on the shoulder the cameras
actually saw, even on a body built to the rig's own proportions.

Now it measures from the shoulder itself. On the delivered characters the arm lands much
closer to where it should: about 5.7 cm better for one performer and 12.8 cm for the other,
and the independent reference fitter we compare against agrees the arms improved. A made-up
number that nobody could account for has been removed from the code and nothing replaced it.

Two things are worth knowing before anyone celebrates. First, a self-consistency test that
feeds the rig its own output back now reads *worse* — and we chased that down: it is a
separate, already-known problem in which the whole skeleton sits about 8 cm too high on its
own hips. Correct for that one thing and the test closes to under a millimetre, so the
collarbone change itself is essentially exact. Second, the arm now shakes more from frame to
frame, because aiming from the shoulder means aiming from something only 6 to 17 cm away
instead of 40 cm, and a short lever magnifies wobble. The arm ends up in a much better
place and gets there less smoothly. Both of those are in the record with numbers, because a
measurement that only reports the half you liked is the failure this project keeps
guarding against.

---

## 11. The root-placement variant, measured, not shipped

One measurement pass on the branch after the gate review. **Nothing here ships.** Every
swap is instrument-side, in `tools/compare/d2_clavicle_gate.py`; `src/` is untouched by it.
Regenerate with the gate's own command; the block lands in `gate.json` under
`root_placement_variant`, and the gate's verdict stays **FAIL**.

### 11.1 The derivation, and it introduces no constant

Forward kinematics puts the leg roots at `UpperLegMid = root + rest[Hips] + R_hips·mid`,
with `mid = ½(rest[LeftUpperLeg] + rest[RightUpperLeg])` and `rest[Root] = 0`. The captured
`left_hip`/`right_hip` landmarks *are* the femoral joint centres, so the rig's **leg roots**
— not `Hips` — belong on their midpoint. Setting `UpperLegMid = pelvis`:

&nbsp;&nbsp;&nbsp;&nbsp;`root_translation = pelvis − rest[Hips] − R_hips·mid`

against the shipped `pelvis − rest[Hips]`. Every term is the skeleton's own rest geometry.
It acts in two places and only using both is faithful: on the **rotations**, because
`_joint_origin` reads `root_translation` (this is exactly what `hip_drop_removed` produces,
since a change in the root shifts every FK origin by that vector); and on the **root
itself**, which must move *before* `project_generated_foot_contacts` runs or item 2's
question is not being asked. So the projection is **wrapped, never re-implemented** — the
wrapper shifts the incoming track, hands it to the real function, and captures the
diagnostics `positions_to_body_track` discards.

### 11.2 The pre-registered prediction was refuted, and its premise was wrong

> *Set by the coordinator before this ran:* on the canonical rig the root fix raises the
> pivot to landmark height and **shortens the lever further**, so canonical placement may
> get **worse** than D2 alone and jitter may **grow**; on the sized rig both should improve.

**Placement (median mm, lower is better):**

| | rig | D2 alone | **D2 + root** |
|---|---|---|---|
| subj 0 | canonical, delivered arms | 124.39 | **50.62** |
| subj 0 | canonical, round trip | 67.25 | **0.51** |
| subj 0 | sized, delivered arms | 78.33 | **24.99** |
| subj 0 | sized, round trip | 49.36 | **0.07** |
| subj 1 | canonical, delivered arms | 89.56 | **30.28** |
| subj 1 | canonical, round trip | 79.32 | **0.08** |
| subj 1 | sized, delivered arms | 79.63 | **23.56** |
| subj 1 | sized, round trip | 52.72 | **0.04** |

Legs and torso are unchanged in every cell (32.48 / 30.03 canonical, 19.54 / 15.96 sized;
round-trip legs and torso 0.00 throughout). Arm B moves the same way on its own reference:
canonical 130.52 → 52.84 and 89.55 → 30.80, sized 62.96 → 32.57 and 66.92 → 30.05.

**Canonical placement improved by a factor of 2.5–3, not worse.** The prediction is
refuted, and so is the mechanism behind it:

**Direction lever — `|landmark − the point the DIRECTION is measured from|` (median / p5 mm):**

| variant | subj 0 L / R | subj 1 L / R |
|---|---|---|
| legacy anchor, canonical | 205.8 / 196.4 · 179.4 / 139.2 | 195.9 / 136.5 · 190.9 / 130.2 |
| D2, canonical | 88.2 / 73.0 · 78.7 / 61.5 | 98.5 / 55.1 · 84.8 / 48.8 |
| **D2 + root, canonical** | 100.7 / 67.4 · 109.4 / 48.0 | 159.8 / 98.8 · 136.8 / 87.7 |
| **D2 + root, sized** | 123.4 / 100.1 · 112.3 / 59.9 | 125.8 / 63.6 · 115.1 / 59.7 |

(This is deliberately *not* "pivot-to-landmark distance" in general: under the legacy anchor
the bone still pivots at the rig's Shoulder origin while the direction comes off the torso
axis, and it is the **direction's** lever that converts landmark noise into bone noise.)

**The root fix LENGTHENS the lever in every cell** — 88.2 → 100.7, 98.5 → 159.8 — because
the rig's shoulder origin was already at or above the captured shoulder, so raising it
moves it *away*. The prediction's premise ("shortens the lever further") is measurably
backwards.

And the lever explains the placement, which is the useful part: the arm root is placed one
`UpperArm` rest length (160 mm canonical) along the ray, so the residual tracks
**|lever − 160 mm|**, not the lever itself. Subject 0's D2+root canonical right shoulder
has a 109.4 mm lever → |109.4 − 160| = 50.6 mm, against a measured 50.62 mm. Subject 1's
159.8 / 136.8 mm levers sit almost exactly at the rest length → 30.28 mm. **The
discriminating constraint is not "short lever" but "lever ≠ the bone's own rest length",**
and the root fix moves both subjects toward it on the canonical rig by accident of their
proportions — which is precisely why D5 owns the general case.

**Jitter (clavicle-chain frames above 26.67°/frame; median / p95 / max step, degrees):**

| | rig | D2 alone | D2 + root |
|---|---|---|---|
| subj 0 | canonical, over-ceiling | 48 | **40** |
| subj 0 | canonical, LSh · RSh max | 88.94 · 109.22 | 138.98 · 164.48 |
| subj 0 | sized, over-ceiling | 43 | 41 |
| subj 1 | canonical, over-ceiling | 49 | **33** |
| subj 1 | canonical, LSh · RSh max | 160.57 · 150.47 | **39.34 · 53.57** |
| subj 1 | sized, over-ceiling | 22 | **35** |

Mixed, and it does not resolve the regression. Median steps fall everywhere (2.58 → 1.90,
2.46 → 1.53); the over-ceiling count falls on both canonical rigs; but the *worst* single
step grows on subject 0 canonical (109 → 164°) and the sized rig gets **worse** on subject 1
(22 → 35 frames). So the second half of the prediction — "on the sized rig both should
improve" — is **half met**: placement improves everywhere, jitter does not.

**The two cells that were supposed to decide the scoping question do not decide it.**
Placement says "D2 + root placement" is a strong pairing on either rig. Jitter says the
per-performer skeleton is still load-bearing, because the one cell that should have been
best — sized, root-corrected — is the cell where subject 1's jitter is worst. On this
evidence D2 + root placement is a **viable scoping option for placement and not a fix for
jitter**, and jitter remains D3/D5's to answer.

### 11.3 The ground projection absorbs the lift, through the term that has no cap

The coordinator's premise here is also off by a call-site: `project_generated_foot_contacts`
defaults to `maximum_root_correction_m = 0.05`, but `positions_to_body_track` **overrides it
to 0.08** (`commercial_multiview.py:1752`). The gate records the value actually passed.

| | requested lift | contact frames | max contact correction (cap 80 mm) | penetration lift applied (med / max) | lowest foot min / median (m) | floor (m) | penetration after |
|---|---|---|---|---|---|---|---|
| subj 0, D2 alone | 0.00 mm | [47, 42] | 34.49 mm | 142.37 / 170.99 mm | 0.00000 / 0.04181 | 0.01169 | 0.000 mm |
| subj 0, **D2 + root** | 80.00 mm | [37, 60] | 38.23 mm | **83.01 / 94.67 mm** | −0.00000 / 0.05461 | 0.00071 | 0.000 mm |
| subj 1, D2 alone | 0.00 mm | [6, 27] | 16.08 mm | 109.54 / 113.40 mm | 0.00000 / 0.07959 | 0.01473 | 0.000 mm |
| subj 1, **D2 + root** | 80.00 mm | [7, 27] | 16.72 mm | **49.06 / 51.57 mm** | 0.00000 / 0.08991 | 0.01254 | 0.000 mm |

**The cap is never approached and never was the risk.** The capped, per-contact-run
correction moves barely at all under an 80 mm lift (34.49 → 38.23, 16.08 → 16.72 mm) and
stays less than half the 80 mm cap. What absorbs the lift is a different, **uncapped**
term: `body_projection.py:1209–1211` adds `penetration_before` to the root's Y outright,
with no bound at all. That term *falls* by ~59–60 mm (the lift's vertical component after
the subjects' lean), 142.37 → 83.01 and 109.54 → 49.06 mm. Ground penetration after
projection is 0.000 mm in every case, and contact counts barely move.

Two things worth carrying out of that. First, **the root fix does not fight the ground
projection; it relieves it** — the pipeline was hoisting the character ~140 / ~110 mm to
keep it out of the floor, and after the fix it hoists ~83 / ~49 mm, because the rig is no
longer sunk by its own hip convention. Second, **the projection's real degree of freedom is
uncapped**, which means "the root correction is capped at 80 mm" is not a statement about
how far this stage can move the character vertically. Nothing here changes it.

One more, and it limits what the sized rows above can be asked to mean: the sized and
canonical ground rows are **byte-identical**, because `project_generated_foot_contacts`
hardcodes `DETAILED_HUMANOID` (`body_projection.py:1127` and its FK calls) and never sees
the sized skeleton. Pre-existing, unchanged here, recorded because a sized ground figure
that cannot see the sizing is not a sized ground figure.

### 11.4 The picture

`tools/compare/d2_jitter_sheet.py` → `artifacts/compare/d2-clavicle/jitter/`:
`jitter-contact-sheet-subject01-LeftShoulder.png` (frames 44–54, both builds, each tile
labelled with the step it is about to take and boxed red when it exceeds a human's peak
rate), `jitter-zoom-subject01-f047-050.png` (the spike at 5×), and
`jitter-A001-subject01-before-vs-after.mp4` (frames 40–70 side by side). It wraps
`tools/swap-harness/camera_overlay.py`, which sets the scene fps from the track *before* the
glTF import; the POSE CHECK reads 57 / 95 mm, which is the known ground-projection offset
and not a timebase error.

**What the eye sees:** in the delivered build the near performer's outstretched arm swings
forward smoothly across frames 47–50, while in the D2 rebuild it pops — at frame 48 the arm
is drawn in and angled down with the hand by the other performer's neck, and one frame later
it has snapped out straight and horizontal, a visible single-frame shoulder dislocation that
the positional score prices at zero.

---

## 12. D2b: the root placed on the captured hips

Branch `ladder/D2`, from `662c844`, 2026-09-03. One expression ships, in
`src/autoanim_gnm/commercial_multiview.py`:

```python
root_translation[frame] = pelvis - rest["Hips"] - _leg_root_offset(hips_world, rest)
```

with `_leg_root_offset` returning `R_hips · mid(rest["LeftUpperLeg"], rest["RightUpperLeg"])`.
Forward kinematics puts the rig's leg roots at `root + rest[Hips] + R_hips · mid`; the
captured `left_hip` / `right_hip` landmarks are taken to be the femoral joint centres, so
the **leg roots** — not `Hips` — belong on their midpoint. Setting `UpperLegMid = pelvis`
gives exactly the line above. Every term is the skeleton's own rest geometry, read from the
caller's `rest` dict; **0.08 is never written down**, and
`tests/test_root_placement.py::test_no_constant_arrived_with_the_root_fix` parses the
shipped function with `ast` and allows no numeric literal but the `0.5` of a midpoint.

The helper is module level and called by bare name, exactly as `_joint_origin` is, so every
control below runs through the identical code path rather than a re-implementation of it.
The clavicle origin needed no change: `_joint_origin` walks from `root_translation`, so
D2's fix carries automatically, which is what §9 predicted.

### 12.1 Regenerating everything

```
PYTHONPATH=$PWD/src .venv/bin/python scripts/build_commercial_multiview_comparison.py \
   --videos .cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos \
   --calibration-yaml .cache/mamma/configs/examples/calib/iphones_outdoors.yaml \
   --detector soma77 --output artifacts/compare/d2-clavicle/delivery-root
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d2_clavicle_gate.py
PYTHONPATH=$PWD/src python3 tools/swap-harness/retarget_cost.py \
   --tracks artifacts/commercial-multiview-soma77 \
   --out artifacts/compare/d2-clavicle/retarget-cost-d2b.json
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/mamma_scoreboard.py \
   --tracks artifacts/compare/d2-clavicle/delivery-root --label d2b-root-after
PYTHONPATH=$PWD/src .venv/bin/python tools/compare/facing_location.py \
   --delivery artifacts/compare/d2-clavicle/delivery-root \
   --out artifacts/compare/d2-clavicle/facing-d2b.json --label "D2b"
.venv/bin/python tools/compare/silhouette.py \
   --delivery artifacts/compare/d2-clavicle/delivery-root \
   --work artifacts/compare/d2-clavicle/silhouette-work-d2b \
   --out artifacts/compare/d2-clavicle/silhouette-d2b.json     # and the same for `delivery`
```

The rebuild re-extracted and re-detected nothing: all eight `work/*observations.jsonl` are
byte-identical to the delivered build's. **Deviation from the plan, recorded:** the plan
said `--work <rebuild>/work` for the silhouette; a dedicated work directory is used instead,
seeded with `artifacts/compare/i6`'s MAMMA-derived caches (`masks-960x540*.npz`,
`mean-body-0*.npy`). Those read MAMMA's masks and mean bodies and never our track, and the
ORACLE arm reading **bit-identical (max IoU difference 0.0000)** between the two runs is the
proof that reusing them changed nothing.

**Every D2 figure in §§1–11 above still stands, and is checked.** The gate now runs every
pre-existing block inside `with root_offset(zero_offset)` — the pre-D2b converter, reached
by swapping one module attribute — and `d2_regression_check` compares the result against the
gate.json this run replaced: **max absolute difference 0.000 mm across every D2 band.**

### 12.2 The pre-registered expectations, and whether each was met

| | expectation | outcome |
|---|---|---|
| **1** | the shipped derivation reproduces D2's instrument-side variant to 0.00 mm | **MET, 0.00 on every figure**, including the hoist and the integer over-ceiling counts. Round trip 0.51 / 0.08 canonical and 0.07 / 0.04 sized; delivered arms 50.62 / 30.28; hoist 83.01 / 49.06. |
| **2** | the four theorems | **MET**, and the plan's set for theorem 3 was wrong — see below. T1 max 4.2e-7 m; T2a 0.0 at 1e-12 in float64; T2b 4.5e-7 m; T3 exactly the clavicle chain and the two hands. |
| **3** | horizontal placement and tilt correlation collapse; vertical barely moves; rung 11 legs improve modestly, not to zero | **MET, almost exactly as written.** |
| **4** | the replayed sized arm gets a new hip misplacement | **DID NOT OCCUR**, and the mechanism is the reason, not luck — see below. |
| **5** | temporal reported, no band, 40 / 33 expected on the canonical rig | **MET: 40 and 33.** |
| **6** | contacts change; penetration after must be 0 | **MET.** [47, 42] → [37, 60] and [6, 27] → [7, 27]; penetration after 3.4 and 1.5 **nano**metres. |
| — | the silhouette | **NOT PRE-REGISTERED AS A BAND, AND IT FELL.** §12.6. |

### 12.3 The gate

`tools/compare/d2_clavicle_gate.py` → `artifacts/compare/d2-clavicle/gate.json`, block
`d2b_root_placement`, verdict `d2b_verdict`. Bootstraps are moving-block, block 15, 2000
draws, seed 20260902, both arms on identical drawn frames.

| band | before (D2 alone) | D2b | interval | verdict |
|---|---|---|---|---|
| **FAITHFUL SWAP.** reproduces D2's variant, both subjects | — | 0.00 | — | PASS |
| **THEOREM T1.** FK UpperLeg midpoint = captured hip midpoint, both rigs, both subjects | — | ≤ 4.2e-7 m | band 1e-6 | PASS |
| **THEOREM T2a.** the offset is `R_hips·(0, 0.08, 0)`, 0.08 read from `rest` | — | 0.0 | band 1e-12 | PASS |
| **THEOREM T2b.** the root moved by exactly that and nothing else | — | ≤ 4.5e-7 m | band 1e-6 | PASS |
| **THEOREM T3.** pre-projection, only the clavicle chain + hands moved | — | held | — | PASS |
| **A(D2b).** round-trip arms ≤ 5 mm, canonical, subj 0 / 1 | 67.25 / 79.32 | **0.51 / 0.08** | — | PASS |
| **A(D2b).** round-trip arms ≤ 5 mm, sized, subj 0 / 1 | 49.36 / 52.72 | **0.07 / 0.04** | — | PASS |
| **A(D2b).** round-trip legs and torso = 0.00, both rigs, both subjects | 0.00 | 0.00 | — | PASS |
| **CONTROL (b).** the lift sign-flipped must fail, subj 0 / 1 | — | 54.71 / 85.69 | — | PASS |
| **CONTROL (c1).** the sweep's minimum is at the skeleton's own lift, both | — | 80 mm | — | PASS |
| **CONTROL (c2).** ONLY that lift clears the band, subj 0 | — | 1 of 17 | — | PASS |
| **CONTROL (c2).** same, subj 1 | — | **2 of 17** | — | **FAIL** |
| **CONTROL (c3).** no lift but the skeleton's own clears the band on BOTH subjects | — | 1 | — | PASS |
| **CONTROL (d).** the same lift as a WORLD vertical must fail, subj 0 / 1 | — | 26.04 / 38.59 | — | PASS |
| **CONTROL (e).** legs and torso 0.00 under every variant | — | 0.00 | — | PASS |
| **ABSOLUTE.** delivered hip joints vs captured hips, HORIZONTAL, subj 0 / 1 | 26.20 / 42.95 | **0.00 / 0.00** | p95 10.69 / 4.17 | reported |
| **ABSOLUTE.** the same, VERTICAL, subj 0 / 1 | 53.30 / 32.31 | 55.11 / 27.71 | p5–p95 49.6–62.0 / 22.9–27.7 | reported |
| **RUNG 11.** canonical, all joints, subj 0 / 1 | 151.58 / 103.91 | **71.30 / 67.92** | [50.6, 87.5] / [27.1, 57.9] | reported |
| **RUNG 11.** canonical, ARMS, subj 0 / 1 | 186.93 / 117.27 | **83.72 / 68.34** | [83.7, 120.8] / [35.8, 66.0] | reported |
| **RUNG 11.** canonical, LEGS, subj 0 / 1 | 64.82 / 81.80 | 58.62 / 63.54 | [−7.3, 14.3] / [−3.8, 62.9] | reported |
| **CONTACTS.** penetration after projection ≈ 0, both rigs, both subjects | 0.000 | 0.0000034 mm | — | PASS |
| **THEOREM (on disk).** positions, raw positions, ticks, observations byte-identical; differing locals = clavicle chain + hands + **feet** | — | held | — | PASS |
| **I6 SILHOUETTE.** median IoU must not fall | 0.5847 | **0.5191** (D2 alone: 0.5574) | 8 of 8 cells fell | **FAIL** |
| **META.** extending this gate moved no D2 figure | — | 0.000 mm | — | PASS |

**Overall D2b: FAIL**, on control (c2) for subject 1 and on the silhouette.

### 12.4 Three things the plan got wrong on contact with the code

**(a) Theorem 3's joint set is right only before the ground projection.**
`project_generated_foot_contacts` rewrites `rotations[:, foot_index]` inside contact runs,
and the runs move when the root does ([47, 42] → [37, 60]). So on the **delivered** tracks
the differing set is the clavicle chain, the two hands **and the two feet**, and the gate
asserts that corrected set on disk while asserting the plan's set on the pre-projection
track. The consequence the plan drew from theorem 3 survives with one correction: the
facing instrument reads the Hips, chest, Neck and Head rotations and the mesh nose, and not
one of those is a foot. The plan's own inference — "rotations identical means the facing
forward-dots cannot move" — was nonetheless one joint too broad: the delivered **feet**
forward-dot is a facing figure, and it moves. Measured rather than asserted — `facing-d2b.json` against the
committed `facing-location.json`: **every forward-dot median and every handedness figure is
identical to five decimals**, including the mesh nose, except
`delivered_feet vs_our_capture_forward` on subject 0 (0.96980 → 0.97071), which is the
contact-run rewrite. The position-derived figures (`basis_assertion`, the yaw-180
displacement control in `i6_reconciliation`) move, as they must: the root moved, and they
are distances.

**(b) The sized-replay defect cannot occur under `sized_skeleton`.** The plan expected the
replayed `sized` arm's hips to be misplaced by `R·(0, 0.08·(k−1), 0)`. Measured
misplacement: **0.0000 mm**, and the mechanism is why. `tools/head/sized_skeleton.py`
scales the hip half-span in **X only** —
`rest_translation_m=(off[0]*k, off[1], off[2])` — and never touches `Hips`. The two
UpperLeg rest offsets are mirror images in X, so their **midpoint** is unchanged by any
hip-span sizing. The assertion stays in the gate: a future sizing that scaled the UpperLeg
Y would reintroduce the defect and this check would catch it.

**(c) The 5 mm band admits one neighbour on subject 1, and it has not been tightened.**
Sweeping the lift 0 → 160 mm in 10 mm steps along the hips' own up axis:

| lift, mm | 0 | 40 | 60 | 70 | **80** | 90 | 100 | 160 |
|---|---|---|---|---|---|---|---|---|
| subj 0, round-trip arms | 67.25 | 36.62 | 17.82 | 8.13 | **0.51** | 6.46 | 11.46 | 22.70 |
| subj 1, round-trip arms | 79.32 | 32.38 | 12.98 | 5.81 | **0.08** | 4.64 | 8.25 | 18.26 |

The minimum is at the skeleton's own 80 mm on both subjects, but on subject 1 the 90 mm
step also clears 5 mm at 4.64. That row is reported as **FAIL** rather than the band being
moved. What disposes of the tuned-constant degenerate is control (c3): a shipped constant
is **one number for every performer**, and 90 mm misses the band on subject 0 at 6.46 mm,
so exactly one lift of the seventeen clears it on both — the one the code reads from
`rest` and never writes down.

### 12.5 What D2b actually changed, and what it did not

**It moved the placement error out of the converter and into the ground projection, where
it is now visible. It did not remove it.**

The delivered rig's own hip joints — FK's UpperLeg midpoint of the shipped track, *after*
the projection — against the captured hip midpoint, in absolute capture world:

| | horizontal (median) | vertical, capture +Z (median) | norm (median / p95) | correlation with pelvis tilt |
|---|---|---|---|---|
| subj 0, delivered (pre-D2) and D2 — identical | 26.20 mm | 53.30 mm | 59.48 / 117.53 | **0.975** |
| subj 0, **D2b** | **0.00 mm** (p95 10.69) | 55.11 mm | **55.11 / 61.98** | **0.183** |
| subj 1, delivered and D2 — identical | 42.95 mm | 32.31 mm | 53.55 / 127.20 | **0.999** |
| subj 1, **D2b** | **0.00 mm** (p95 4.17) | 27.71 mm | **27.71 / 27.71** | **0.357** |

The pre-registered mechanism, written before the numbers: the projection adds **one**
uncapped scalar vertical hoist per take, chosen by the single worst frame
(`body_projection.py:1209`, `roots[:, 1] += penetration_before`, no bound), while the root
fix adds `R_hips·(0, 0.08, 0)` **per frame**, whose vertical part shrinks and whose
horizontal part grows as the pelvis tilts. So the horizontal term and the tilt correlation
should collapse and the median vertical should barely move, because the hoist re-solves.
**That is what happened**: the hoist falls 142.37 → 83.01 mm and 109.54 → 49.06 mm, and the
median vertical hip offset moves by +1.8 mm and −4.6 mm. The delivered rows the D2 report
could not see — every own-capture figure there is root-relative — are the p95 of the norm:
117.53 → 61.98 and 127.20 → 27.71 mm. **The tail is where the tilt-dependent term lived.**

That the pre-D2 and D2 rows are *identical* is itself the point: D2 moved no hip, and no
root-relative instrument in this lane could see this vector at all.

Rung 11, MAMMA's joints, absolute capture world, the scoreboard's own statistic, legs and
arms separated, paired moving-block bootstrap on identical draws:

| | pre-D2 | D2 | D2b | D2 − D2b margin, 95 % CI, p(wrong sign) |
|---|---|---|---|---|
| subj 0, canonical, all 15 | 151.58 | 151.58 | **71.30** | 72.39 [50.61, 87.48] p 0.000 |
| subj 0, canonical, six arms | 231.21 | 186.93 | **83.72** | 108.90 [83.69, 120.81] p 0.000 |
| subj 0, canonical, six legs | 64.82 | 64.82 | 58.62 | 4.79 [−7.33, 14.33] **p 0.124** |
| subj 0, sized, six legs | 77.90 | 77.90 | **80.44** | −3.93 [−7.16, 10.81] **p 0.246** |
| subj 1, canonical, all 15 | 137.38 | 103.91 | **67.92** | 41.17 [27.12, 57.89] p 0.000 |
| subj 1, canonical, six arms | 241.47 | 117.27 | **68.34** | 48.72 [35.82, 65.97] p 0.000 |
| subj 1, canonical, six legs | 81.80 | 81.80 | 63.54 | 23.48 [−3.83, 62.89] **p 0.115** |
| subj 1, sized, six legs | 81.45 | 81.45 | 71.42 | 15.09 [1.36, 51.35] p 0.012 |

**Read the leg rows carefully.** Three of the four have intervals that reach zero, and
subject 0's sized legs read *worse* by 3.93 mm with the interval spanning zero — not
established either way. That is the pre-registration met, not evaded: the legs were
predicted to improve **modestly and not to zero**, because only the horizontal,
tilt-dependent part of the hip error leaves and the vertical residue — the legs' surplus
length, canonical thigh 430 mm against roughly 400 measured — stays. The arm rows are the
ones whose intervals clear zero at p 0.000.

Arm B, on its own reference and never differenced with the rows above: canonical arms
130.52 → 52.84 and 89.55 → 30.80; sized 62.96 → 32.57 and 66.92 → 30.05. Its legs and torso
do not move at all, which is the same theorem in a different body.

Contacts and the ground, both rigs (the sized rows are byte-identical to the canonical ones
because `project_generated_foot_contacts` hardcodes `DETAILED_HUMANOID` — §11.3,
pre-existing and unchanged):

| | contacts | hoist (uncapped) | max per-contact correction (cap 80 mm) | lowest foot median | penetration after |
|---|---|---|---|---|---|
| subj 0, D2 alone | [47, 42] | 142.37 mm | 34.49 mm | 0.04181 m | 0.000 mm |
| subj 0, **D2b** | [37, 60] | **83.01 mm** | 38.23 mm | 0.05461 m | 0.0000034 mm |
| subj 1, D2 alone | [6, 27] | 109.54 mm | 16.08 mm | 0.07959 m | 0.0000015 mm |
| subj 1, **D2b** | [7, 27] | **49.06 mm** | 16.72 mm | 0.08991 m | 0.0000015 mm |

Temporal, the baseline D2c inherits. **No band; D2b makes no claim here.** Clavicle-chain
frames above 26.67°/frame (a human's ~800°/s at 30 fps), canonical rig:

| | pre-D2 | D2 | D2b |
|---|---|---|---|
| subj 0 | 32 | 48 | **40** |
| subj 1 | 11 | 49 | **33** |
| subj 0, sized | — | 43 | 41 |
| subj 1, sized | — | 22 | **35** |

Median step falls everywhere (subj 0 LeftShoulder 2.58° → 1.90°, subj 1 2.46° → 1.53°) and
the worst single step goes both ways: subject 1's LeftShoulder 160.57° → 39.34°, subject
0's RightShoulder 109.22° → **164.48°**. Legs, neck and head step-angles are bit-identical.
**D2's jitter regression is reduced and not resolved, and the sized rig gets worse on
subject 1.** That is D2c's.

### 12.6 The silhouette fell, on every camera and both performers

`artifacts/compare/d2-clavicle/silhouette-d2b.json`, against MAMMA's SAM2 masks — the one
retained artifact on this fixture that is **not** model-mediated. Median IoU over the eight
camera-subject cells: **0.5847 (delivered, pre-D2) → 0.5574 (D2 alone) → 0.5191 (D2b)**.
It fell in 8 of 8 cells, and **both precision and recall fell in every one**, so the mesh
moved *off* the pixels rather than changing size. The ORACLE arm — MAMMA's own mesh through
our scoring path, which reads none of our track — is bit-identical between the runs (max IoU
difference 0.0000), so the two runs are comparable and the fall is real.

What moved: the delivered rig's every joint translates by the **same** median displacement,
27.87 mm on subject 0 and 43.16 mm on subject 1, of which the vertical part is +2.53 and
−4.59 mm. It is an almost purely **horizontal** rigid shift, and it agrees to within 2 mm
with the horizontal hip offset D2b removes (26.20 and 42.95 mm) — not identically, because
the rig moves by `−R_hips·(0, 0.08, 0)` per frame *plus* the change in the projection's
hoist, and the offset is the horizontal part of the first term alone. So the silhouette is
pricing this step's own correction, and it prices it as worse.

**Two readings, and this instrument cannot separate them.**

1. **The derivation's premise may be false.** "The captured `left_hip` / `right_hip`
   landmarks ARE the femoral joint centres" is an assumption, inherited from
   `retarget_cost.landmarks_from_fk`'s comment and from the SOMA-77 adapter, and never
   measured. Everything downstream of it here is exact — the theorems hold to a micron —
   but exactness about a premise is not evidence for the premise. The silhouette is the
   first instrument in this lane with evidence bearing on it, and it points the other way.
2. **The mesh is the wrong body, by a lot.** Even the *before* arm reads precision ≈ 0.87
   with recall ≈ 0.64: our mesh covers under two thirds of the mask while almost all of it
   lands inside, against MAMMA's oracle at 0.84–0.88 IoU. That is a shape mismatch large
   enough (D5: shoulder span 540 mm against 346 and 363 measured; thigh 430 against ~400)
   that a 3–4 cm rigid shift can move the overlap either way without saying which placement
   is anatomically right. D2 alone, which moved no hip at all, also cost 0.027 of IoU.

**No mechanism is claimed here.** What would discriminate is a per-frame IoU series
correlated with pelvis tilt — `silhouette.py` retains only medians and p05, so it was not
available — or a direct measurement of where the delivered mesh's own hip surface sits
relative to its `UpperLeg` joint. Neither was done, and neither is a gap in D2b: nothing in
this step's plan asked for a mechanism, and building the instrument that would supply one is
D5's or the owner's call. The honest statement is that the step's
internal derivation is exact, its absolute placement against **our own** captured hips is
now zero horizontally, and the one instrument that reads photographs disagrees.

### 12.7 The committed test that D2b makes false, and the one-line fix

`tests/test_clavicle_origin.py::test_the_roundtrip_residual_is_root_placement_not_the_clavicle`
**fails**, by design:

```
assert max(shipped[n] for n in ARM_LANDMARKS) > 5.0
E   assert 0.008797753306019427 > 5.0
```

It asserts the *inverse* of D2b: that the shipped round trip exceeds 5 mm and only closes
when the hip drop is removed from the re-solve. That was D2's finding and its own docstring
named the remedy — "Fix the root/hip placement convention first". Both of its assertions are
now false (`_hip_drop_removed` on pass 2 double-corrects). It was **deliberately not
edited**: the brief for this step forbids touching an existing test file. The owner's fix is
two inequalities:

```python
assert max(shipped[n] for n in ARM_LANDMARKS) <= 1.0          # was > 5.0
assert max(corrected[n] for n in ARM_LANDMARKS) > 5.0         # was <= 1.0
```

and the docstring's tense. `tests/test_root_placement.py::test_the_canonical_round_trip_now_lands_the_arms_within_a_millimetre`
and `::test_the_zero_offset_variant_fails_the_same_round_trip` are that pair, written the
right way round.

Two other failures are **pre-existing at `662c844`** and reproduce with identical assertion
text with the D2b edit stashed out:
`test_body_compositor.py::test_unified_preview_is_explicitly_diagnostic_and_uses_one_video_clock`
and `test_head_orientation.py::test_body_track_head_is_a_constant_without_a_solve_and_moves_with_one`
(`assert 12.470848083496094 > 20.0`, identical before and after). Everything else passes:
95 of 98 in the committed set, and 9 of 9 in `tests/test_root_placement.py`.

`body_projection.py`'s second projection function, `constrain_restrained_root_travel`
(:108), is **not** on the capture path — its only callers are
`speech_motion_candidates.py:199` and `scripts/build_audio_acting_shot.py:325`. Checked, not
assumed. `body_compositor.py:237` and `body_export.py:463` pass the root through unchanged
and need no edit; their tests pass.

### 12.8 What this is blind to

* **The hoist that remains.** D2b does not put the character on the ground correctly. It
  moves the error from the converter into the projection, where it is one uncapped vertical
  scalar per take (83 and 49 mm), and that residue is the legs' surplus length. **D5.**
* **The premise.** Nothing here measures whether the detector's hip landmark is the femoral
  joint centre. §12.6 is the only evidence, and it is ambiguous.
* **Self-reference.** The round trip scores the converter against its own output. Its
  0.51 / 0.08 mm is a *consistency* figure, not an accuracy figure — and its own hip
  convention was exactly such a shared defect, which is why it took a change of reference
  frame to expose it in the first place. Now that the two agree again, it can no longer see
  this class of error at all.
* **Twist**, unchanged from D2 §6: `_world_for_bone` passes the clavicle's roll down the
  arm and no joint-origin score in this lane can see it.
* **The temporal defect.** Reduced, not resolved, and worse on the sized rig for
  subject 1. **D2c.**
* **The scoreboard** is agreement with an instrument, not accuracy; neither side has ground
  truth.
* **The silhouette** is blind to depth, to a left/right mirror of a fore-aft symmetric pose,
  and to everything inside the outline. It cannot separate a shape error from a pose error,
  which is exactly the ambiguity in §12.6.

### 12.9 In plain language

The rig has two hip joints, one at the top of each thigh, and the cameras see roughly where
the performer's are. Until now the code put a *different* bone on that spot — the pelvis
root, which sits about 8 cm higher — so the whole skeleton hung 8 cm low on its own hips,
on every frame. Nobody noticed, because every check in this lane measures distances
*between* joints or *relative to* the hips, and a whole body sliding 8 cm cancels out of
both. The floor-contact stage then quietly hoisted the character back up by about 14 cm to
stop its feet sinking through the ground.

Now the thigh joints go where the cameras say the hips are. Sideways, the delivered
character's hips land exactly on the measured ones — the miss goes from 2.6 and 4.3 cm to
zero, and, more tellingly, it stops depending on how far the performer is bent over, which
it used to almost perfectly. The independent reference fitter agrees the joints improved:
15 cm to 7 cm on one performer, 10 cm to 7 cm on the other. The floor stage now has to lift
the character only half as far, because it is no longer sunk by its own convention.

Two things must be said beside that. The up-and-down miss did **not** go away — it moved
from one stage to another, where it is now a single visible number per shot, and what is
left of it is that the rig's legs are about 3 cm too long for these performers. That is a
later step. And the one measurement here that is scored against the actual photographs
rather than against other software — how well the character's outline covers the person cut
out of the video — got **worse**, on all four cameras and both performers. The character
has moved off the person by about 3–4 cm sideways, which is precisely the correction this
step makes. Either the point the detector calls a hip is not the point the rig calls a hip,
or the stock character is simply the wrong shape by enough that moving it correctly can
still look worse. This report does not know which, says so, and leaves the number in plain
sight rather than in a footnote.

---

## 13. Does the silhouette's fall track trunk tilt? One measurement pass

Section 12.6 left two readings open and claimed no mechanism. The coordinator set a
hypothesis and a decision rule, both recorded verbatim in
`artifacts/compare/d2-clavicle/silhouette-vs-tilt.json` **before the instrument ran**:

> the rig has no pelvis frame separate from the torso (Hips, Spine, Chest, UpperChest all
> take `torso_up`), so `R_hips · (0, 0.08, 0)` has a horizontal component of
> 80·sin(trunk tilt) … **PREDICTION: on upright frames (tilt under ~10°) D2b's IoU equals
> the delivered build's; the fall grows with tilt.** If it does, the mechanism is the
> offset applied along the trunk axis and the underlying defect is the missing pelvis
> frame (converter work, its own step). If the fall is flat in tilt, it is the mesh's
> shape (D5/D6) and D2b's silhouette cost is a price to weigh, not a defect.

`tools/compare/silhouette_vs_tilt.py` — **instrument only**, imports `silhouette.py` and
reuses its rasteriser, scorer, mask store and posed-mesh caches, so the three builds and
the oracle go through one pixel path; `silhouette.py`'s own outputs are untouched.
Regenerate with
`PYTHONPATH=$PWD/src .venv/bin/python tools/compare/silhouette_vs_tilt.py`
(per-frame scores are cached in `silhouette-vs-tilt-per-frame.npz`, so a re-cut is free).
Folded into `gate.json` as `d2b_root_placement.silhouette_vs_tilt`; figure at
`artifacts/compare/d2-clavicle/silhouette-vs-tilt.png`.

The three builds share byte-identical triangulated positions (asserted), so trunk tilt is
one variable across all of them, and the id resolution is read from the retained
silhouette reports and cross-checked between them rather than re-derived.

### 13.1 The tercile table

Trunk tilt = angle of (neck − hip midpoint) from capture +Z. IoU is the mean over the four
cameras of that frame's IoU, on frames every camera scored — one denominator for all arms.
Bootstrap: block 15, 2000 draws, blocks drawn over the whole 150-frame axis and the
statistic taken over the drawn frames in the tercile, both arms on the same draws.

**Subject 0** (tercile edges 14.68° / 26.84°)

| tercile | tilt range | n | delivered IoU | D2 IoU | D2b IoU | D2b − delivered [95 % CI] | oracle vs its own upright | rig moved, horiz / vert |
|---|---|---|---|---|---|---|---|---|
| upright | 6.5–14.5° | 50 | 0.6056 | 0.6087 | 0.5566 | **−0.0490** [−0.0700, −0.0418] | +0.0000 | 12.1 / +6.3 mm |
| middle | 14.8–26.8° | 50 | 0.5508 | 0.5530 | 0.4712 | **−0.0796** [−0.0865, −0.0462] | −0.0400 | 27.4 / +2.3 mm |
| most bent | 27.0–65.3° | 50 | 0.5369 | 0.5035 | 0.4219 | **−0.1151** [−0.1212, −0.0564] | −0.0403 | 67.4 / −28.9 mm |

**Subject 1** (tercile edges 17.35° / 53.06°)

| tercile | tilt range | n | delivered IoU | D2 IoU | D2b IoU | D2b − delivered [95 % CI] | oracle vs its own upright | rig moved, horiz / vert |
|---|---|---|---|---|---|---|---|---|
| upright | 0.5–17.2° | 50 | 0.6241 | 0.5992 | 0.5610 | **−0.0631** [−0.0977, −0.0186] | +0.0000 | 5.8 / +8.0 mm |
| middle | 17.5–52.4° | 50 | 0.5730 | 0.5445 | 0.5233 | **−0.0497** [−0.1028, −0.0206] | −0.0480 | 43.0 / −4.6 mm |
| most bent | 54.5–90.8° | 50 | 0.5715 | 0.5833 | 0.5051 | **−0.0664** [−0.0954, −0.0190] | −0.0953 | 79.2 / −60.5 mm |

**At the threshold the hypothesis names, tilt ≤ 10°:**

| | n | delivered IoU | D2b IoU | D2b − delivered [95 % CI] | rig moved (norm) | hands moved |
|---|---|---|---|---|---|---|
| subject 0 | 31 | 0.6148 | 0.5703 | **−0.0445** [−0.0508, −0.0312] | 12.67 mm | 144 / 175 mm |
| subject 1 | 35 | 0.6596 | 0.5668 | **−0.0928** [−0.0991, −0.0516] | 9.64 mm | 183 / 188 mm |

### 13.2 The prediction is refuted, and by its own discriminating clause

**On upright frames D2b's IoU does not equal the delivered build's.** At tilt ≤ 10°, where
the trunk-axis offset's horizontal component is about 14 mm and the whole rig has in fact
moved **9.6–12.7 mm**, D2b is already 0.045 and 0.093 IoU below the delivered build, both
intervals clear of zero. On subject 1 that is **the largest fall anywhere on the take** —
larger than in the most-bent tercile, where the rig moves 99 mm.

The second clause, "the fall grows with tilt", holds on subject 0 (−0.049 → −0.080 →
−0.115, monotone) and does not on subject 1 (−0.063 → −0.050 → −0.066, a change of 0.003
IoU that sits inside the width of its own interval). And the oracle — MAMMA's own mesh,
which reads none of our track — itself loses 0.040 and 0.095 IoU in the most-bent tercile
against its own upright one, so **bent frames are intrinsically harder for any mesh in this
instrument** and part of subject 0's gradient belongs to the instrument, not to us. The
gate's mechanical label is `PARTLY TILT-DEPENDENT`; the magnitudes above are what it means.

### 13.3 What the shift is, and what the fall is not

The geometry the coordinator derived is confirmed exactly. The unit shift vector's dot
product with the unit horizontal component of `torso_up` is **+0.932 (p5 0.838)** on
subject 0 and **+0.904 (p5 0.435)** on subject 1: the offset does go along the trunk's
lean, because on this rig the hips' up axis *is* the trunk axis. Its magnitude tracks
80·sin(tilt) — 27.87 and 43.16 mm median against 27.4 and 43.0 mm predicted.

But the fall is not that shift, and three measurements say so.

1. **The fall is at full size where the shift is smallest.** 10 mm of body displacement
   costs 0.045 and 0.093 IoU; 74 and 99 mm costs 0.115 and 0.066. On subject 1 the ranking
   is inverted outright.
2. **It does not track the silhouette-visible part of the shift.** A shift along a
   camera's viewing ray is depth and a silhouette cannot see it; the lateral part is what
   it can. Across the eight camera-subject cells the ray angle runs 28.8°–76.4° and the
   lateral component 9.3–36.5 mm, and its rank correlation with the IoU fall is
   **Spearman +0.38, Pearson +0.31** — the *wrong sign* for that mechanism. Eight cells is
   a reading, not a test, and is quoted as one. (B001/subject 0 has the smallest lateral
   shift, 9.3 mm, and the largest fall, −0.104; B001/subject 1 has the largest lateral
   shift, 36.5 mm, and one of the smallest falls, −0.044.)
3. **Something much larger than the root moved.** On the tilt ≤ 10° band the rig's Hips,
   head and feet moved 9.6–12.7 mm — and the delivered **hands moved 144–188 mm**, because
   D2 re-aims the clavicle chain and D2b amplifies it by moving the origin that aim is
   taken from. An arm is a large part of an outline.

### 13.4 The arms-only arm, measured rather than inferred

`delivered → D2` moves the **Hips by 0.00 mm** — D2 changes no root — and on subject 1's
tilt ≤ 10° band it moves the hands 108 and 127 mm and costs **0.0387 IoU, interval
[−0.0436, −0.0147], clear of zero.** So a clavicle re-aim of this size costs silhouette
agreement *on its own*, with the body held still. On subject 0 the same arm moves the hands
only 45 and 66 mm and costs +0.002 IoU, interval spanning zero.

**What is not isolated** is how much of D2b's further 0.054 IoU is its extra 74–88 mm of
hand travel and how much its 9.6 mm of body shift. A fourth build — the root fix carrying
the **pre-D2 clavicle anchor** — separates them exactly, and is the obvious next
instrument. It was not run: this pass was capped at one.

### 13.4b The gradient and the baseline are two different things

Per tercile, the delivered hands' displacement beside the body's and the IoU fall:

| | tercile | body moved | hands moved (D2b - delivered) | D2b - delivered IoU |
|---|---|---|---|---|
| subj 0 | upright | 13.6 mm | 144 / 177 mm | -0.0490 |
| subj 0 | middle | 27.5 mm | 137 / 187 mm | -0.0796 |
| subj 0 | most bent | 73.5 mm | 193 / 204 mm | -0.1151 |
| subj 1 | upright | 9.9 mm | 181 / 189 mm | -0.0631 |
| subj 1 | middle | 43.2 mm | 210 / 199 mm | -0.0497 |
| subj 1 | most bent | 99.7 mm | 176 / 161 mm | -0.0664 |

**Subject 1 settles the baseline and subject 0 does not settle the gradient.** On subject 1
the body displacement grows **ten-fold**, 9.9 -> 99.7 mm, and the fall does not move; the
hand travel is flat and so is the fall. On subject 0 the fall grows 2.3x while the hand
travel grows only about 30 % and the body displacement grows 5.4x, so **subject 0's
gradient is NOT explained by the arms**, and the trunk-axis offset -- plus the instrument's
own 0.040 IoU of extra difficulty on bent frames, which the oracle measures -- remains a
live contributor to it.

The correct statement is therefore two statements. The **baseline** fall, present at full
size where the body has moved 10 mm, is not the root placement. The **gradient**, which
exists on one subject of two, is consistent with the trunk-axis offset and with the
instrument, and this pass does not separate those two either.

### 13.5 Which mechanism the data support

**Neither of the two on offer, and the honest answer is a third.** It is not the
trunk-axis root offset: the fall is at full size on the frames where that offset is ~10 mm,
it does not track the part of the shift a silhouette can see, and on one subject it does
not grow with tilt at all. It is not simply "the mesh's shape" either — shape is the
*background level* (every arm sits at 0.52–0.66 IoU against the oracle's 0.71–0.88, and
that gap is D5/D6), but shape does not change between two builds of the same mesh, and
what moved between them is measured. **The fall travels with the arms**, and the one arm
that isolates them — a root-identical, clavicle-only change — costs IoU on its own with
its interval clear of zero.

That does not vindicate the root placement, and 13.4b is the limit of the claim: the
**baseline** fall is not the root placement, while subject 0's **gradient** may well be,
and this pass does not separate that from the instrument's own difficulty on bent frames.
What the pass does establish is that the silhouette is not the instrument that indicts the
root placement on the frames where the root placement barely moves anything, and that D2's
own clavicle re-aim carries a pixel cost nobody had measured until now. The missing pelvis frame remains a real modelling gap — the rig genuinely has no
pelvis-versus-thorax separation, and the offset genuinely rides the trunk axis at +0.93 —
but on this fixture it is not what the silhouette is pricing.

### 13.6 What this pass is blind to

It says **whether** the fall tracks tilt, never **which** placement is anatomically right;
no instrument in this lane does that yet. A silhouette cannot see depth, cannot see a
left/right mirror of a fore-aft symmetric pose, and cannot see anything inside the outline —
and our mesh covers only about two thirds of the mask on *every* arm, so a 1–10 cm change is
read against a large standing shape mismatch. Eight camera-subject cells is a reading, not a
test. The arm/root separation is not made; §13.4 names the build that would make it.

---

## 14. Which part of the body is the fall in? The part-wise cut, and the control I6 never had

Section 13 established that the silhouette's fall is present at full size on frames where
the whole rig has moved 10 mm and the delivered hands have moved 144–188 mm, and named the
build that would separate the two. This pass separates them a different way — by splitting
the mesh instead of the pipeline — and adds the control I6 has never carried.

`tools/compare/silhouette_partwise.py`, **instrument only**; `silhouette.py`'s outputs are
untouched and its rasteriser, scorer, mask store and posed-mesh caches are imported.
Regenerate: `PYTHONPATH=$PWD/src .venv/bin/python tools/compare/silhouette_partwise.py`
(per-frame scores cached per arm, so a re-cut is free). Folded into `gate.json` as
`d2b_root_placement.silhouette_partwise`; figure at `silhouette-partwise.png`. The
folded-arm control build lives in `artifacts/compare/d2-clavicle/folded-arms/` and was
exported through the **real** `export_animated_body_glb` and the **real**
`blender_export_mesh.py`.

**The split.** Dominant skin weight from the body asset the delivery was built from:
ARMS = the clavicle chain and below (Shoulder, UpperArm, LowerArm, Hand, every finger),
4364 of 13380 vertices and 8724 of 26756 faces; TORSO+LEGS = everything else, head
included. A face goes wholly to the part owning two or three of its corners, so the two
rasters partition the surface. The oracle is split the same way on SMPL-X weights, with
the collars included so both meshes are cut at the same anatomical place.

**Is the split labelling the right vertices?** The asset's weights label *asset* vertices
while the raster uses *Blender-exported* ones. Equal counts are only suggestive, so the
check is an anatomy-free one: under `delivered → D2`, where **the root moves 0.00 mm** and
only the clavicle chain changes, the arm set's median vertex displacement is **70.7 and
131.1 mm** while the torso set's is **exactly 0.0 mm**, with only 8.4 % of torso
vertex-frames moving at all. A scrambled vertex order would make the two sets random halves
of one mesh with identical distributions. (A weaker second check — agreement with a
nearest-FK-joint labelling — reads 0.83 / 0.89 against a shuffled-label chance level of
0.52 / 0.54. It does not reach 1.0 because nearest-*joint* and nearest-*weight* disagree at
the shoulder cap and the armpit. An earlier version of the gate row demanded ≥ 0.90 of that
second check; **that band was mine, invented without asking what the criterion does at the
shoulder, and it was replaced by the control-derived one. The substitution is recorded
rather than hidden.**)

### 14.1 The pre-registered predictions, and whether each was met

| | prediction | outcome |
|---|---|---|
| **1a** | torso+legs IoU unchanged `delivered → D2` | **NOT MET.** −0.0038 [−0.0128, −0.0009] and −0.0141 [−0.0154, −0.0031]. The premise was false: dominant-weight labelling says no *arm* joint is a vertex's largest influence, not that it has none. 8.4 % of torso vertex-frames move under a clavicle-only change. |
| **1a** | D2 → D2b on torso+legs is D2b's own cost | **MET AS A FIGURE, SUPERSEDED AS AN ISOLATION.** −0.0307 [−0.0505, −0.0194] and −0.0185 [−0.0348, −0.0100] — material. But that arm also carries the further clavicle re-aim through the same weight bleed, so §14.2 isolates the root exactly instead. |
| **1b** | arms-only precision falls `delivered → D2 → D2b` | **NOT MET, and refuted in direction on the first step.** It *rises* delivered → D2: 0.8175 → **0.8514** and 0.8086 → **0.8508**. D2b then lowers it to 0.7492 and 0.8263 — still above delivered on subject 1. Oracle ceiling 0.9285 / 0.8959. |
| **1c** | the arm-hidden-inside-the-body fraction falls `delivered → D2` | **NOT MET, and refuted in direction on both subjects.** It *rises*: 0.5076 → 0.5225 → 0.5730 and 0.5481 → 0.6030 → 0.6586. Oracle 0.4547 / 0.5936. |
| **2** | the folded-arm body scores HIGHER | **NOT MET — and this is the reassuring one.** Whole IoU −0.0384 [−0.0640, −0.0217] and −0.0420 [−0.0672, −0.0334]; recall −0.0436 and −0.0430. A limb-collapsed body scores **lower**. |

**The reading behind 1b and 1c is refuted in its direction, and the correction matters.**
The hypothesis was that D2 pulls the arms *out* onto a rig 190 mm too wide, losing
precision. Measured, D2 does the opposite: it tucks the arms *in* — the hidden fraction
rises and arm precision rises with it, because an arm inside the person is inside the mask.
What falls is **recall**: the real performers' arms are out where our rig's are no longer,
so the outline stops covering them. Arm precision up and whole-person recall down is one
event, not two.

### 14.2 The root's own silhouette cost, isolated exactly

`root_translation` enters forward kinematics only at the Root joint, so changing it
translates every skinned vertex by exactly that vector. Two arms therefore cost no render
to construct and split `D2 → D2b` exactly:

* **ROOT_ONLY** — D2's own rendered mesh moved by the per-frame root delta. Same pose, same
  mesh, same pixels but the translation.
* **CLAVICLE_ONLY** — D2b's mesh moved back. Same root as D2, D2b's arms.

| | ROOT_ONLY − D2 (whole person) | CLAVICLE_ONLY − D2 (whole person) | sum | measured D2 → D2b |
|---|---|---|---|---|
| subject 0 | **−0.0216** [−0.0332, −0.0041] | **−0.0379** [−0.0582, −0.0200] | −0.0595 | −0.0606 |
| subject 1 | **−0.0134** [−0.0457, −0.0013] | **−0.0225** [−0.0342, −0.0144] | −0.0359 | −0.0321 |

Torso+legs only: ROOT_ONLY −0.0230 and −0.0145; CLAVICLE_ONLY −0.0099 and −0.0046.

**Say it plainly: the root placement has a silhouette cost of its own, and no joint
instrument in this lane can see it.** About 0.022 and 0.013 IoU whole-person, both
intervals clear of zero, on a mesh that differs from its comparator by a rigid translation
and nothing else. It is roughly a third of D2b's own fall; the clavicle re-aim that D2b
induces is the other two thirds, and the two terms are near-additive.

**And it is tilt-dependent, which reconciles this pass with §13.** On the upright band
(tilt ≤ 10°, where the root moves 10–13 mm) the root's own cost on torso+legs is
**−0.0048** [−0.0075, −0.0048] on subject 0 and **−0.0011** [−0.0045, +0.0039] on subject 1
— the second spanning zero. §13 found the *whole-person* fall at full size on exactly those
frames (−0.0445 and −0.0928). Both are true: where the root barely moves it barely costs,
and the fall that remains there is the arms.

### 14.3 The control I6 never had

Both arms folded across the chest, built from the **delivered** track by re-aiming only the
two UpperArm locals with the converter's own `_world_for_bone`; root and every other joint
untouched.

| | whole IoU | whole precision | whole recall |
|---|---|---|---|
| subject 0, folded − delivered | **−0.0384** [−0.0640, −0.0217] | −0.0181 [−0.0316, +0.0207] | −0.0436 [−0.0610, −0.0300] |
| subject 1, folded − delivered | **−0.0420** [−0.0672, −0.0334] | −0.0308 [−0.0616, −0.0059] | −0.0430 [−0.0625, −0.0352] |

**The degenerate loses, so I6's whole-person figure is not passed by a limb-collapsed
body.** The prediction that it would go up is refuted, and the mechanism is recall: the
performers' arms are outside the torso in the mask, so hiding ours forfeits their pixels.
This is the "no gate a constant can pass" question asked of I6 itself, and I6 survives it —
narrowly, on one degenerate, on one take. It does not make a person mask a limb-*placement*
gate: precision alone barely moves on subject 0 (interval spans zero), so a mask still
cannot tell a well-aimed arm from a badly-aimed one *inside* the outline. What it can see is
an arm that stops covering the mask's arm.

### 14.4 Which branch the data put us on

**Both branches, with the arms about twice the root, and the root's share is real.** Over
the whole take D2b's fall splits −0.022 / −0.013 IoU to the root and −0.038 / −0.023 to the
clavicle re-aim it induces, on top of D2's own −0.008 / −0.018; on upright frames the root's
share collapses to −0.005 and ~0 while the fall persists at −0.045 and −0.093. So the
coordinator's first branch — the arms on a rig that is the wrong shape — carries most of it
and all of it where the root is still -- though "the arms" means the clavicle chain's whole
blast radius, which on subject 1 is felt in the *torso* part through weight bleed rather
than in the arm's own coverage. But the second branch is not empty: **the root placement
costs measurable silhouette agreement, tilt-dependently, and nothing else in this lane would
have found it.** Whether that cost means the placement is *wrong* is a different
question, and this instrument cannot answer it: a silhouette scored against a body 190 mm
too wide at the shoulders penalises moving anything toward where the cameras say it belongs.

### 14.5 What this pass is blind to

The part split cures "everything inside the outline is invisible" only for the two parts it
cuts: an arm wrongly placed but still inside the arm region is not separated from a right
one. Depth is still invisible, so is a left/right mirror of a fore-aft symmetric pose. The
folded-arm control tests **one** degenerate, and not the one it was set to test: passing it
says I6 penalises a wrong folded arm pose, not that a limb inside the outline is charged
for. That degenerate remains untested and needs a single-camera version. Nothing here
measures the arms' direction in 3D either — the hidden fraction is a projection.
Every figure here is scored against a mesh
whose shoulder span is 540 mm against 346 and 363 measured, and that mismatch is the
background all of it is read against — the oracle reaches 0.76 / 0.72 on torso+legs where
we reach 0.52 / 0.54. None of this says which placement is anatomically right.

---

## 15. D2c: the clavicle's temporal defect — pre-registration, written before the fix exists

Committed on its own, exactly as §0 was, so the history proves these are predictions and
not descriptions. Nothing was in hand when this was written but the D2 and D2b gate
reports, a reading of `positions_to_body_track`, I7's committed
`artifacts/compare/temporal.json`, and **one** measurement the brief ordered taken first
(§15.1) because it decides whether the step may proceed at all.

### 15.1 The go/no-go, taken before anything was pre-registered

D2c may only be selected on synthetic truth. That is worthless unless the synthetic arm
actually contains the phenomenon. The brief's rule: *if injected noise never brings the
landmark near the pivot, it is a different phenomenon and cannot select — say so and stop.*

I7's `fk_trajectory` (stride 3, two full-body SOMA-77 clips, 28 frames × 2 subjects
= **56 frames per seed**), through the real record builder, the real
`reconstruct_multiview` and the real converter, with I8's own noise definitions
(`thorax_window_sweep.NOISE_SIGMA_PX = 3.20 px at 1280`, our own detector's
self-agreement, never MAMMA's residual) injected into the 2D observations of the six
landmarks the clavicle reads:

| arm | clavicle-chain steps over 26.67°/frame | max step |
|---|---|---|
| clean, no noise | **0 / 324** | 11.81° (LowerArm); **shoulders 1.38°** |
| `heavy_tail` (i.i.d. in time) | 30 / 972 = 3.09 % | 63.89° |
| `heavy_tail_frame_correlated` | **45 / 972 = 4.63 %** | 66.57° |
| replayed real outage + `heavy_tail_frame_correlated` | 54 / 972 = 5.56 % | 66.00° |

The real take under D2b reads **40 / 894 (4.5 %)** and **33 / 894 (3.7 %)** on the two
performers. The frame-correlated arm reproduces it at 4.63 %. **GO**, on the arm the brief
names first, without needing the replayed-outage fallback.

Broken out over five seeds, the reject's own joints: **LeftShoulder 5 / 270 and
RightShoulder 5 / 270, 1.85 % of shoulder steps**; the four arm joints carry the rest
(1.85–4.81 %), which they inherit from the parent they hang off. Every number below is
pre-registered against those, not against the six-joint pooled rate.

### 15.2 The mechanism, as it will be built

A per-frame **reachability** rule on each clavicle's LOCAL rotation, with a replacement
policy and a re-solve:

1. Accept frame *t* iff its local rotation is within `CEILING × (t − t_last_accepted)` of
   the last accepted frame's. Frame 0 is accepted by definition. The envelope grows with
   elapsed frames, so a rejected run terminates on its own and needs no run-length
   constant. Both clavicles independently. **Not a step test:** a step test catches the
   transition into a wrong plateau and then accepts the plateau.
2. Replace each rejected run by shortest-arc slerp between the accepted endpoints; hold the
   last accepted value on a trailing run. Accepted frames are never touched.
3. **Replace, then re-solve.** `positions_to_body_track`'s single per-frame loop becomes
   three passes — (A) root, `Hips`, torso, neck, head, eyes and the two clavicles, per
   frame; (B) the sequence reject on the clavicle locals; (C) every downstream chain
   against the replaced parent, per frame. Without (C) the whole arm swings rigidly with
   the replaced clavicle and the elbow and wrist leave their landmarks.
4. `CLAVICLE_MAXIMUM_FRAME_TRAVEL_DEG_PER_S = 800.0`, converted at the track's own sample
   rate — 26.67°/frame at 30 fps. The same physical figure the head lane's
   `MAXIMUM_FRAME_TRAVEL_DEG` and the D2/D2b gates already use. A reject ceiling, not a
   knob: it must never fire on true motion.

### 15.3 The predictions

**On the real take (reported, never banded — frames-over-ceiling is exactly what a reject
zeroes):**

* **P1.** Clavicle-chain frames over the ceiling and the worst single step FALL. The
  **median** step does NOT: the median is the short lever D2 and D2b created, and D2c does
  not touch it. Expect the D2b medians (LeftShoulder 1.90° / 1.53°) to move by less than
  0.1°, and the D2b over-ceiling counts (40 / 33 canonical) to fall.
* **P2.** After D2c the two `Shoulder` joints have **zero** steps over the ceiling **by
  construction** — consecutive accepted frames are inside one envelope step and the slerp
  divides a run's angle evenly. That is arithmetic, not evidence, and it is written here so
  nobody reads it as one. Any residual chain count is `UpperArm`/`LowerArm`, which inherit.
* **P3.** Accepted frames are bit-identical to D2b. Only rejected runs and their downstream
  chains change; every joint outside the clavicle chain — legs, feet, toes, neck, head,
  eyes, and the finger locals — is bit-identical to D2b on every frame.
* **P4.** Rejected runs are **short and bounded**: the envelope reaches 180° at
  `ceil(180 / 26.67) = 7` frames, so **no rejected run can exceed 6 frames** and an outlier
  first frame recovers within **7**. Those are theorems; the measured distribution is
  reported beside them. The rejected fraction is a few percent (the probe says ~2 % of
  shoulder steps).
* **P5.** The delivered arms on our own capture and on arm B are not regressed beyond the
  paired moving-block bootstrap (block 15, 2000 draws, identical draws) against D2b, and
  rung 11 against MAMMA is unmoved beyond its interval. The reject touches ~2 % of frames;
  a whole-take median should barely move.
* **P6. A correction to the brief.** The brief asks the canonical and sized round trips to
  "reproduce D2b to 0.000 mm". That cannot be right as written: D2c changes pass 1 on
  rejected frames, so the FK'd landmarks pass 2 sees change too, and the round-trip
  statistic must move a little from D2b's 0.51 / 0.08 (canonical) and 0.07 / 0.04 (sized).
  What IS exactly true, and what is banded here instead: **on identical pass-1 input the
  reject fires on ZERO frames in pass 2** (the synthetic body is noise-free by
  construction, so nothing is unreachable) **and pass 2 is bit-identical with the reject on
  and off.** The round-trip statistic is predicted to stay under 2 mm on the canonical rig,
  reported not banded.
* **P7.** The D2 and D2b regression checks still reproduce their committed reports to
  0.000 mm; I6's silhouette is REPORTED and not banded.

**On synthetic truth (the selector — must PASS):**

* **P8. Exactness on clean input.** On the clean fixture the reject fires on **zero**
  frames on both clavicles and the track is bit-identical to no-reject. The observed true
  clavicle peak is **1.38°/frame = 41°/s**, a 19× margin under the ceiling.
* **P9.** On the noisy arm (≥ 5 seeds), clavicle local rotation error against truth —
  truth being *the converter's own output on the noise-free landmarks* — has **max better**
  and **median not worse** (within 0.05°). **p95 is the marginal band**: the reject touches
  ~2 % of shoulder frames and p95 is the top 5 %, so the two populations only partly
  overlap. It is pre-registered as "p95 better", as the brief asks, and if it comes back
  flat that is a real failure of a real band and will be reported as one, not reworded.
* **P10. Attenuation, and a second correction to the brief.** The brief asks for
  "attenuation no worse than I7's shipped-window figure" (0.3183). That figure is the
  attenuation of **3D joint displacement over 17 joints**; ours is the attenuation of
  **clavicle angular speed**. They are different quantities, they do not share a reference
  and they must never share an axis; I7's number is reported beside ours for shape only.
  The bands that bite: on **clean** input D2c's attenuation of true clavicle peaks is
  **exactly 0** (P8 makes it bit-identical); on the **noisy** arm it is **no worse than
  D2b's on identical draws**.
* **P11. Lag zero.** Shift sweep, as I7 does it — the same fixed frame set at every shift.
  D2c's argmin is 0 frames. The reject is not a filter and introduces no phase.

**Controls (must FAIL, all through the identical code path):**

* **P12. The over-smoother** — a zero-phase Savitzky-Golay of the clavicle locals at
  3 × I7's shipped window (27) — wins on **max** and loses on **attenuation**. It cannot
  lose on lag: zero-phase means lag is zero by construction, and I7's own window-27 arm
  reads argmin 0. That is exactly why it is a control and not the fix, and why the
  discriminating band is attenuation. (On a 28-frame take a 27-frame window is close to a
  global parabola; recorded so nobody reads more into its max than is there.)
* **P13. A third correction to the brief. The brief's 5°/frame low ceiling is INERT on
  this fixture.** The true clavicle peak is 1.38°/frame, so a 5°/frame ceiling fires on
  zero true frames and the must-fail control would pass. The control that discriminates has
  to sit **below the fixture's own true peak**: **1.0°/frame (30°/s)**, which fires on true
  frames of the clean fixture and attenuates true motion. The general lesson, and it is the
  reason the substitution was needed: *a reject ceiling's "no gate a constant can pass"
  demonstration needs true motion faster than the constant, and this fixture's clavicle is
  slow.* Both ceilings are run and both are reported.
* **P14. The world-measured variant** — the same reject read on the clavicle's WORLD
  rotation — rejects honest motion on a synthetic turning body. The local clavicle rotation
  is exactly invariant to a whole-body yaw, so the yaw is applied to the noise-free
  positions directly. Two rows: a **pirouette at 12°/frame (360°/s)**, where neither
  variant fires, and a **scratch spin at 45°/frame (1350°/s)**, where the world variant
  rejects nearly every frame and the local one still rejects none. The head lane's lesson,
  demonstrated rather than asserted.
* **P15. The step test** — accept unless the *single* step exceeds the ceiling — accepts
  the plateau of a synthetic multi-frame excursion that reachability rejects whole. The
  excursion must be larger than `3 × 26.67 = 80°` for a three-frame plateau to be
  unreachable; ~100° is used, which is the size of the real pops (139–164°).

**The oracle (reports, selects nothing).** MAMMA's collar joints (SMPL-X 13 / 14 against
shoulders 16 / 17) as a clavicle DIRECTION through the same ceiling: the floor a real
clavicle sequence shows on this take. It is a **direction** step (2 degrees of freedom),
ours is a **rotation** step (3, including twist); the two never share an axis. Subject
correspondence through `tools/head/subject_map.py`, never the index, and never a `gt_*`
array.

### 15.4 What this will be blind to, written now

The median step is the short lever and stays: D2c removes unreachable transitions, not the
noise that produces them, and the lever is D5's. The silhouette's arm share is D6's. A
wrong clavicle plateau that is *reachable* is invisible to a reachability rule by
construction — the rule can only see physics violations, not errors. Anything selected on
synthetic truth inherits the fixture's blindness to calibration error, camera sync and soft
tissue, and — specific to this fixture — its clavicle moves **slower than the real take's**
(true peak 1.38°/frame against a real-take median of 1.5–1.9°/frame), so "the ceiling never
fires on true motion" is established above *this fixture's* clavicle motion and not above a
sprinter's.

### 15.5 What shipped

`src/autoanim_gnm/commercial_multiview.py`, three things and one constant:

* `CLAVICLE_MAXIMUM_FRAME_TRAVEL_DEG_PER_S = 800.0` — the head lane's own physiology,
  converted at the track's sample rate (26.67°/frame at 30 fps). Registered in
  `tools/compare/provenance.py` as **ANATOMY**, not as a category called "physical": the
  taxonomy has no such value, the head's identical `MAXIMUM_FRAME_TRAVEL_DEG` is ANATOMY,
  and adding one would move `tests/test_provenance_audit.py`, which is the registry
  owner's file. *Deviation from the brief, recorded.*
* `_reachable_clavicle_sequence(local, parent_world, ceiling)` — module level and called by
  bare name, so every control in the gate runs through the identical call site. The
  `parent_world` argument is unused by the shipped rule and is in the signature so the
  world-measured control can be substituted rather than re-implemented.
* `positions_to_body_track`'s single per-frame loop, now three passes: (A) root, `Hips`,
  torso, neck, head, eyes and the two clavicles, per frame; (B) the sequence reject;
  (C) every downstream chain re-solved against the replaced parent, per frame.

**The restructure changes nothing, and that is checked rather than argued.** With the rule
swapped for a pass-through, the converter reproduces the pre-D2c code **bit for bit** on
both performers and both rigs (`artifacts/compare/d2-clavicle/d2b-baseline-tracks.npz`;
the gate re-checks it on every run). Getting there took one non-obvious line: `_set_world`
normalises its argument **in place**, and without a head solve `neck_world`, `head_world`
and the eyes' argument are all the same array as `torso_world` — so by the time the feet
were handed it, it had been through seven normalisations. Freezing the torso frame for
pass C any earlier hands the feet a differently-rounded quaternion, and the 1e-17 gap
survives the float32 cast in the toes' near-identity local rotation. Every "before" arm in
the gate is that pass-through, so the two arms of every comparison are one converter with
one attribute swapped.

### 15.6 The gate

`tools/compare/d2c_clavicle_temporal_gate.py` → `artifacts/compare/d2-clavicle/gate-d2c.json`.
**Overall: FAIL**, on two bands, and neither failure is a defect in the mechanism — one is
a band that is not interpretable on the arm it was pre-registered against, and one is a
prediction the code refuted. Both are below, with what replaced them.

| band | before | after | verdict |
|---|---|---|---|
| P8 exactness on clean synthetic input | 0 rejects | 0 rejects, bit-identical, attenuation 0.0000 | **PASS** |
| P9a clavicle error vs synthetic truth, median | 9.4106° | 9.4001° | **PASS** |
| P9b same, p95 | 54.85° | **52.70°** | **PASS** |
| P9c same, max | 90.40° | **89.10°** | **PASS** |
| P9d same, on the frames the reject touches (n = 40) | 90.40° | **66.62°** | **PASS** |
| P10 attenuation of true clavicle peaks, noisy, identical draws | −2.1855 | −2.1855 | **PASS** |
| P11 phase lag zero, on the NOISY arm | −1.5 | −1.5 | **FAIL** |
| P12 CONTROL over-smoother must fail on attenuation | 0.0000 | **0.6352** | **PASS** |
| P13 CONTROL a ceiling below the true peak must fire on true motion | 0 | **10 of 112** | **PASS** |
| P14 CONTROL world-measured reject on a body spinning at 1350°/s | 0 | **44 of 56** | **PASS** |
| P15 CONTROL a step test must accept what reachability rejects | 25 | 17 (**9 accepted**) | **PASS** |
| GO/NO-GO the synthetic arm reproduces the real flip rate | 4.47 % real | 2.01 % synthetic | **PASS** |
| P3 accepted frames bit-identical; only the clavicle chain moves | — | — | **PASS** |
| P4 no rejected run exceeds the envelope bound (6) | 6 | 5 | **PASS** |
| P5 delivered arms, subj 0 canonical / sized | 50.62 / 24.99 mm | 50.79 / 27.19 mm | **PASS** |
| P5 delivered arms, subj 1 canonical / sized | 30.28 / 23.56 mm | 30.50 / 23.75 mm | **PASS** |
| P6 the round trip's second pass rejects nothing | 0 | 12 and 4 | **FAIL** |
| P7 committed D2 / D2b figures through the pass-through | 0.000 mm | **0.000 mm** | **PASS** |
| P1 REPORTED, no band: clavicle-chain frames over the ceiling | 40 / 33 | **14 / 13** | report |
| P1 REPORTED, no band: MEDIAN clavicle step | 1.90 / 1.53° | **1.90 / 1.53°** | report |

Delivered-arm margins are paired moving-block bootstraps, block 15, 2000 draws, identical
drawn frames: subject 0 canonical −0.25 mm [−1.43, 0.00], sized −0.80 [−6.23, 0.00];
subject 1 canonical −0.04 [−0.63, 0.00], sized −0.21 [−2.58, 0.00]. Every interval touches
zero: the reject moves 1–9 % of clavicle frames and a whole-take median cannot see it.

### 15.7 The real take, and the pre-registered expectations

Clavicle-chain frames above 26.67°/frame, canonical rig, over 894 steps:

| | pre-D2 | D2 | D2b | **D2c** |
|---|---|---|---|---|
| subject 0 | 32 | 48 | 40 | **14** |
| subject 1 | 11 | 49 | 33 | **13** |

Per joint, median / p95 / max step in degrees, D2b → D2c:

| joint | D2b | D2c | frames over the ceiling |
|---|---|---|---|
| subj 0 LeftShoulder | 1.90 / 12.94 / **138.98** | 1.90 / 9.78 / **17.69** | 4 → **0** |
| subj 0 RightShoulder | 1.71 / 33.82 / **164.48** | 1.71 / 24.68 / **25.75** | 12 → **0** |
| subj 0 LeftUpperArm | 3.24 / 11.27 / 168.15 | 3.24 / 11.03 / **18.19** | 4 → 0 |
| subj 0 RightUpperArm | 2.62 / 38.75 / 164.65 | 2.62 / 27.80 / 38.05 | 12 → 8 |
| subj 1 LeftShoulder | 1.53 / 15.94 / 39.34 | 1.53 / 14.02 / **24.53** | 7 → **0** |
| subj 1 RightShoulder | 1.99 / 21.06 / 53.57 | 1.99 / 18.32 / **26.47** | 6 → **0** |

**P1 met.** The worst single step falls from 164.48° (4934°/s) to 25.75°, and the median
does **not** move — 1.90 → 1.90 and 1.53 → 1.53, to the reported precision, on both
clavicles. The median is the short lever and it stays; that is D5's.

**P2 met and it is arithmetic.** Both `Shoulder` joints read **zero** frames over the
ceiling after D2c, by construction: consecutive accepted frames lie inside one envelope
step and a slerp divides a run's angle evenly. It is not evidence and it is written here so
nobody reads it as any. The residual 14 and 13 are `UpperArm` and `LowerArm`, which inherit
their parent's frame and carry their own landmark noise, which D2c does not touch.

**P3 met.** The differing local rotations are exactly `{Left,Right}×{Shoulder, UpperArm,
LowerArm, Hand}` on both performers. `root_translation_m`, `foot_contacts`, the
triangulated positions, the legs, the neck, the head, the eyes, the toes and every finger
local are bit-identical on every frame, and the accepted clavicle frames are bit-identical
to D2b's. The eight cached `*observations.jsonl` are byte-identical to the delivered
build's: nothing re-extracted and nothing re-detected.

**P4 met.** Rejected fractions 1.3 %, 8.7 % (subject 0) and 3.3 %, 2.7 % (subject 1); run
lengths 1–5 against the theorem's bound of 6.

**P5 met**, above. **P7 met**: every committed D2 and D2b figure reproduces to **0.000 mm**
through the pass-through rule.

### 15.8 Synthetic truth, which is the selector

I7's `fk_trajectory` (stride 3, 28 frames × 2 subjects per seed), the real record builder,
the real `reconstruct_multiview`, the real converter, **20 noise seeds** of I8's
`heavy_tail_frame_correlated` at our own detector's measured amplitude. Truth is the
converter's own output on the noise-free landmarks. 2240 clavicle frames per arm; every arm
sees the *same* noisy positions, so the denominators are identical by construction.

| arm | median | p95 | max | on the 40 rejected frames (med / p95 / max) | attenuation | lag |
|---|---|---|---|---|---|---|
| D2b (pass-through) | 9.41° | 54.85° | 90.40° | 36.34 / 80.71 / 90.40 | −2.1855 | −1.5 |
| **D2c** | **9.40°** | **52.70°** | **89.10°** | **28.02 / 62.57 / 66.62** | −2.1855 | −1.5 |
| CONTROL over-smoother, window 27 | 8.55° | 43.82° | 85.29° | 22.99 / 62.44 / 67.48 | −0.2163 | −2.0 |
| CONTROL ceiling 1.0°/frame | 9.21° | 31.17° | 65.52° | 25.70 / 38.70 / 64.80 | −0.4196 | −3.0 |
| CONTROL step test | 9.40° | 53.02° | 90.40° | 30.11 / 80.71 / 90.40 | −2.1855 | −1.5 |

On the **clean** fixture the true clavicle peak is **1.3837°/frame = 41.5°/s**, a 19.3×
margin under the ceiling; D2c rejects 0 frames, the track is bit-identical to no-reject,
attenuation is 0.0000 and lag 0. The 0.0446° "error" floor there is float32 `arccos` noise
on identical quaternions, not an error.

**The two controls that carry the argument, on clean input:** the over-smoother replaces
every frame, destroys **63.5 %** of the true clavicle peak and reads a lag of −0.5; the
1.0°/frame ceiling fires on **10** true frames and destroys 14.2 %. Both would sail through
"frames over the ceiling", which is exactly why that figure is reported and never banded.

**P9 met on every statistic, and the interesting row is P9d.** The pooled max improves only
89.10 from 90.40, because the worst *error* frame is a reachable one the reject does not and
should not touch. On the 40 frames the reject does touch — scored identically on all five
arms — the max falls **90.40 → 66.62°** and the p95 **80.71 → 62.57°**. The step test on
the same 40 frames leaves both at D2b's values, because it accepts the plateau.

**P10 met, and the figure needs its caveat.** Attenuation on the noisy arm is negative
(−2.19) on *every* arm that is not a filter: noise makes the clavicle move faster than
truth, so "attenuation" measures the noise, not the rule. The band that bites is the clean
one, where D2c is exactly 0 and the over-smoother is 0.6352.

**P14 and P15 met.** A body yawing at a pirouette's 12°/frame (360°/s) fires neither
variant; at a skater's 45°/frame (1350°/s) the world-measured variant rejects **44 of 56**
frames and the local one still rejects **none** — the head lane's lesson, demonstrated
rather than asserted, on a clavicle held perfectly still. The step test rejects 17 frames
where reachability rejects 25, and **9** of the frames it accepts are ones reachability
rejects: the transitions caught, the wrong plateau kept.

**The GO/NO-GO, and the honest thing about it.** Pooled over 20 seeds the synthetic arm puts
**2.01 %** of clavicle-chain steps over the ceiling against the real take's 3.7–4.5 %. But
the per-seed counts are `[6, 38, 1, 0, 0, 0, 18, 0, 0, 23, 4, 7, 0, 22, 4, 0, 5, 0, 2, 0]`
— **9 of 20 seeds contain no flip at all**, and one seed contains a quarter of them. The
phenomenon is a heavy-tail event: a flip needs a tail draw to land on a shoulder while the
lever is short, and a 28-frame take usually contains none. The five seeds the brief asked
for would have been one seed's number; that is why this ran at 20, and the spread is in the
report rather than smoothed over.

### 15.9 Three predictions the code refuted, and one band that could not be read

**P11 FAILED as pre-registered, and the band is not interpretable on the arm it was written
against.** On the noisy arm the lag sweep reads −1.5 frames for D2b, D2c *and* the step
test alike — it is a property of the noisy positions, not of any rule. On the clean fixture,
where a phase question is answerable, D2b and D2c both read **0** and the over-smoother
reads −0.5. Both readings are in the report as `P11a` and `P11b`; **neither is a substitute
for the FAIL**, and the band is left exactly as it was written. What can honestly be
claimed is the weaker statement: *D2c adds no phase of its own* — its lag is identical to
D2b's to the frame on the noisy arm and identical to zero on the clean one.

**P6 REFUTED, and the reason is the round trip's own self-reference.** Pass 2 does not
reject nothing: it rejects 12 and 4 frames. With the reject switched off for pass 2 alone,
its raw clavicle steps peak at **33.52°** and **33.37°** — an order of magnitude below the
real capture's 139–164°, and still over the ceiling. The cause: `landmarks_from_fk` writes
the rig's **UpperArm origins** into the `left_shoulder` / `right_shoulder` slots, and the
converter builds the **torso frame** from exactly those two landmarks. So a clavicle the
reject replaced moves the arm roots, which moves the frame pass 2 measures the clavicle's
*local* rotation against, and the local picks up a jumping twist while the arm root itself
barely moves. The round-trip statistic is untroubled: canonical 0.51 → 0.55 and 0.08 → 0.08
mm, sized 0.07 → 0.07 and 0.04 → 0.05. **The round trip cannot see this step**, which is
§7's self-reference blindness in miniature, and it is the second half of P6's correction to
the brief: not only can it not reproduce D2b to 0.000 mm, it cannot serve as a "the reject
fires zero times" check either.

**The brief's 5°/frame low-ceiling control is inert here**, pre-registered as P13 before the
run and confirmed by it: the fixture's true clavicle peak is 1.38°/frame, so a 5°/frame
ceiling fires on nothing and the must-fail control would have *passed*. 1.0°/frame was
substituted and both are reported. The general lesson: **a reject ceiling's "no gate a
constant can pass" demonstration needs true motion faster than the constant.**

### 15.10 The oracle, and the arms nothing here touches

**MAMMA's own collar → shoulder direction**, through the identical ceiling — the floor a
real clavicle sequence shows on this take. It reports and selects nothing; it is a **2-DoF
direction** step where ours is a 3-DoF rotation step including twist, and the two never
share an axis.

| | median | p95 | max | over the ceiling | collar→shoulder |
|---|---|---|---|---|---|
| subj 0, left / right | 2.41° / 1.61° | 4.79° / 4.37° | 5.44° / 4.96° | **0 / 0** of 149 | 146 / 128 mm |
| subj 1, left / right | 1.59° / 1.90° | 5.25° / 5.49° | 5.88° / 6.73° | **0 / 0** of 149 | 146 / 128 mm |

A real clavicle sequence on this footage never comes within a factor of four of the ceiling.
That is what makes 139–164° a defect rather than a performance.

**Rung 11**, absolute capture world, MAMMA's `pred_joints`, the scoreboard's own statistic,
paired moving-block bootstrap on identical draws, D2b → D2c: subject 0 canonical arms
83.72 → 83.62 (margin 0.13 [0.00, 2.16], p 0.26), legs 58.62 → 58.62 (0.00, exactly), all
15 joints 71.30 → 71.30; subject 1 arms 68.34 → 68.29, legs 63.54 → 63.54, all 67.92 →
67.82. Nothing moves, which is the expectation: the reject touches a few percent of frames.
**Arm B** (MAMMA's joints in) is *bit-identical* on both rigs and both subjects — that input
contains no unreachable clavicle frame at all, which is its own small oracle.

**I6, the silhouette** — reported, never banded, on the only reference in this lane that is
not another model. Median IoU over the eight camera-subject cells **0.5191 (D2b) →
0.5207 (D2c)**, up in 7 of 8. MAMMA's own mesh through our scoring path is unchanged to
0.000000, so the two runs are comparable. It is a small rise and it is *not* evidence for
D2c: 2 mm of arm on 8 % of frames is inside this instrument's noise.

### 15.11 The picture, which is what priced the pop correctly in the first place

`artifacts/compare/d2-clavicle/jitter-d2c/d2c-zoom-subject01-f047-050.png` and
`d2c-contact-sheet-subject01-LeftShoulder.png` — subject 1, camera A001, frames 44–54, D2b
on top and D2c beneath, rendered through `tools/compare/d2_jitter_sheet.py`'s own overlay
(its module-level paths rebound, not edited), which sets the scene fps from the track before
the glTF import.

**What the eye sees:** in the D2b row every tile from f47 to f50 is boxed red — 39.3°, then
27.6°, 28.5°, 29.7° — and the outstretched arm jerks unevenly through them, the shoulder
buckling as it goes; in the D2c row the same four frames carry an identical, even 24.5° per
frame, the arm sweeps down through the same range in one continuous motion, and the shoulder
holds its shape. The steps the reject removed are the *unevenness*, not the travel.

### 15.12 What D2c is blind to

* **The median step, which is the lever.** D2c removes unreachable transitions, not the
  noise that produces them. The typical frame-to-frame clavicle turn is unchanged to two
  decimal places, and it is 1.5–1.9° because the pivot is 100–160 mm from the landmark. That
  is D5's, and nothing here shortens the wait for it.
* **A wrong plateau that is reachable.** A physical reject can only see physics violations.
  A clavicle that is 30° wrong and stays there is invisible to it by construction, and no
  band in this gate would notice.
* **The arms below the clavicle.** Their own landmark noise is untouched; the residual 14
  and 13 over-ceiling frames are `UpperArm` and `LowerArm`.
* **Whatever the synthetic fixture does not contain** — calibration error, camera sync, soft
  tissue — and specifically **its clavicle is slower than the real take's** (true peak
  1.38°/frame against a real-take median of 1.5–1.9). "The ceiling never fires on true
  motion" is established above *this fixture's* clavicle and not above a sprinter's. The
  MAMMA oracle above is the only reading of a *real* clavicle's rate here, and it agrees.
* **The fixture's length.** 28 frames × 2 subjects per seed. The over-smoother's window 27
  is close to a global parabola on it, and the flip phenomenon is present in only half the
  seeds.
* **The round trip**, which is structurally unable to score this step (§15.9).
* **The silhouette's arm share**, which is D6's.

### 15.13 In plain language

The collarbone fix put the arm in a much better place and made it travel there badly. On
four frames of the delivered take the arm did something no shoulder can do — turning 164
degrees between one frame and the next, about five thousand degrees a second — and on
another seventy-odd it moved faster than a human joint can. A score that measures *where*
the arm ends up prices all of that at zero, because on both sides of the jump the arm is
roughly where it should be.

This step throws those frames away. It asks, of each frame, whether a real collarbone could
have got there from the last frame it believed, at 800 degrees a second — the same speed
limit the head already uses — and if not it draws a smooth line across the gap and re-solves
the elbow, wrist and hand so they still land on what the cameras saw. Frames it believes are
passed through untouched, bit for bit. Impossible frames: 40 and 33 before, **14 and 13**
after, and none at all on the collarbones themselves. The worst single frame goes from 164
degrees to 26.

Two things are worth knowing. The everyday shake is **not** fixed, and was never going to
be: it comes from measuring the collarbone's direction over a 10–16 cm distance instead of
40, and the typical frame-to-frame wobble is identical before and after, to two decimal
places. And the speed limit itself was chosen from anatomy, never from data: it was tested
on a synthetic performer whose true pose we know exactly, where it correctly does nothing at
all, and against two deliberately wrong answers run through the same code — a heavy smoother
that throws away two thirds of the performer's real motion, and a limit set below what this
body actually does, which throws away a seventh. The research fitter we compare against
never exceeds a quarter of the limit on this footage, which is what makes 164 degrees a
fault and not a performance.
