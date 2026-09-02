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
