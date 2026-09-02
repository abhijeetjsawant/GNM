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

*(Everything below is filled in from `artifacts/compare/d2-clavicle/gate.json` after the
gate runs. Numbers come only from the JSON.)*
