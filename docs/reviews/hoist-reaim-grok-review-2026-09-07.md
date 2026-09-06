# Grok 4.6 (Cursor, ask mode) review of the D9b card — 2026-09-07, in Sol's place

Brief: the state, the code excerpts (the projection, the converter tail, the root line and `_joint_origin`, the D7b trunk aim, the clavicle aim / pass B / the D9 arm aim, `_reachable_clavicle_sequence`, `retarget_cost.score`), the pre-card measurement on the delivered files and the six oracle bodies, the card as first drafted, eight questions. Verified against the source before adoption: pass C rewrites the legs, feet and toes (`commercial_multiview.py` ~:2817–2871) and would overwrite the projection's `candidate_local` foot lock; the global quaternion hemisphere pass (~:2883–2886) walks from frame 0 and can flip the bits of the frame after a rewritten one; `groups_mm` is a median over concatenated joint-frames (`d3_skeleton_gate.py` :285); `oracle_block` passes no `head_world_rotations`. What each finding changed in the card is listed at the end.

---

The chosen mechanism is the right one. The card still has a few places where the merge rule and the printed oracle statistic do not do what the text claims.

## 1. Mechanism / projection consistency

**None**, if the re-solve is exactly what the card names: D7b trunk from the hoisted `Spine` origin, neck slerp, clavicles, pass B, the four arm bones, hands/fingers/eyes, and the projection’s **root, contacts, and foot/toe locals are never written again**.

Hips come from the pelvis Kabsch (`_pelvis_world_frames`). Legs are landmark-to-landmark about `Hips`. Feet are ankle→ball plus shin (toes are supplied on the delivery and on the D3 oracle). None of those parents are the trunk/clavicle/arm chain, so a trunk re-aim does not move the FK the detector used.

**The case that breaks it is a literal re-run of pass C.** Pass C still does this:

```2817:2871:src/autoanim_gnm/commercial_multiview.py
        for joint_name, child_name, direction in (
            ("LeftUpperLeg", "LeftLowerLeg", at("left_knee") - at("left_hip")),
            ...
        ):
            ...
            _set_world(local, world, frame, joint_name, target_world)
        ...
            _set_world(local, world, frame, foot, foot_world)
        for toe in ("LeftToes", "RightToes"):
            ...
            _set_world(local, world, frame, toe, world[frame, parent])
```

That overwrites `project_generated_foot_contacts`’s `candidate_local` lock. Contacts and `roots += correction` would still be the old answer; the delivered feet would not. The plant would be a translation without a locked foot.

Same class of bug: re-entering pass A’s `root_translation[frame] = pelvis - rest["Hips"] - _leg_root_offset(...)` (around 2613) wipes the hoist.

**Change:** the src block must be a dedicated re-solve of the root-dependent chain, not “call pass C again.” If toes were ever missing, `foot_world = torso_world` would also couple feet to the new trunk; that path is not this take, but it is why the feet loop must not run.

---

## 2. What the candidate can optimise

| Clause | Candidate can set it directly? |
|---|---|
| **B1** excess over the arm-aim floor → 0 | **Yes.** That is the re-aim. |
| **O1** absolute ray miss → unhoisted floor | **Yes.** Same quantity on the oracle. |
| **B2 / O2** root, contacts, hips, legs, feet | Only by *not* writing them (construction). |
| **B4** contact count/runs/plant travel | Same: do not re-run projection. |
| **O4 / tripwire** zero-hoist bit-identity | A wrong float order fails it; a correct re-aim cannot “score” it. |
| **B3** silhouette | No. Not in the loop. |
| **B5 / O3** | Report only. |

The heading “bands the candidate cannot optimise” is wrong for **B1 and O1**. Those are the defect-closed proofs.

The merge rule is `hygiene ∧ tripwire ∧ B1 ∧ B2 ∧ B3 ∧ B4 ∧ O1 ∧ O2`. It **does** rest on quantities the candidate controls (B1, O1). That is acceptable *only* because B2/B4/O2/tripwire are the dual: they fail the cheats that zero B1 by undoing the plant. A merge that was only B1 would be a constant you can pass.

**Change:** relabel B1/O1 as closure, not as “cannot optimise.” Keep them in the merge *conjunct* with B4 and B2.

---

## 3. Degenerates

**(a) Re-aim, then re-derive the root from the re-aimed body.**  
If that re-derivation is the converter line `pelvis - rest[Hips] - offset`, the hoist is undone. Arms then point along `target − origin_hoisted` from the *pre-hoist* origin: **B1 fails**, **B2 fails** (root), **B4 plant travel fails**.  
If they iterate re-aim + re-root to a fixed point, they recover the pre-projection solve: **B1 passes**, **B2/B4 fail**.  
**Caught.**

**(b) Re-aim only frames with hoist > 0.5 mm.**  
On *this* take that partition is exhaustive (67+83 and 22+128). It is the intended split, not a cheat. **No clause catches it here.** On a take with 0.2 mm hoists, B1’s “excess → 0 on **every** frame” would catch it.  
**Change:** 0.5 mm is a **report** cut, not control flow. Re-aim every frame; the tripwire is what says the ~0 frames do not move.

**(c) Skip pass B, keep old clavicle locals.**  
Shoulder (and then arm) rays stay at the measured 4.4–6.3 mm. **B1 on shoulders/trunk fails.** That is the card’s third must-fail.  
Skip pass B *after* per-frame clavicle re-aim: on this take pass B is a no-op (oracle rejected `[0,0]`; take margin 11.6–20.8°). **Nothing catches it.** Harmless here; not a proof the sequence rule was rerun.  
**Change:** B2’s “accepted set identical” does not prove the rerun happened. The tripwire plus a before/after dump of accepted masks is enough; do not pretend skip-B is tested on this fixture.

**(d) Hoist = 0, contacts still on** (lock foot locals, skip `roots += correction`).  
**B1 and O1 pass** (origins match the original aims). **B2 fails** (root ≠ D8c). **B4 plant travel fails** (the lock without the translation lets the FK foot slide). The card’s `maximum_root_correction_m=0` degenerate is *different*: that path `continue`s before `contacts[...]=True`, so contacts go `(0,0)`. (d) is the one B4’s *travel* clause is for, and it is the one that matters on this take because `penetration_before` is 0.

---

## 4. Oracle / `retarget_cost.score`

Sign of the **geometry** is right. Pure translation cancels in `score` (leg-root midpoint minus truth hips), which is why pre / delivered / hoist-removed groups are identical today. After a re-aim, first order, the aligned residual is the **perpendicular** part of the hoist (`e ≈ −h_⊥`), i.e. the 2.6–6.7 mm ray miss already measured, not the 10–14 mm p95 `|h|`.

**The printed D3 figure may not get worse.** `groups_mm` is a **median** over concatenated joint-frames:

```285:289:tools/compare/d3_skeleton_gate.py
def groups_mm(error: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        group: round(float(np.median(np.concatenate([error[n] for n in names]))), 2)
```

Every seed has **34–67 / 150** frames hoisted — always a minority. Unhoisted frames keep today’s ~0.8–1.17 mm. The group median can **hold**. p95 or the hoisted-frame subset will move. O3’s “predicted WORSE” on the number `worst_arms_mm` reads is therefore **not guaranteed**.

**Do not change the gauge in this step.** The arm band is already a standing fail (0.5 mm vs 0.80–1.17). Re-pinning the aligned score in the same commit as the candidate is the clavicle-round-trip mistake again. The absolute-frame companion row is the right instrument. If the printed median does not move, that is the statistic, not a surprise about the fix.

**Change:** O3 should predict “aligned **median** may be unchanged; aligned **p95 / hoisted-frame median** worse by ~3–7 mm; legs identical.” Leave the band. O3 stays **report**, not merge — good, because a wrong “WORSE” prediction must not block or invite a band move.

---

## 5. B2 on unhoisted frames

Landmarks byte-identical ⇒ D3 whole-take **sizing** does not enter. `PELVIS_SMOOTHING_FRAMES = 0`. `THORAX_SMOOTHING_FRAMES` lives in `_thorax_frames` for the **head** reference, not in `positions_to_body_track`’s torso aim. The head solve is an input; it does not see the hoist.

**The coupling that can rewrite an unhoisted frame is the quaternion hemisphere walk:**

```2883:2886:src/autoanim_gnm/commercial_multiview.py
    for frame in range(1, frames):
        dots = np.sum(local[frame] * local[frame - 1], axis=1)
        local[frame, dots < 0.0] *= -1.0
```

A re-aimed Spine/clavicle/arm on a hoisted frame can flip the sign of the **next** unhoisted frame (same rotation, different bits). That is the D7b seven-normalisations lesson applied across time.

Pass B is the other whole-take coupling: a newly rejected run slers **every** interior frame, including unhoisted ones (see 6).

**Change:** after the re-aim, match each rewritten quaternion to the **delivered** local on that frame (`dot ≥ 0`), or run the sign pass only on joints you wrote, without walking from frame 0. Do not re-run 2883–2886 as a global pass.

---

## 6. Pass B

Omitted second-order path, as the card says: trunk re-aim rotates `UpperChest`, which **displaces** `LeftShoulder`/`RightShoulder` by `ΔR · rest[Shoulder]` (~110 mm). Pass B then scores **local** travel in the new parent frame. Worst case is a **hoisted | unhoisted boundary**: two different parent worlds and two different origins in one 26.67° step. The `atan(h/L)` bound (10.2 / 6.6°) ignores both.

**Registration is honest, not adequate as a merge patch.** “If the set changes, list frames and scope B2 to the outside” lets a slerp rewrite unhoisted frames and still merge. That is a hole.

**Change:** accepted-set change is a **hold**, not an auto-scope. Merge on B2 as written (set identical). If the set moves, stop, publish the frames, and only then decide a scoped clause — do not pre-authorise it.

---

## 7. New constants / MAMMA

**No.** Envelope unchanged (`0.30 / 0.08 / 2 / 0.08`). Window 0. Nothing selected on MAMMA. B3 is SAM2. O3 does not select. 0.5 mm is an existing **report** cut; using it as a re-aim switch would be a new constant — don’t (3b).

---

## 8. Missing, and dispatch

**Dispatch**, after the card is tightened. The defect is real (two hoist recoveries at 0.0 mm; ray miss 0.0 from the pre-hoist origin; same signature on all six oracle bodies). The D2c mechanism is the right one. Do not block.

Patch before the agent starts:

1. **Do not call pass C.** Named joints only; feet/toes/legs/root/Hips are read-only after projection.  
2. **Hemisphere:** 2883–2886 must not be allowed to flip unhoisted frames (B2).  
3. **O3:** predict the statistic `groups_mm` actually is (median; likely flat), plus a hoisted-frame / p95 companion. Do not move the D3 band.  
4. **B1** is the closure quantity; merge it only with B2/B4/tripwire.  
5. **Pass B set-change** is a hold, not a pre-scoped B2 exception.  
6. **Oracle vs delivery Head:** the D3 oracle does **not** pass `head_world_rotations`, so `Head`/`Neck`/`eyes` inherit the torso and **must** move with the trunk re-aim. The sentence that Head rotation is the absolute solve is delivery-only. If the agent freezes Head on the oracle, O1’s trunk ray and O4 will fight.  
7. **Hands/fingers/eyes:** B2’s “from `Spine` up and out the arms” must list them as allowed-to-move on hoisted frames, or an agent will leave wrist/hand world stale.  
8. **Tripwire:** “hoist forced to zero” means **re-solve still runs** on the pre-projection root, not “skip projection” (that also drops the foot lock).

None of that is a reason to skip the step. It is a reason not to hand an agent the pass-C loop and a “scope B2 if pass B moves” blank cheque.

---

## What changed in the card

1. The re-solve is a DEDICATED block over the root-dependent chain; never a re-run of pass C (feet/toes/legs) or pass A (the root line). The projection's root, contacts and foot/toe locals are read-only after it.
2. The global hemisphere pass is not re-run; each rewritten local is signed against the delivered local of its own frame.
3. O3 predicts the statistic `groups_mm` is: the median may hold; the hoisted-frame median and p95 worse by ~3–7 mm; legs identical. Band not moved.
4. B1 (and O1) relabelled as the CLOSURE quantity, in the merge rule only with B2, B4 and the tripwire.
5. A pass-B accepted-set change is a HOLD (the step stops, the frames are published), not a pre-scoped B2 exception; the skip-pass-B degenerate is recorded as indistinguishable on this take.
6. The oracle's Head/Neck/eyes inherit the trunk (no head solve there) and must move; hands, fingers and eyes listed as allowed to move.
7. The tripwire's meaning fixed: the re-solve still runs, on the pre-projection root, with the locks in place.
8. Two more must-fails: the lock-without-correction degenerate (B2 and B4 catch it) and the re-derived-root degenerate; 0.5 mm stays a report cut, never control flow.
