# Grok 4.6 (Cursor, ask mode) review of the D9b MERGE — 2026-09-07, in Sol's place

Brief: the card as dispatched, the full src diff, the merge_rule block of gate.json, review sections 2, 5, 6, 7, 8, 9, eight questions. Verdict: MERGE stands. Verified against the source before adoption: the re-solve block calls only the two lifted helpers and pass B, never the leg/foot/toe loops or the root line; `_world_for_bone` returns a new array so recording `torso_world` from the helper's return is the same value at the same rounding; the only code between the moved hand writes and their old site is the landmark-to-landmark leg loop. Adopted at the merge: the two structural test pins re-pinned in place (the call count 2 -> 4 and the `_joint_origin(` pin pointed at the helpers), the skip-second-pass-B hole recorded as untestable on this take, and the close-out read on the gate lines with worst_arms_mm 2.70 expected.

---

**Verdict: MERGE stands**, on the src in the brief. This workspace still has the pre-D9b converter (`return projected` at the projection call), so every line citation below is to **that diff** (and to current HEAD where the lift is compared). I have not re-read `artifacts/compare/d9b-hoist/*`; conjunct numbers are taken as the review reports them.

The block is the card: projection once, root/contacts/foot locks read-only, dedicated re-solve, pass B rerun, per-frame hemisphere sign, no second projection, no pass-A root line, no pass-C legs/feet/toes.

---

### 1. Does the re-solve do exactly what the card names?

**Yes**, with one listed omission (fingers as locals) that the card already allowed as follows.

| Card | Diff |
|---|---|
| Projection once | Single `project_generated_foot_contacts(...)`; then the D9b block; **no second call** |
| Root / contacts / foot locks untouched | `return replace(projected, local_rotations_xyzw=resolved)` — only locals replaced; `projected.root_translation_m` and `foot_contacts` stay |
| Legs / feet / toes never rewritten | Re-solve calls `_aim_trunk_neck_and_clavicles` then `_aim_arms_and_hands` only. The pass-C leg/foot/toe loops are **not** in the tail |
| Pass B rerun on re-aimed clavicles | After the trunk/clavicle loop: `for clavicle in ("LeftShoulder", "RightShoulder"):` … `_reachable_clavicle_sequence` … write rejected locals/world — same shape as the existing pass-B loop (~2752–2763 on HEAD) |
| Sign pass, not the walk from frame 0 | `flip = np.sum(rewritten * delivered_local[:, joint_index], axis=1) < 0.0` then `np.where(flip, -rewritten, rewritten)`. The global `for frame in range(1, frames): local[frame, dots < 0] *= -1` stays **before** the projection and is not repeated |
| Every frame | `for frame in range(frames):` on both aim helpers; hoist is not a gate. `hoist = float64(projected.root) - float64(track.root)`; `hoisted_root = root_translation + hoist` |

`_ROOT_DEPENDENT_JOINTS` is Spine/Chest/UpperChest, Neck, Head, both eyes, both shoulders, four arm bones, both hands. Absent by comment: Root, Hips, legs, feet, toes, **fingers** (rest curl). Eyes/hands are in the rewrite list so world follow cannot go stale; fingers keep locals and follow the hand at FK.

`replace` re-runs `BodyTrack.__post_init__` → `validate_body_track`, which scores contact lock on foot/toe **positions**. Those chains do not include the trunk, so a trunk re-aim cannot break the 1e-5 m assertion. That matches §5.2: dropping `roots += correction` while keeping contacts fails there (`foot contact moved` vs 1e-5 m).

---

### 2. Verbatim lift, and what the tripwire cannot see

Pass A’s tail and pass C’s arm half match HEAD (~2617–2745 and ~2770–2833) **except**:

1. **`torso_world_by_frame` is taken from the helper return**, after the clavicle `_set_world`s, not after the eyes. On HEAD it is assigned immediately after the eye loop (~2678). The helper comment is right about `_set_world`: `np.asarray(target_world)` on an existing float64 array is a view, then `normalized /= …` mutates **that** quaternion. Clavicle targets come from `_world_for_bone`, which returns `output / np.linalg.norm(output)` — a **new** array — so clavicle `_set_world` does not retouch `torso_world`. Same value, same rounding. A recording-time bug would only matter if feet used `torso_world`; on this take the toe branch supplies `foot_world`.

2. **Hands move** with the arms (Q3).

3. **`shoulder_across` is computed inside the helper**; pass A drops it. Arithmetic is the same.

The old-src / new-src zero-hoist tripwire is the right control for “same helper, both callers.” It **cannot** see:

- Any bug that only exists when `hoist ≠ 0` (always aiming from the pre-projection root; that is B1/O1).
- Pass B actually substituting (inert on this take; eight all-accept masks).
- `torso_world_by_frame` timing, if every foot uses the toe branch.
- A future in-process “re-solve changed nothing” assert (both callers share the helper).
- Aligned O3 / photographs (different instruments).

Zero-hoist using `root_translation + hoist` rather than the float32 track root is the right tripwire denominator.

---

### 3. Hands before the legs — does anything between read an arm world?

**Not in this diff.** New pass C: `_aim_arms_and_hands` → **leg loop** → feet (`torso_world_by_frame` + ankle/knee/toes) → toes → fingers.

The only code between the new hand writes and the old site is the landmark-to-landmark leg loop (`left_knee - left_hip`, parent Hips / upper leg). It does not read `world[UpperArm/LowerArm/Hand]`.

Fingers still run **after** hands and compose `world[parent] * rest curl`. Reorder is free **as long as that stays true**. The helper comment states the invariant.

---

### 4. Candidate-controlled conjuncts and cheats that still merge

**B1 and O1 are the closure.** The candidate sets ray miss / excess-over-floor to ~0. That is allowed only because **tripwire ∧ B2 ∧ B4 ∧ O2** fail the undoing-the-plant cheats.

| Degenerate | Merge rule |
|---|---|
| D8c as-is | Fails B1 |
| `maximum_root_correction_m=0` | Fails B4 (and B2 root); penetration is 0 either way |
| Lock, skip `roots += correction` | Shipping path refuses (`validate_body_track`); forced, fails B2/B4 |
| Arms only, no trunk/clavicles | Fails B1 on neck/shoulders (computed, not built — stated) |
| Re-aim then re-derive root | Excluded; would fail B2/B4 |
| **Skip the second pass B after per-frame clavicle re-aim** | **Passes this take.** Pass B rejects nothing; B2’s accepted-set check does not prove the rerun. Card limitation; not a B1 cheat. |

No measured degenerate **passes the full merge rule** except that skip-B hole, which does not zero B1.

`gate.json` still lists hygiene and B3 as `outstanding` / `SEE THAT FILE` while `conjuncts_decided_here` is PASS. Read those two files; do not merge on the outstanding list.

---

### 5. O3 “median may hold” failed — attribution and close-out

**Attribution is right.** `retarget_cost.score` subtracts the leg-root midpoint from FK and the hip midpoint from the reference (translation only). A bone aimed from the hoisted origin at the true target, then shifted back by that alignment, misses by **h⊥**. Aligned arms 1.17 → 2.70 whole-take, 1.18 → 8.88 on hoisted frames, legs 0.07 identical, **absolute** arms better, is that geometry — two gauges, both honest.

The “median may hold” line was the statistic (`groups_mm` = median over joint-frames; hoisted frames a minority), not a claim that the fix is wrong. Hoisted-frame medians ~8.9 mm were enough to move the whole-take median. **Do not move the 0.5 mm D3 band** in this commit (already a standing fail; re-pinning with the candidate is the clavicle-round-trip mistake). O3 stays report.

At close-out, **read the gate lines**, not the verdict:

- Expect `worst_arms_mm` **2.70** (was 1.17); legs 0.05–0.07.
- Treat that as predicted, not a regression.
- Canonical / round-trip pins frozen at D2c/D3 may fire if those fixtures plant feet — owed re-pin, not a D9b revert.
- Do not “fix” the aligned oracle in the same pass.

---

### 6. B1: farther raw placement, zero excess — regression the band hides?

**The attribution is sound. It is not a merge-blocking regression of this defect.**

O1 / B1 ray miss from the **delivered** origin is 0.0003–0.0005 mm: the child sits on the ray from the new origin. Raw landmark distance is the **length** error of that chain. Trunk + clavicle re-aim moves `UpperArm`’s origin, so the D9 floor is chained. Excess-over-**own**-floor going to 0 and a minority of frames reading p95 +2.5–7.2 mm vs the old placement can both be true.

What the band cannot see: “this elbow is farther from the landmark than D8c’s elbow.” That is a different quantity (D5 length + new parent origin), not leftover hoist-ray miss. B3 is the dual that is not in the aim loop; it stayed inside ±0.001 IoU. Record the farther-out tail as a **chained-floor caveat**, not as B1 failure. Hips/knees/ankles 0.0000 mm keeps the step from quietly rewriting legs.

---

### 7. The two old tests

**Edit them at merge. Do not leave CI red, and do not delete the properties.**

`test_a_reachable_take_is_accepted_whole_and_the_track_is_untouched` (`len(accepted) == 2`) is a **call-count pin**, not the inert-reject property. After D9b the wrapper sees **four** calls (L/R, then L/R). Keep: both clavicles, every mask all-true, bit-identical to passthrough. Change 2 → 4 (or assert two pairs). The pop test’s `accepted["call0"]` is still first-pass left; optionally assert the second pair also rejects that frame.

`test_no_constant_arrived_with_the_fix` slices `positions_to_body_track` and requires `_joint_origin(`. That call now lives in the helpers. Point the pin at `_aim_trunk_neck_and_clavicles` / `_aim_arms_and_hands`, keep `0.72 * torso_up` absent from the **module**, keep `torso_up` on the hips line. The new-file re-pins are the right property; the old files should say the same so a later revert of the helpers fails in one place.

---

### 8. Block vs open

**Nothing in the src should block MERGE**, given hygiene/B3 files actually PASS as §8 claims.

**Record as open / handoff (do not silently drop):**

- D9-legs: 4.6–9.2 mm ray miss on hoisted frames; legs still landmark-to-landmark.
- Contact model’s absolute cost on exact truth (p95 10–14 mm, 34–67 frames) — projection’s step, not this one.
- D3 aligned gauge cannot see a root move — instrument-debt; close-out will print 2.70.
- Skip-second-pass-B untested on this take (masks + tripwire only).
- Two structural test pins (edit at merge, §7).
- D8c head-gate 0.01–0.07° vs its own log — D8c close-out debt, not D9b.
- Whole-take medians are 0 by construction; the hoisted-frame cut is the change.
- Predicted canonical-block / round-trip movement if those fixtures plant.

No new constant; envelope `0.30 / 0.08 / 2 / 0.08` unchanged.
