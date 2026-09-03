# D3 — one rest skeleton per performer, carried end to end

Branch `ladder/D3`, from `1035ece`, 2026-09-03. Worktree
`.claude/worktrees/ladder-D3`; every command below runs with `PYTHONPATH=$PWD/src`,
and the gate records the `autoanim_gnm.__file__` it actually imported.

---

## 0. Pre-registration — written and committed BEFORE the rebuild ran

Committed before `scripts/build_commercial_multiview_comparison.py` was run on this
branch and before any figure in §§4–6 existed. Refuted predictions are marked **REFUTED**
in §5 and are not rewritten.

### 0.1 What is being changed, in one sentence

`positions_to_body_track` is given a rest skeleton, stamps it on the `BodyTrack`, and
every consumer — forward kinematics, validation, the ground projection, the glTF
exporter, the unified character, the verify script and the instruments — rebuilds **that**
body instead of reconstructing the canonical one from the joint names.

### 0.2 The finding this closes, re-measured on this branch

The delivery carried two skeletons. Re-measured here, at commit `1035ece`, on
`artifacts/commercial-multiview-soma77/subject-00.glb`: the GLB's joint world positions
sit **81–195 mm** from `forward_kinematics_positions` of the same track on
`DETAILED_HUMANOID` (Hips 107, LeftUpperLeg 92, LeftFoot 81, LeftShoulder 194,
LeftHand 189, Head 140; median 131 mm over 55 joints), because
`export_animated_body_glb` wrote the MPFB asset's `local_rest_matrices` as the node
translations. Confirmed twice: by a hand-rolled glTF reader and by Blender 4.2's own
importer, whose bone heads agree with the reader to the millimetre.

### 0.3 The bands, and what each cannot see

**CLOSURE — the discriminating band.**
The delivered GLB, parsed and forward-kinematicked node by node, must reproduce
`forward_kinematics_positions` on the track's own rest to **≤ 1e-4 m**, max over every
frame and every joint, on both delivered subjects.
*Prediction:* it closes by construction — the exporter now writes
`t_local[j] = inv(alignment[p]·rest_world[p])·rest_track[j]`, which is an identity, not a
fit — so the residual should be float32 rounding, **order 1e-6 m or smaller**.
*Must-fail:* the pre-D3 exporter (`git show 1035ece:src/autoanim_gnm/body_export.py`,
loaded as a scratch module) on the same track must fail the band by orders of magnitude.
*Prediction:* **≥ 50 mm median** on the sized delivery — larger than the 131 mm measured
on the canonical track is possible in either direction, so only the ≥ 50 mm floor is
pre-registered.
*Also asserted:* the delivered GLB's rest bone spans (shoulder, hip, thigh, shin,
root→neck) equal the track's rest to 1e-6 m.
*Blind to:* whether the rest itself is right. Closure says the two halves agree, not
that either agrees with the performer.

**EXACT-SKELETON ORACLE (synthetic).**
Per-bone scale channels perturbed independently on the canonical rest (every limb
length, both spans, each spine segment, the upper-leg vertical), factors drawn in
0.7–1.3, ≥ 5 seeds. The truth motion is posed through our own forward kinematics on the
perturbed rest; its landmarks go to the converter **with that skeleton**; the track is
exported through the **real** `export_animated_body_glb` and the **real** MPFB asset; the
GLB is parsed and forward-kinematicked and compared to the posed truth joints.
*Bands:* legs **0.00 mm**; arms **≤ 0.5 mm** (D2's known clavicle residual); root exact.
*Must-fail:* the same skeleton arriving canonical downstream (FK and export on
`DETAILED_HUMANOID`) must read **tens of mm**.
*Reported, not banded:* a global-scale-only skeleton — that is D5's control.
*Deviation from the plan card, recorded:* the card says "an MHR subject". pymomentum is
installed in neither python on this machine, and the property the card wants — exact
truth with independent scale channels — is met without it, by perturbing our own rest
directly. What is lost is MHR's particular parameterisation; what is kept is exactness.
*Blind to:* anything the converter cannot represent. The truth motion is deliberately
one the converter *can* represent (one torso frame for Spine/Chest/UpperChest, hands
following the forearm, toes following the foot), because otherwise the band would fail
for a reason that is not D3.

**CANONICAL BODY UNCHANGED.**
`DETAILED_HUMANOID` passed explicitly through the new plumbing must reproduce D2's
committed figures from `artifacts/compare/d2-clavicle/gate-d2c.json`: the canonical round
trip **arms 0.55 / 0.08 mm**, **legs 0.00**, **torso 0.00**, with the clavicle reject held
as pass-through exactly as D2c's P7 rule does.
*And:* the delivered **track rotations** for a canonical skeleton must be **bit-identical**
to `artifacts/commercial-multiview-soma77/subject-0X.body-track.npz`
(`local_rotations_xyzw`, `root_translation_m`). The converter must be byte-stable under a
canonical rest; if this fails the exact line is found and named, not excused.

**SCHEMA must-fails** (as tests): a rest of the wrong shape, a non-finite rest and a
non-zero Root offset are all rejected; legacy JSON with no rest field loads as canonical;
`as_dict`/`from_dict` round-trips bit-identically for both canonical and sized tracks; a
sized track cannot be validated against the canonical skeleton.

### 0.4 Delivered figures — all REPORTED, none banded

Both subjects, before → after, each with its reference named. Predictions written here
before the rebuild:

| figure | before (D2c, committed) | prediction |
|---|---|---|
| round trip, canonical, arms | 0.55 / 0.08 mm | unchanged to 0.01 mm |
| round trip, canonical, legs / torso | 0.00 | unchanged, exactly 0.00 |
| delivered vs our own capture, arms (root-relative) | canonical 50.79 / 30.50; sized replay 27.19 / 23.75 | the delivery becomes the sized arm: **25–30 / 21–27 mm** |
| … legs | canonical 32.48 / 30.03; sized 19.54 / 15.96 | **17–23 / 13–19 mm** |
| … torso (the neck) | canonical 18.87 / 73.82; sized 15.51 / 23.78 | **IMPROVES on the sized replay on both subjects** — the torso-drop fold is exactly this term, and the uncorrected version showed as a neck sitting at 80.00 mm |
| rung 11 vs MAMMA `pred_joints`, all joints | canonical 71.30 / 67.92; sized replay 72.27 / — | within **±10 mm** of the sized replay; direction not predicted |
| ground hoist | 83.01 / 49.06 mm (D2b) | **FALLS.** The performers' shins and thighs are ~30 mm shorter than canonical each, so the rig no longer stands too tall in its own contacts |
| foot contact counts | [37, 60] / [7, 27] | change; no prediction of direction |
| clavicle chain frames over the 800 °/s ceiling | 0 (D2c) | **0**, by construction |
| facing forward-dots | committed `artifacts/compare/facing-location.json` | the medians MOVE — a sized rig has different rotations, and the facing instrument reads rotations — but every median stays within **0.02** of committed and every handedness triple-product sign is **unchanged**. Nothing here touches handedness |
| I6 silhouette median IoU | 0.5207 (D2c) | **pre-registered both ways**, below |

**The silhouette, both ways, with a rationale each.**
*It should rise:* the mesh's hips now sit on the captured hips rather than 72 mm below
them — D2b moved the code skeleton's leg roots onto the captured hips and, by the same
vector, moved the *mesh's* leg roots away from them, which is the competing explanation
for D2b's 8-of-8 fall. D3 removes that gap by construction.
*It should fall:* the mesh is skinned to the MPFB asset's bind pose and its weights are
unchanged. Moving every joint by 80–195 mm deforms it far outside anything those weights
were solved for; the shoulder cap and the hip region will stretch. Re-skinning is D6.
*Therefore:* the root's own share should shrink; the whole-person figure is genuinely
unknown, and whichever way it goes it is reported and not used to select anything.
The ORACLE arm (MAMMA's own mesh through our rasteriser) must be **bit-identical**
between runs — it reads none of our track — and that is the proof that reusing
`artifacts/compare/i6`'s caches changed nothing.

### 0.5 Statistics, references, and the rules held

Block bootstrap (moving block 15, 2000 draws, fixed seed, both arms on identical draws)
wherever a take statistic is quoted; lag-1 autocorrelation on this take is 0.99.
MAMMA's `body_id-00` is our subject **1**, resolved through `tools/head/subject_map.py`
by 3D pelvis agreement, never by index. The MAMMA arm reports and selects nothing.
Figures on different references never share a key prefix.

