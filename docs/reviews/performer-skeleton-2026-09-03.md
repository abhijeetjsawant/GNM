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


---

## 1. What shipped

Commit `fe1a17b` carried the plumbing; this commit carries the gate, the instruments, the
tests and this review. One rest skeleton per performer, built from that performer's own
triangulated limb lengths (`src/autoanim_gnm/performer_skeleton.py`, the sizing that was
`tools/head/sized_skeleton.py` with one derivation folded in), is stamped on the
`BodyTrack` (`rest_translations_m`, float64, schema `autoanim.body-track/1.3` +
`autoanim.humanoid-skeleton/1.1`; legacy files without the field load as canonical) and
read from there by every consumer: `positions_to_body_track(skeleton=...)`,
`forward_kinematics_positions` through `skeleton_for_track`, `validate_body_track`,
`project_generated_foot_contacts`, `export_animated_body_glb`, `unified_gltf`, the
verify script, and every instrument that reads the delivery (`skeleton_for_track_dict`
on the JSON, the rest also in the `.npz` and the mapping `.npz`). The build script gained
`--rest-skeleton {performer,canonical}`; `canonical` is the gate's instrument arm and never
the delivery. The consumer manifest is in `gate.json` (`consumer_manifest`, four bins,
every grep hit).

**How this step was run, recorded as a deviation.** The plan says one Opus agent per
step. The first Opus agent committed `fe1a17b`, ran the rebuild and wrote the gate script,
then was cut by an API 500; two Opus resumes and a fresh Opus agent were cut by API 529
overloads before landing any work (four failures, no partial state); a Fable agent was cut
twice the same way. The remaining work (finishing and running the gate, the instrument
edits, the tests, the extractor, this review) was done by the coordinating session
itself, on Claude Fable, in the same worktree under the same rules. Nothing about the
gate's design changed hands; the bands are the ones section 0 fixed.

## 2. The two-skeleton finding, re-measured

Section 0.2 stands. On this branch the gate's own reader gives the same picture for the
pre-D3 exporter on the *sized* track (the must-fail arm): median 96.49 / 92.59 mm across
55 joints, Hips 106.9, LeftUpperLeg 101.6, LeftFoot 107.1, LeftShoulder 157.8, LeftHand
121.9, Head 151.6 on subject 0. The exporter was composing the converter's world rotations
onto the MPFB asset's rest frames and writing the asset's rest translations, so every
delivered GLB carried the asset's body while every joint instrument scored the code rig.

## 3. The mechanism, in one identity

`_compose_rest_and_delta` leaves each joint's world rotation as
`target_world[j] · alignment[j] · rest_world[j]`, where `rest_world` is the asset's rest
frame and `target_world` is the track's. The child's world position is the parent's plus
that parent world rotation applied to the node translation. For the GLB to reproduce the
track's own FK, `pos[child] − pos[parent]` must equal `target_world[parent] · rest[child]`,
so the node translation must be
`t_local[child] = (alignment[parent] · rest_world[parent])⁻¹ · rest[child]`
and the root translation must be the track's root plus the track's own Root offset (zero),
not the asset's. That is an identity, not a fit; the closure band measures float32
rounding of it. Vertices, weights and inverse bind matrices are untouched, so the mesh
follows the joints through the weights it already has.

## 4. The gate — `artifacts/compare/d3-skeleton/gate.json`, 16 rows

| band | subject 0 | subject 1 | verdict |
|---|---|---|---|
| **CLOSURE.** delivered GLB FK = the track's own FK, max over 150 frames × 55 joints, band 1e-4 m | 5.0e-7 m | 5.8e-7 m | PASS |
| **MUST-FAIL.** the pre-D3 exporter on the same track (floor 50 mm) | 96.49 mm median | 92.59 mm | PASS (fails the band) |
| **CLOSURE.** GLB rest bone lengths = the track's | 1.1e-16 m | 5.6e-17 m | PASS |
| **EXACT-SKELETON ORACLE.** 6 perturbed bodies, GLB vs posed truth, legs 0.00 | 0.00 on every seed | | PASS (legs) |
| **EXACT-SKELETON ORACLE.** arms ≤ 0.5 mm | worst **0.69 mm** (seed 20260904); 0.43, 0.69, 0.52, 0.47, 0.43, 0.60 | | **FAIL, kept** — §5 |
| **MUST-FAIL.** the same body arriving canonical downstream | best over seeds: arms 31.4, legs 11.4 mm | | PASS (fails) |
| **CANONICAL UNCHANGED.** round trip with the reject as pass-through (P7) | 0.55 / 0.00 / 0.00 | 0.08 / 0.00 / 0.00 | PASS, matches D2c exactly |
| **CANONICAL UNCHANGED.** the whole delivery rebuilt with `--rest-skeleton canonical` vs the pre-D3 delivered build | rotations, root, contacts, positions all BIT-IDENTICAL | same | PASS |
| **REBUILD.** nothing re-extracted or re-detected (8 observation files) | byte-identical | | PASS |
| **PROPAGATION.** the delivered track carries a per-performer rest | schema 1.3 | schema 1.3 | PASS |
| **SAME DENOMINATOR.** the rebuild's triangulation is byte-identical to the delivered build's | true | true | PASS |

**Overall: FAIL on one pre-registered band**, the oracle's arm residual, by 0.19 mm on one
of six bodies. It is not moved.

**A second route on closure, at Sol's pre-merge request.** Blender 4.2's own importer on
the rebuilt `subject-00.glb` (scene fps set to 30 before import) puts every bone head
within **0.005 mm** (median 0.002) of `forward_kinematics_positions` on the track's own
rest at frame 0, over all 55 joints. The gate's reader and the consumer that renders the
silhouette agree, so the silhouette is scoring a mesh in the place closure describes.

**Temporal, reported (added at review):** clavicles over the 800°/s ceiling 0 → 0 on both
subjects (the D2c reject holds); the arm joints below the clavicle, which no reject bounds,
14 → 21 on subject 0 and 13 → 12 on subject 1 (`gate.json` → `temporal`).

## 5. The pre-registered expectations, and whether each was met

| | expectation (§0) | outcome |
|---|---|---|
| closure | float32 rounding, ≤ ~1e-6 m | **MET**, 5.0e-7 / 5.8e-7 m |
| pre-D3 exporter must-fail | ≥ 50 mm median | **MET**, 96 / 93 mm |
| oracle legs 0.00, arms ≤ 0.5 | | legs **MET** on every seed; arms **REFUTED on one seed (0.69)** and near the band on another (0.60). Attributed, not excused: the residual is identical on shoulder, elbow and wrist of one side (a rigid offset of the whole arm, i.e. the arm ROOT), identical with the clavicle reject held as pass-through, and proportional to the drawn shoulder half-span factor — 0.43 mm at 0.78×, 0.69 mm at 1.26×. It is D2's known clavicle round-trip cost (0.55 mm on the canonical lever) on a longer lever. The band was written from the canonical lever and the fixture draws spans up to 1.3× it. **The band stays; the record says why.** |
| canonical round trip reproduces D2c | 0.55 / 0.08, legs and torso 0.00 | **MET** exactly |
| canonical delivery bit-identical | | **MET**: every array of both subjects byte-identical to `artifacts/commercial-multiview-soma77` |
| delivered vs our capture, arms | 25–30 / 21–27 mm | **MET**: 50.79 → **25.55**, 30.50 → **21.84** |
| … legs | 17–23 / 13–19 | **MET**: 32.48 → **19.54**, 30.03 → **15.96** |
| … torso (the neck) | improves on the sized REPLAY on both (15.51 / 23.78) | **HALF MET**: 18.87 → **14.46** (below the replay's 15.51) and 73.82 → **26.21** (far below canonical, but ABOVE the replay's 23.78). The torso-drop fold explains the fall from 73.82; the remaining 2.4 mm against the replay is not attributed. |
| rung 11 vs MAMMA, all 15 joints | within ±10 mm of the sized replay (72.27 / 75.52) | **REFUTED, in the better direction**: **47.08 / 45.83** mm (canonical delivered was 71.30 / 67.82). The replay put canonical-solved rotations on a sized body; the delivery re-solves on its own rest (the clavicle from its own origin) and hoists on its own legs. Which of those two carries the 25–30 mm is not measured here. MAMMA reports; it selected nothing. |
| ground hoist | FALLS | **MET**: median 83.0 → **72.7** mm, 49.1 → **35.5** (measured without the head and toe solves, both arms the same way) |
| foot contacts | change, no direction | [33, 48] → [33, 49]; [4, 20] → [2, 20] |
| clavicle frames over 800°/s | 0 by construction | **MET at the clavicles** (0 and 0 on both builds). The joints BELOW the clavicle, which no reject bounds, read 14 → **21** on subject 0 and 13 → 12 on subject 1 (RightUpperArm 8 → 12; median clavicle step 1.82 → 1.89° and 1.70 → 2.17°). The sized rig's lever is shorter (shoulder half-span 173 mm against 270), so the same landmark wander becomes more angle: D2's own finding, arriving on the sized body. Reported, not banded. |
| facing | medians move ≤ 0.02, handedness signs unchanged | **MET**: no forward-dot median moved by more than 0.02; all 14 handedness signs unchanged |
| silhouette | pre-registered both ways | **ROSE, 8 of 8 cells**: median IoU **0.5207 → 0.6266**; subject 0 0.484 → 0.605, subject 1 0.538 → 0.627; recall 0.570 → 0.686 and 0.606 → 0.746, precision 0.804 → 0.839 and 0.831 → 0.815; the ORACLE arm bit-identical between runs (max difference 0.0000, oracle 0.855). The "it should rise" rationale won outright; the "it should fall" rationale (the mesh deforming under its canonical weights) did not dominate anywhere. |

**What the silhouette result says about D2b.** D2b's fall (0.585 → 0.519, 8 of 8 cells)
was pre-registered here as possibly the mesh being moved OFF its own skeleton by the root
fix. With one skeleton the same instrument rises past where it stood before D2 (0.585) on
every cell. The joint instruments and the photographs now agree in direction for the first
time in this lane. This is the strongest single piece of evidence the two-skeleton reading
is right; it is still agreement between two instruments, not accuracy.

## 6. Tests

88 in the set the brief names plus the two new files: **87 pass, 1 fails**.
`tests/test_body_export.py::test_export_animated_body_glb_is_one_skin_one_timeline_and_hash_bound`
asserts the OLD exporter: it expects the GLB's root translation to be the asset's Root rest
`(0, 0.8, 0)` plus the track root, and D3 writes the track's own Root offset (zero) plus the
track root, which is what closure requires. Not edited (the brief forbids it). The one-line
fix: expect `root_translation == track.root_translation_m` (the desired array without the
`0.8`). New: `tests/test_performer_skeleton_schema.py` (11: schema must-fails, legacy
loads canonical, bit-identical serialisation, a sized track validated on its own rest and
rejected on the canonical one, the sizing as a derivation) and
`tests/test_d3_export_closure.py` (the exported GLB reproduces a sized track's own FK to
1e-4 m through the real asset, and differs from the canonical FK by centimetres). Every
new test asserts it imported the worktree's package. The provenance audit runs CLEAN on the
branch with the new module reachable (no literal arrives with the sizing).

## 7. What this is blind to

* **Whether the rest is right.** Closure proves the delivery carries ONE body, not that
  the body is the performer's. The sizing is `estimate_limb_lengths_m` on our own
  capture; the shoulder construction scales the clavicle chain uniformly and lowers the
  arm root ~36 mm where X-only would not, the UpperLeg vertical is canonical, the spine
  keeps canonical ratios. All of that is **D5's** (fitted scale channels, exact synthetic
  recovery as the primary gate).
* **The mesh.** Vertices, weights and bind matrices are the asset's; the skin now deforms
  outside the pose its bind was solved for, at the shoulder cap and the hips especially.
  The silhouette rose anyway. Re-skinning is **D6's**.
* **The pelvis frame.** Hips, Spine, Chest and UpperChest still share one torso frame; on a
  bent performer the root offset rides the lean. SOMA-77's unmapped pelvis root and lower
  spine make it observable. **Its own pose step**, not bundled here.
* **The oracle's motion** is one the converter can represent by construction; it cannot
  see what the converter cannot express.
* **The synthetic subject is not MHR.** pymomentum is installed nowhere here; the card's
  "MHR subject" became independently perturbed per-bone channels on our own rest. Exactness
  and independence are kept; MHR's parameterisation is not exercised.
* **Rung 11 and the silhouette** are agreement with instruments. No ground truth.

## 8. Handoffs and stale strings

* **D5:** the shoulder construction (uniform vs X-only), the UpperLeg vertical, the spine
  ratios, the neck's remaining 2.4 mm on subject 1, and the arm joints below the clavicle
  on the shorter lever (14 → 21 frames over the ceiling on subject 0).
* **D6:** the binding; the mesh is stretched onto its skeleton, not re-skinned.
* **Pelvis frame:** a converter/pose step with its own gate.
* **`tools/head/sized_skeleton.py` now re-exports the drop-corrected sizing.** D2's frozen
  gates (`d2_clavicle_gate.py`, `d2c_clavicle_temporal_gate.py`, its `_baseline_check`)
  build their SIZED arms through it, so re-running them will not reproduce their committed
  sized figures; their canonical arms are unaffected. The committed reports stand as the
  record; the ladder's `regenerate` strings for those rungs carry that caveat.
* **`tests/test_body_export.py` is one of the user's uncommitted files** (modified in the
  main tree). The one-line fix in §6 is reported, not applied.
* **Stale:** `docs/parity-board.html` stage 06 says "the rig is one fixed body (540 mm
  shoulders)"; `ladder.py` rung 6 says "Nothing in the delivery path"; the scoreboard's
  `sized` arm label on pre-D3 reports means a replay, on D3 reports the delivered body (the
  report now says which); `tests/test_body_export.py:145` as above.

## 9. In plain language

Until today the character we delivered was built on one skeleton and measured on another.
The code that works out how each joint turns, and every check that scores it, used a
skeleton 54 cm across the shoulders with the hip joints 8 cm below the pelvis. The file we
actually hand over used the body model's own skeleton, 34 cm across, with different bone
directions and lengths. So every "the joints are better placed" result was true of a
skeleton nobody could see, and the one check that looks at the photographs was scoring a
body that had been quietly moved about ten centimetres off it. That is why the last step
made the joints better and the outline worse at the same time.

Now there is one skeleton per performer, sized from that performer's own measured limbs,
and it travels with the animation all the way into the delivered file. The check that
proves it: reading the joints straight out of the delivered file gives the same positions
as the code's own arithmetic, to under a thousandth of a millimetre, where the old file
missed by nine centimetres. The outline check that fell last time rose in every one of the
eight camera-and-performer cells, from 0.52 to 0.63 overlap; the reference fitter's joints
agree better too, from about 7 cm to under 5. One synthetic check misses its band by a fifth
of a millimetre on one of six invented bodies, for a known reason on a longer collarbone;
it is written down as a miss rather than moved. What is still not right is the shape of the
body itself, which is the next step, and the skin, which is the one after.
